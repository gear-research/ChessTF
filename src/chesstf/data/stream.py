"""Stream-download and filter Lichess monthly PGNs in a single pass.

The ``.pgn.zst`` file is never written to disk — the HTTP response is piped
directly through the zstd decompressor and PGN parser into the filter.

The total game count is fetched from the database.lichess.org index page
before streaming begins so that tqdm can display a meaningful progress bar.

Usage::

    python -m chesstf.data.stream --help
    python -m chesstf.data.stream --year 2023 --months 1 2 3
    python -m chesstf.data.stream --year 2016 --months 6 --max-games 100000
    python -m chesstf.data.stream --year 2023 --months 6 --dry-run
"""

from __future__ import annotations

import argparse
import calendar
import io
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests
import yaml  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from chesstf.data.filter import FilterConfig, FilterStats

_LICHESS_BASE = (
    "https://database.lichess.org/standard/"
    "lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
)
_LICHESS_INDEX = "https://database.lichess.org/"


def fetch_game_count(year: int, month: int) -> int:
    """Return the total game count for *year*/*month* from database.lichess.org.

    Scrapes the standard games index table, which has the structure::

        <td>YYYY - MonthName</td>
        <td class="right">SIZE</td>
        <td class="right">GAME_COUNT</td>

    Raises:
        requests.HTTPError: If the index page request fails.
        ValueError: If no entry is found for the requested year/month.
    """
    month_name = calendar.month_name[month]
    resp = requests.get(_LICHESS_INDEX, timeout=30)
    resp.raise_for_status()
    m = re.search(
        rf"<td>\s*{year}\s*-\s*{month_name}\s*</td>"
        rf"\s*<td[^>]*>[^<]*</td>"       # size cell (skip)
        rf"\s*<td[^>]*>([\d,]+)</td>",   # games cell
        resp.text,
    )
    if m is None:
        raise ValueError(f"No entry for {year}-{month:02d} on {_LICHESS_INDEX}")
    return int(m.group(1).replace(",", ""))


def stream_download_filter(
    year: int,
    month: int,
    output_path: Path,
    config: FilterConfig,
    *,
    max_games: int | None = None,
    dry_run: bool = False,
) -> FilterStats:
    """Download and filter a Lichess monthly PGN in one streaming pass.

    The ``.pgn.zst`` is never written to disk. The total game count is fetched
    from database.lichess.org so tqdm can show a meaningful progress bar.

    Args:
        year: Four-digit year.
        month: Month number 1–12.
        output_path: Destination ``.jsonl`` file (ignored when *dry_run* is True).
        config: Filter parameters (ELO, time control, etc.).
        max_games: Stop once this many games have passed the filter.
        dry_run: Count without writing output; print a summary at the end.

    Returns:
        :class:`~chesstf.data.filter.FilterStats` with accumulated counts.
    """
    import zstandard as zstd

    from chesstf.data.filter import filter_stream

    url = _LICHESS_BASE.format(year=year, month=month)

    print(f"Fetching game count for {year}-{month:02d} ...")
    total: int | None
    try:
        total = fetch_game_count(year, month)
        print(f"  {total:,} total games listed on database.lichess.org")
    except Exception as exc:
        print(f"  Warning: could not fetch game count ({exc}); tqdm will run without total")
        total = None

    print(f"Streaming {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    response.raw.decode_content = True  # handle any transfer-encoding transparently

    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(response.raw, closefd=False) as zst_stream:  # type: ignore[arg-type]
        text_stream: io.TextIOBase = io.TextIOWrapper(
            zst_stream, encoding="utf-8", errors="replace"
        )
        stats = filter_stream(
            text_stream,
            output_path if not dry_run else None,
            config,
            total=total,
            max_games=max_games,
            dry_run=dry_run,
        )

    return stats


# ---------------------------------------------------------------------------
# Config / path helpers (mirrors process.py conventions)
# ---------------------------------------------------------------------------


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "min_elo": 1800,
        "min_base_time_seconds": 480,
        "min_moves": 10,
        "val_fraction": 0.02,
        "result_conditioning": True,
        "data_dir": "data/",
    }
    if config_path is None:
        config_path = Path("configs/data/default.yaml")
    if config_path.exists():
        with open(config_path) as fh:
            loaded = yaml.safe_load(fh) or {}
        defaults.update(loaded)
    return defaults


def _interim_path(data_dir: Path, year: int, month: int) -> Path:
    return data_dir / "interim" / f"games_{year}-{month:02d}.jsonl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m chesstf.data.stream",
        description=(
            "Stream-download and filter Lichess PGNs without writing the "
            "compressed file to disk."
        ),
    )
    parser.add_argument("--year", type=int, required=True, help="Four-digit year")
    parser.add_argument(
        "--months",
        type=int,
        nargs="+",
        required=True,
        choices=range(1, 13),
        metavar="MONTH",
        help="Month(s) 1–12, e.g. --months 1 2 3",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N games pass the filter (default: no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count games without writing output",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to YAML config (default: configs/data/default.yaml)",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)

    from chesstf.data.filter import FilterConfig

    config = FilterConfig.from_dict(cfg)
    data_dir = Path(str(cfg["data_dir"]))

    for month in args.months:
        output = _interim_path(data_dir, args.year, month)
        if output.exists() and not args.force and not args.dry_run:
            print(f"Output already exists, skipping (--force to overwrite): {output}")
            continue

        try:
            stats = stream_download_filter(
                args.year,
                month,
                output,
                config,
                max_games=args.max_games,
                dry_run=args.dry_run,
            )
        except requests.HTTPError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        if not args.dry_run:
            print(f"Filtered {stats.passed:,}/{stats.total:,} games → {output}")


if __name__ == "__main__":
    main()
