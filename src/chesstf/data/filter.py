"""Filter Lichess PGN games by ELO, time control, termination, and move count."""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import chess.pgn
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@dataclass
class FilterConfig:
    """Parameters controlling which games are retained."""

    min_elo: int = 1800
    min_base_time_seconds: int = 480  # 8 minutes
    min_moves: int = 10
    val_fraction: float = 0.02

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FilterConfig:
        return cls(
            min_elo=int(d.get("min_elo", cls.min_elo)),
            min_base_time_seconds=int(d.get("min_base_time_seconds", cls.min_base_time_seconds)),
            min_moves=int(d.get("min_moves", cls.min_moves)),
            val_fraction=float(d.get("val_fraction", cls.val_fraction)),
        )


@dataclass
class FilterStats:
    """Counters accumulated while filtering a PGN file."""

    total: int = 0
    passed: int = 0
    # Rejection reasons
    low_elo: int = 0
    bad_time_control: int = 0
    bad_termination: int = 0
    bad_result: int = 0
    too_few_moves: int = 0

    @property
    def rejected(self) -> int:
        return self.total - self.passed

    def summary(self) -> str:
        """Return a human-readable summary table."""
        lines = [
            f"{'Metric':<30} {'Count':>8} {'%':>7}",
            "-" * 48,
            f"{'Total scanned':<30} {self.total:>8}",
            f"{'Passed':<30} {self.passed:>8} {self._pct(self.passed):>7.1f}%",
            "",
            "Rejection reasons:",
            f"  {'low_elo':<28} {self.low_elo:>8} {self._pct(self.low_elo):>7.1f}%",
            f"  {'bad_time_control':<28} {self.bad_time_control:>8} {self._pct(self.bad_time_control):>7.1f}%",
            f"  {'bad_termination':<28} {self.bad_termination:>8} {self._pct(self.bad_termination):>7.1f}%",
            f"  {'bad_result':<28} {self.bad_result:>8} {self._pct(self.bad_result):>7.1f}%",
            f"  {'too_few_moves':<28} {self.too_few_moves:>8} {self._pct(self.too_few_moves):>7.1f}%",
        ]
        return "\n".join(lines)

    def _pct(self, n: int) -> float:
        return 100.0 * n / self.total if self.total else 0.0


_VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def _passes_filter(
    game: chess.pgn.Game,
    config: FilterConfig,
    stats: FilterStats,
) -> bool:
    """Return True if *game* meets all filter criteria; mutate *stats* counters.

    This function is intentionally exposed for unit testing without disk I/O.
    """
    headers = game.headers

    # --- ELO check ---
    for elo_key in ("WhiteElo", "BlackElo"):
        raw = headers.get(elo_key, "?")
        if raw == "?" or not raw.isdigit():
            stats.low_elo += 1
            return False
        if int(raw) < config.min_elo:
            stats.low_elo += 1
            return False

    # --- Time control check ---
    tc = headers.get("TimeControl", "-")
    if tc in ("-", "?", ""):
        stats.bad_time_control += 1
        return False
    try:
        base_str = tc.split("+")[0]
        base_seconds = int(base_str)
    except (ValueError, IndexError):
        stats.bad_time_control += 1
        return False
    if base_seconds < config.min_base_time_seconds:
        stats.bad_time_control += 1
        return False

    # --- Termination check ---
    if headers.get("Termination", "") != "Normal":
        stats.bad_termination += 1
        return False

    # --- Result check ---
    result = headers.get("Result", "*")
    if result not in _VALID_RESULTS:
        stats.bad_result += 1
        return False

    # --- Move count check ---
    moves = _extract_moves(game)
    if len(moves) < config.min_moves:
        stats.too_few_moves += 1
        return False

    return True


def _extract_moves(game: chess.pgn.Game) -> list[str]:
    """Extract UCI move strings from a parsed game.

    Uses ``move.uci()`` which requires no board state, making this O(n) with
    no intermediate board replay.
    """
    return [move.uci() for move in game.mainline_moves()]


def _iter_games(pgn_source: io.TextIOBase | io.StringIO) -> Iterator[chess.pgn.Game]:
    """Yield parsed games from a PGN text source."""
    while True:
        game = chess.pgn.read_game(pgn_source)  # type: ignore[arg-type]
        if game is None:
            break
        yield game


def _count_games(pgn_path: Path) -> int:
    """Quick pre-pass: count games by counting '[Event ' header lines."""
    count = 0
    if pgn_path.suffix == ".zst":
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        with (
            open(pgn_path, "rb") as raw_fh,
            io.TextIOWrapper(
                dctx.stream_reader(raw_fh), encoding="utf-8", errors="replace"
            ) as text,
        ):
            for line in text:
                if line.startswith("[Event "):
                    count += 1
    else:
        with open(pgn_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("[Event "):
                    count += 1
    return count


def filter_stream(
    text_stream: io.TextIOBase,
    output_path: Path | None,
    config: FilterConfig,
    *,
    total: int | None = None,
    max_games: int | None = None,
    dry_run: bool = False,
) -> FilterStats:
    """Filter games from an already-open PGN text stream.

    The caller is responsible for opening and closing *text_stream*. This
    function is the core of :func:`filter_games` and is exposed separately for
    streaming use cases (e.g. piping from an HTTP response without a temp file).

    Args:
        text_stream: Readable text stream of PGN data.
        output_path: Destination ``.jsonl`` file, or ``None`` to skip writing.
        config: Filtering parameters.
        total: Expected total game count forwarded to tqdm for display.
        max_games: Stop once this many games have passed the filter.
        dry_run: When True, print a summary table instead of writing output.

    Returns:
        A :class:`FilterStats` instance with accumulated counts.
    """
    stats = FilterStats()

    with contextlib.ExitStack() as stack:
        if output_path is not None and not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_fh = stack.enter_context(open(output_path, "w", encoding="utf-8"))
        else:
            out_fh = None

        for game in tqdm(_iter_games(text_stream), "Filtering games", total=total):
            stats.total += 1
            if not _passes_filter(game, config, stats):
                continue
            stats.passed += 1
            if out_fh is not None:
                moves = _extract_moves(game)
                record = {
                    "result": game.headers.get("Result"),
                    "white_elo": int(game.headers.get("WhiteElo", 0)),
                    "black_elo": int(game.headers.get("BlackElo", 0)),
                    "moves": moves,
                }
                out_fh.write(json.dumps(record) + "\n")
            if max_games is not None and stats.passed >= max_games:
                break

    if dry_run:
        print(stats.summary())

    return stats


def filter_games(
    pgn_path: Path,
    output_path: Path,
    config: FilterConfig,
    *,
    dry_run: bool = False,
) -> FilterStats:
    """Filter games from a (potentially zstd-compressed) PGN file.

    Args:
        pgn_path: Path to the source PGN or ``.pgn.zst`` file.
        output_path: Destination ``.jsonl`` file (one game JSON per line).
            Ignored when *dry_run* is True.
        config: Filtering parameters.
        dry_run: When True, count games without writing any output and print
            a summary table to stdout.

    Returns:
        A :class:`FilterStats` instance with accumulated counts.
    """
    total = _count_games(pgn_path)

    with contextlib.ExitStack() as stack:
        # Open source — handle optional zstd compression
        if pgn_path.suffix == ".zst":
            import zstandard as zstd

            dctx = zstd.ZstdDecompressor()
            raw_fh = stack.enter_context(open(pgn_path, "rb"))
            text_stream: io.TextIOBase = stack.enter_context(
                io.TextIOWrapper(
                    dctx.stream_reader(raw_fh), encoding="utf-8", errors="replace"
                )
            )
        else:
            text_stream = stack.enter_context(
                open(pgn_path, encoding="utf-8", errors="replace")
            )

        return filter_stream(text_stream, output_path, config, total=total, dry_run=dry_run)
