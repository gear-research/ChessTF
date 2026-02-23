"""CLI for the full data pipeline: download → filter → vocab → encode.

Usage::

    python -m chesstf.data.process --help
    python -m chesstf.data.process download --year 2023 --months 1 2 3
    python -m chesstf.data.process filter  --year 2023 --months 1 2 3 [--dry-run] [--stream]
    python -m chesstf.data.process vocab
    python -m chesstf.data.process encode  --year 2023 --months 1 2 3
    python -m chesstf.data.process full    --year 2023 --months 1 2 3 [--stream]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


def _load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML config, falling back to defaults."""
    defaults: dict[str, Any] = {
        "min_elo": 1800,
        "min_base_time_seconds": 480,
        "min_moves": 10,
        "val_fraction": 0.02,
        "max_seq_len": 256,
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


def _raw_path(data_dir: Path, year: int, month: int) -> Path:
    return data_dir / "raw" / f"lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"


def _interim_path(data_dir: Path, year: int, month: int) -> Path:
    return data_dir / "interim" / f"games_{year}-{month:02d}.jsonl"


def _vocab_path(data_dir: Path) -> Path:
    return data_dir / "processed" / "vocab.json"


def _processed_dir(data_dir: Path) -> Path:
    return data_dir / "processed"


def _processed_month_dir(data_dir: Path, year: int, month: int) -> Path:
    return data_dir / "processed" / f"{year}-{month:02d}"


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_download(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    from chesstf.data.download import download_lichess_month

    data_dir = Path(str(cfg["data_dir"]))
    raw_dir = data_dir / "raw"
    for month in args.months:
        download_lichess_month(args.year, month, raw_dir, force=args.force)


def cmd_filter(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    from chesstf.data.filter import FilterConfig, filter_games

    data_dir = Path(str(cfg["data_dir"]))
    config = FilterConfig.from_dict(cfg)
    for month in args.months:
        output = _interim_path(data_dir, args.year, month)
        if args.stream:
            from chesstf.data.stream import stream_download_filter

            stats = stream_download_filter(
                args.year, month, output, config, dry_run=args.dry_run
            )
        else:
            pgn = _raw_path(data_dir, args.year, month)
            if not pgn.exists():
                print(f"ERROR: PGN file not found: {pgn}", file=sys.stderr)
                print("Run 'download' first, or pass --stream.", file=sys.stderr)
                sys.exit(1)
            stats = filter_games(pgn, output, config, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"Filtered {stats.passed:,}/{stats.total:,} games → {output}")
        # Dry run already printed the summary inside filter_stream


def cmd_vocab(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    from chesstf.data.tokenizer import ChessTokenizer

    data_dir = Path(str(cfg["data_dir"]))
    tok = ChessTokenizer.build_complete_vocab()
    vocab_file = _vocab_path(data_dir)
    tok.save(vocab_file)
    print(f"Vocabulary built: {tok.vocab_size} tokens → {vocab_file}")


def cmd_encode(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    from chesstf.data.dataset import encode_to_binary
    from chesstf.data.tokenizer import ChessTokenizer

    data_dir = Path(str(cfg["data_dir"]))
    vocab_file = _vocab_path(data_dir)

    if not vocab_file.exists():
        print(f"ERROR: Vocab file not found: {vocab_file}", file=sys.stderr)
        print("Run 'vocab' first.", file=sys.stderr)
        sys.exit(1)

    tok = ChessTokenizer.load(vocab_file)
    for month in args.months:
        jsonl = _interim_path(data_dir, args.year, month)
        if not jsonl.exists():
            print(f"ERROR: Filtered JSONL not found: {jsonl}", file=sys.stderr)
            sys.exit(1)
        out_dir = _processed_month_dir(data_dir, args.year, month)
        counts = encode_to_binary(
            jsonl,
            out_dir,
            tok,
            val_fraction=float(cfg["val_fraction"]),
            result_conditioning=bool(cfg["result_conditioning"]),
        )
        print(
            f"Encoded → {out_dir}  "
            f"(train={counts['train']}, val={counts['val']})"
        )


def cmd_full(args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    """Run the complete pipeline: [download →] filter → vocab → encode."""
    args.dry_run = False
    if args.stream:
        print("=== Step 1/3: Filter (streaming) ===")
        cmd_filter(args, cfg)

        print("\n=== Step 2/3: Build vocabulary ===")
        cmd_vocab(args, cfg)

        print("\n=== Step 3/3: Encode to binary ===")
        cmd_encode(args, cfg)
    else:
        print("=== Step 1/4: Download ===")
        cmd_download(args, cfg)

        print("\n=== Step 2/4: Filter ===")
        cmd_filter(args, cfg)

        print("\n=== Step 3/4: Build vocabulary ===")
        cmd_vocab(args, cfg)

        print("\n=== Step 4/4: Encode to binary ===")
        cmd_encode(args, cfg)

    print("\nPipeline complete.")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chesstf.data.process",
        description="Chess Transformer data pipeline CLI",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to YAML config file (default: configs/data/default.yaml)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # Shared year/months arguments
    def _add_ym(p: argparse.ArgumentParser) -> None:
        p.add_argument("--year", type=int, required=True, help="Four-digit year")
        p.add_argument(
            "--months",
            type=int,
            nargs="+",
            required=True,
            choices=range(1, 13),
            metavar="MONTH",
            help="Month(s) 1–12, e.g. --months 1 2 3",
        )

    # download
    dl = sub.add_parser("download", help="Download one or more Lichess monthly PGN dumps")
    _add_ym(dl)
    dl.add_argument("--force", action="store_true", help="Re-download if file exists")
    dl.set_defaults(func=cmd_download)

    # filter
    filt = sub.add_parser("filter", help="Filter PGN games by ELO/time/result")
    _add_ym(filt)
    filt.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Count matching games without writing output",
    )
    filt.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream directly from Lichess instead of reading a local .pgn.zst file",
    )
    filt.set_defaults(func=cmd_filter)

    # vocab
    vocab = sub.add_parser(
        "vocab",
        help="Build complete UCI move vocabulary (enumerates all possible chess moves)",
    )
    vocab.set_defaults(func=cmd_vocab)

    # encode
    enc = sub.add_parser("encode", help="Encode filtered games to binary token arrays")
    _add_ym(enc)
    enc.set_defaults(func=cmd_encode)

    # full
    full = sub.add_parser("full", help="Run the complete pipeline end-to-end")
    _add_ym(full)
    full.add_argument("--force", action="store_true", help="Re-download if file exists")
    full.add_argument(
        "--stream",
        action="store_true",
        default=False,
        help="Stream directly from Lichess, skipping the download step",
    )
    full.set_defaults(func=cmd_full)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cfg = _load_config(args.config)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
