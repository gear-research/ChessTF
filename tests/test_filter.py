"""Tests for filter.py — no disk I/O required (uses io.StringIO)."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import chess.pgn

from chesstf.data.filter import (
    FilterConfig,
    FilterStats,
    _extract_moves,
    _passes_filter,
    filter_games,
)
from tests.conftest import (
    SAMPLE_PGN,
    SAMPLE_PGN_ABANDONED,
    SAMPLE_PGN_BULLET,
    SAMPLE_PGN_LOW_ELO,
)

if TYPE_CHECKING:
    from pathlib import Path


def _parse_first_game(pgn_text: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None
    return game


def _default_config() -> FilterConfig:
    return FilterConfig(min_elo=1800, min_base_time_seconds=480, min_moves=5)


class TestPassesFilter:
    def test_valid_game_passes(self) -> None:
        game = _parse_first_game(SAMPLE_PGN)
        stats = FilterStats()
        assert _passes_filter(game, _default_config(), stats) is True
        assert stats.total == 0  # caller increments total

    def test_low_elo_rejected(self) -> None:
        game = _parse_first_game(SAMPLE_PGN_LOW_ELO)
        stats = FilterStats()
        assert _passes_filter(game, _default_config(), stats) is False
        assert stats.low_elo == 1

    def test_bullet_rejected(self) -> None:
        game = _parse_first_game(SAMPLE_PGN_BULLET)
        stats = FilterStats()
        assert _passes_filter(game, _default_config(), stats) is False
        assert stats.bad_time_control == 1

    def test_abandoned_rejected(self) -> None:
        game = _parse_first_game(SAMPLE_PGN_ABANDONED)
        stats = FilterStats()
        assert _passes_filter(game, _default_config(), stats) is False
        assert stats.bad_termination == 1

    def test_missing_elo_rejected(self) -> None:
        pgn = """\
[Event "Test"]
[WhiteElo "?"]
[BlackElo "1900"]
[TimeControl "600+0"]
[Result "1-0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 1-0

"""
        game = _parse_first_game(pgn)
        stats = FilterStats()
        assert _passes_filter(game, _default_config(), stats) is False
        assert stats.low_elo == 1

    def test_bad_result_rejected(self) -> None:
        pgn = """\
[Event "Test"]
[WhiteElo "2000"]
[BlackElo "1900"]
[TimeControl "600+0"]
[Result "*"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 *

"""
        game = _parse_first_game(pgn)
        stats = FilterStats()
        assert _passes_filter(game, _default_config(), stats) is False
        assert stats.bad_result == 1

    def test_too_few_moves_rejected(self) -> None:
        pgn = """\
[Event "Test"]
[WhiteElo "2000"]
[BlackElo "1900"]
[TimeControl "600+0"]
[Result "1-0"]
[Termination "Normal"]

1. e4 e5 1-0

"""
        game = _parse_first_game(pgn)
        stats = FilterStats(total=1)
        # min_moves=5, only 2 moves → rejected
        config = FilterConfig(min_elo=1800, min_base_time_seconds=0, min_moves=5)
        assert _passes_filter(game, config, stats) is False
        assert stats.too_few_moves == 1

    def test_missing_time_control_rejected(self) -> None:
        pgn = """\
[Event "Test"]
[WhiteElo "2000"]
[BlackElo "1900"]
[TimeControl "-"]
[Result "1-0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 1-0

"""
        game = _parse_first_game(pgn)
        stats = FilterStats()
        assert _passes_filter(game, _default_config(), stats) is False
        assert stats.bad_time_control == 1


class TestExtractMoves:
    def test_extracts_correct_uci(self) -> None:
        game = _parse_first_game(SAMPLE_PGN)
        moves = _extract_moves(game)
        assert moves[0] == "e2e4"
        assert moves[1] == "e7e5"
        assert moves[2] == "g1f3"

    def test_extract_returns_list_of_strings(self) -> None:
        game = _parse_first_game(SAMPLE_PGN)
        moves = _extract_moves(game)
        assert all(isinstance(m, str) for m in moves)

    def test_empty_game_returns_empty_list(self) -> None:
        pgn = "[Event \"Test\"]\n\n*\n\n"
        game = _parse_first_game(pgn)
        assert _extract_moves(game) == []


class TestFilterStats:
    def test_rejected_count(self) -> None:
        stats = FilterStats(total=10, passed=7)
        assert stats.rejected == 3

    def test_summary_contains_key_labels(self) -> None:
        stats = FilterStats(total=100, passed=60, low_elo=20, bad_time_control=10)
        summary = stats.summary()
        assert "Total scanned" in summary
        assert "Passed" in summary
        assert "low_elo" in summary
        assert "bad_time_control" in summary

    def test_summary_zero_total_no_division_error(self) -> None:
        stats = FilterStats()
        stats.summary()  # must not raise


class TestFilterGames:
    def test_filter_writes_jsonl(self, tmp_path: Path) -> None:
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(SAMPLE_PGN)
        output = tmp_path / "out.jsonl"

        config = FilterConfig(min_elo=1800, min_base_time_seconds=480, min_moves=5)
        stats = filter_games(pgn_file, output, config)

        assert stats.total >= 1
        assert stats.passed >= 1
        assert output.exists()
        lines = output.read_text().splitlines()
        assert len(lines) == stats.passed
        record = json.loads(lines[0])
        assert "moves" in record
        assert "result" in record

    def test_filter_dry_run_does_not_write(self, tmp_path: Path) -> None:
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(SAMPLE_PGN)
        output = tmp_path / "out.jsonl"

        config = FilterConfig(min_elo=1800, min_base_time_seconds=480, min_moves=5)
        filter_games(pgn_file, output, config, dry_run=True)

        assert not output.exists()

    def test_filter_dry_run_returns_stats(self, tmp_path: Path) -> None:
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(SAMPLE_PGN)
        output = tmp_path / "out.jsonl"

        config = FilterConfig(min_elo=1800, min_base_time_seconds=480, min_moves=5)
        stats = filter_games(pgn_file, output, config, dry_run=True)

        assert isinstance(stats, FilterStats)
        assert stats.total >= 1

    def test_filter_low_elo_rejected(self, tmp_path: Path) -> None:
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(SAMPLE_PGN_LOW_ELO)
        output = tmp_path / "out.jsonl"

        config = FilterConfig(min_elo=1800, min_base_time_seconds=0, min_moves=1)
        stats = filter_games(pgn_file, output, config)

        assert stats.low_elo > 0
        assert stats.passed == 0

    def test_filter_multiple_games(self, tmp_path: Path) -> None:
        # Two valid games back-to-back
        combined = SAMPLE_PGN + "\n" + SAMPLE_PGN
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(combined)
        output = tmp_path / "out.jsonl"

        config = FilterConfig(min_elo=1800, min_base_time_seconds=480, min_moves=5)
        stats = filter_games(pgn_file, output, config)

        assert stats.total == 2
        assert stats.passed == 2
        lines = output.read_text().splitlines()
        assert len(lines) == 2

    def test_jsonl_record_structure(self, tmp_path: Path) -> None:
        pgn_file = tmp_path / "games.pgn"
        pgn_file.write_text(SAMPLE_PGN)
        output = tmp_path / "out.jsonl"

        config = FilterConfig(min_elo=1800, min_base_time_seconds=480, min_moves=5)
        filter_games(pgn_file, output, config)

        record = json.loads(output.read_text().splitlines()[0])
        assert isinstance(record["moves"], list)
        assert isinstance(record["result"], str)
        assert isinstance(record["white_elo"], int)
        assert isinstance(record["black_elo"], int)
