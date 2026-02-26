"""Tests for StockfishMetric."""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import chess
import pytest
import torch

from chesstf.data.tokenizer import SPECIAL_TOKENS, ChessTokenizer
from chesstf.model.stockfish_metric import StockfishMetric

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tok() -> ChessTokenizer:
    return ChessTokenizer.build_complete_vocab()


@pytest.fixture(scope="module")
def id_to_move(tok: ChessTokenizer) -> dict[int, str]:
    return {v: k for k, v in tok._token_to_id.items() if k not in SPECIAL_TOKENS}


def _score_mock(cp: int) -> MagicMock:
    """Fake chess.engine.PovScore where .relative.score(...) returns *cp*."""
    m = MagicMock()
    m.relative.score.return_value = cp
    return m


def _seq(tok: ChessTokenizer, result: str, moves: list[str]) -> list[int]:
    """Build [<bos>, <result>, move0, ..., moveN, <eos>] token ID list."""
    result_name = {"1-0": "<w_win>", "0-1": "<b_win>", "1/2-1/2": "<draw>"}[result]
    return (
        [SPECIAL_TOKENS["<bos>"], SPECIAL_TOKENS[result_name]]
        + [tok.token_to_id(m) for m in moves]
        + [SPECIAL_TOKENS["<eos>"]]
    )


def _batch(
    tok: ChessTokenizer,
    result: str,
    moves: list[str],
    preds: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, input_ids) for a single sequence.

    preds[i] is the model's predicted move for moves[i] (token at position i+2).
    logits[:, t-1, :] predicts the token at position t, so for moves[i] at
    position t = i+2 we set logits[0, i+1, pred_id] = 1.0.
    """
    seq = _seq(tok, result, moves)
    seq_len = len(seq)
    vocab_size = len(tok)
    input_ids = torch.tensor([seq], dtype=torch.long)
    logits = torch.zeros(1, seq_len, vocab_size)
    for i, pred in enumerate(preds):
        t = i + 2  # sequence position of moves[i]
        logits[0, t - 1, tok.token_to_id(pred)] = 1.0
    return logits, input_ids


@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock()
    engine.analyse.return_value = {"score": _score_mock(100)}
    return engine


@pytest.fixture
def metric(id_to_move: dict[int, str], mock_engine: MagicMock) -> StockfishMetric:
    with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
        return StockfishMetric(id_to_move=id_to_move, max_positions=8)


# 4-move opening used across tests: white plays e2e4+g1f3, black plays e7e5+b8c6.
# Predicting the moves actually played keeps all preds legal.
_MOVES = ["e2e4", "e7e5", "g1f3", "b8c6"]
_PREDS = ["e2e4", "e7e5", "g1f3", "b8c6"]


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestInit:
    def test_engine_unavailable_sets_none(self, id_to_move: dict[int, str]) -> None:
        with patch("chess.engine.SimpleEngine.popen_uci", side_effect=FileNotFoundError):
            m = StockfishMetric(id_to_move=id_to_move)
        assert m.engine is None

    def test_engine_available(
        self, id_to_move: dict[int, str], mock_engine: MagicMock
    ) -> None:
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
            m = StockfishMetric(id_to_move=id_to_move)
        assert m.engine is mock_engine


# ---------------------------------------------------------------------------
# update — colour filtering
# ---------------------------------------------------------------------------


class TestUpdateColorFiltering:
    def test_no_engine_is_noop(
        self, id_to_move: dict[int, str], tok: ChessTokenizer
    ) -> None:
        """update returns immediately when engine is None; no state is changed."""
        with patch("chess.engine.SimpleEngine.popen_uci", side_effect=FileNotFoundError):
            m = StockfishMetric(id_to_move=id_to_move)
        logits, input_ids = _batch(tok, "1-0", _MOVES, _PREDS)
        m.update(logits, input_ids)
        assert int(m.total_positions) == 0

    def test_white_win_evaluates_only_white_turns(
        self, metric: StockfishMetric, tok: ChessTokenizer
    ) -> None:
        """<w_win>: only the 2 white-to-move positions are evaluated."""
        logits, input_ids = _batch(tok, "1-0", _MOVES, _PREDS)
        with patch.object(metric, "_centipawn_loss", return_value=0.0) as mock_cp:
            metric.update(logits, input_ids)
        assert mock_cp.call_count == 2

    def test_black_win_evaluates_only_black_turns(
        self, metric: StockfishMetric, tok: ChessTokenizer
    ) -> None:
        """<b_win>: only the 2 black-to-move positions are evaluated."""
        logits, input_ids = _batch(tok, "0-1", _MOVES, _PREDS)
        with patch.object(metric, "_centipawn_loss", return_value=0.0) as mock_cp:
            metric.update(logits, input_ids)
        assert mock_cp.call_count == 2

    def test_draw_evaluates_all_turns(
        self, metric: StockfishMetric, tok: ChessTokenizer
    ) -> None:
        """<draw>: all 4 positions are evaluated."""
        logits, input_ids = _batch(tok, "1/2-1/2", _MOVES, _PREDS)
        with patch.object(metric, "_centipawn_loss", return_value=0.0) as mock_cp:
            metric.update(logits, input_ids)
        assert mock_cp.call_count == 4

    def test_board_advances_on_skipped_loser_turns(
        self, metric: StockfishMetric, tok: ChessTokenizer
    ) -> None:
        """The board is still pushed forward on the loser's turns so that the
        next winner-turn evaluation sees the correct position."""
        logits, input_ids = _batch(tok, "1-0", _MOVES, _PREDS)
        captured: list[chess.Board] = []

        def capture(board: chess.Board, pred_uci: str) -> float:
            captured.append(board.copy())
            return 0.0

        with patch.object(metric, "_centipawn_loss", side_effect=capture):
            metric.update(logits, input_ids)

        # Two white positions evaluated
        assert len(captured) == 2
        # Before white's first move (e2e4): board is at the starting position
        assert captured[0].ply() == 0
        # Before white's second move (g1f3): e2e4 and e7e5 have both been played
        assert captured[1].ply() == 2


# ---------------------------------------------------------------------------
# update — move validity and early termination
# ---------------------------------------------------------------------------


class TestUpdateMoveHandling:
    def test_illegal_predicted_move_not_counted(
        self, metric: StockfishMetric, tok: ChessTokenizer
    ) -> None:
        """A predicted move that is illegal on the board is silently skipped."""
        # a5a4 is never legal from the starting position (a5 is empty)
        logits, input_ids = _batch(tok, "1-0", ["e2e4", "e7e5"], ["a5a4", "e7e5"])
        with patch.object(metric, "_centipawn_loss", return_value=0.0) as mock_cp:
            metric.update(logits, input_ids)
        assert mock_cp.call_count == 0
        assert int(metric.total_positions) == 0

    def test_eos_terminates_loop(
        self, metric: StockfishMetric, tok: ChessTokenizer
    ) -> None:
        """<eos> in the move region stops evaluation; tokens after it are ignored."""
        bos = SPECIAL_TOKENS["<bos>"]
        draw = SPECIAL_TOKENS["<draw>"]
        eos = SPECIAL_TOKENS["<eos>"]
        e2e4_id = tok.token_to_id("e2e4")
        e7e5_id = tok.token_to_id("e7e5")

        # BOS DRAW e2e4 EOS e7e5 — e7e5 comes after EOS and must never be reached
        seq = [bos, draw, e2e4_id, eos, e7e5_id]
        input_ids = torch.tensor([seq], dtype=torch.long)
        logits = torch.zeros(1, len(seq), len(tok))
        logits[0, 1, e2e4_id] = 1.0  # preds[1] predicts position 2 (e2e4)

        with patch.object(metric, "_centipawn_loss", return_value=0.0) as mock_cp:
            metric.update(logits, input_ids)

        assert mock_cp.call_count == 1

    def test_pad_terminates_loop(
        self, metric: StockfishMetric, tok: ChessTokenizer
    ) -> None:
        """<pad> in the move region (padded batch) ends evaluation."""
        bos = SPECIAL_TOKENS["<bos>"]
        draw = SPECIAL_TOKENS["<draw>"]
        pad = SPECIAL_TOKENS["<pad>"]
        e2e4_id = tok.token_to_id("e2e4")

        seq = [bos, draw, e2e4_id, pad, pad]
        input_ids = torch.tensor([seq], dtype=torch.long)
        logits = torch.zeros(1, len(seq), len(tok))
        logits[0, 1, e2e4_id] = 1.0

        with patch.object(metric, "_centipawn_loss", return_value=0.0) as mock_cp:
            metric.update(logits, input_ids)

        assert mock_cp.call_count == 1

    def test_max_positions_caps_total_evaluations(
        self, id_to_move: dict[int, str], tok: ChessTokenizer, mock_engine: MagicMock
    ) -> None:
        """No more than max_positions evaluations occur per update() call."""
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
            m = StockfishMetric(id_to_move=id_to_move, max_positions=2)

        # Draw → 4 eligible positions, but cap is 2
        logits, input_ids = _batch(tok, "1/2-1/2", _MOVES, _PREDS)
        with patch.object(m, "_centipawn_loss", return_value=0.0) as mock_cp:
            m.update(logits, input_ids)

        assert mock_cp.call_count == 2

    def test_cp_loss_accumulates_correctly(
        self, id_to_move: dict[int, str], tok: ChessTokenizer, mock_engine: MagicMock
    ) -> None:
        """total_cp_loss is the sum and total_positions is the count."""
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
            m = StockfishMetric(id_to_move=id_to_move, max_positions=8)

        # White win → 2 positions evaluated, each returning 30 cp loss
        logits, input_ids = _batch(tok, "1-0", _MOVES, _PREDS)
        with patch.object(m, "_centipawn_loss", return_value=30.0):
            m.update(logits, input_ids)

        assert int(m.total_positions) == 2
        assert float(m.total_cp_loss) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# _centipawn_loss
# ---------------------------------------------------------------------------


class TestCentipawnLoss:
    @pytest.fixture
    def cp_metric(
        self, id_to_move: dict[int, str], mock_engine: MagicMock
    ) -> StockfishMetric:
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
            return StockfishMetric(id_to_move=id_to_move)

    def test_calls_engine_analyse_twice(
        self, cp_metric: StockfishMetric, mock_engine: MagicMock
    ) -> None:
        """One analyse call for the position, one for after the predicted move."""
        mock_engine.analyse.return_value = {"score": _score_mock(0)}
        cp_metric._centipawn_loss(chess.Board(), "e2e4")
        assert mock_engine.analyse.call_count == 2

    def test_zero_when_pred_matches_engine_best(
        self, cp_metric: StockfishMetric, mock_engine: MagicMock
    ) -> None:
        """If the predicted move is as strong as the engine's best, loss = 0."""
        # best_cp = 100; after e2e4, opponent score = -100 → pred_cp = 100 → loss = 0
        mock_engine.analyse.side_effect = [
            {"score": _score_mock(100)},
            {"score": _score_mock(-100)},
        ]
        assert cp_metric._centipawn_loss(chess.Board(), "e2e4") == pytest.approx(0.0)

    def test_positive_when_pred_is_suboptimal(
        self, cp_metric: StockfishMetric, mock_engine: MagicMock
    ) -> None:
        """Loss = best_cp - pred_cp when the predicted move loses centipawns."""
        # best_cp=100; after move, opponent has -50 → pred_cp=50 → loss=50
        mock_engine.analyse.side_effect = [
            {"score": _score_mock(100)},
            {"score": _score_mock(-50)},
        ]
        assert cp_metric._centipawn_loss(chess.Board(), "e2e4") == pytest.approx(50.0)

    def test_clamped_to_zero_when_pred_beats_engine(
        self, cp_metric: StockfishMetric, mock_engine: MagicMock
    ) -> None:
        """Loss is clamped to 0; a move better than engine best gives 0, not negative."""
        # best_cp=50; after move, opponent has -200 → pred_cp=200 → max(0, -150)=0
        mock_engine.analyse.side_effect = [
            {"score": _score_mock(50)},
            {"score": _score_mock(-200)},
        ]
        assert cp_metric._centipawn_loss(chess.Board(), "e2e4") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


class TestCompute:
    def test_no_engine_returns_nan(self, id_to_move: dict[int, str]) -> None:
        with patch("chess.engine.SimpleEngine.popen_uci", side_effect=FileNotFoundError):
            m = StockfishMetric(id_to_move=id_to_move)
        assert math.isnan(m.compute().item())

    def test_no_positions_returns_nan(
        self, id_to_move: dict[int, str], mock_engine: MagicMock
    ) -> None:
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
            m = StockfishMetric(id_to_move=id_to_move)
        assert math.isnan(m.compute().item())

    def test_returns_mean_cp_loss(
        self, id_to_move: dict[int, str], mock_engine: MagicMock
    ) -> None:
        with patch("chess.engine.SimpleEngine.popen_uci", return_value=mock_engine):
            m = StockfishMetric(id_to_move=id_to_move)
        m.total_cp_loss = torch.tensor(150.0)
        m.total_positions = torch.tensor(3)
        assert m.compute().item() == pytest.approx(50.0)
