"""Tests for LegalityMetric."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch

from chesstf.data.tokenizer import SPECIAL_TOKENS, ChessTokenizer
from chesstf.model.legality_metric import LegalityMetric

if TYPE_CHECKING:
    from collections.abc import Sequence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tok() -> ChessTokenizer:
    return ChessTokenizer.build_complete_vocab()


@pytest.fixture(scope="module")
def id_to_move(tok: ChessTokenizer) -> dict[int, str]:
    return {v: k for k, v in tok._token_to_id.items() if k not in SPECIAL_TOKENS}


@pytest.fixture
def metric(id_to_move: dict[int, str]) -> LegalityMetric:
    return LegalityMetric(id_to_move=id_to_move)


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
    preds: Sequence[str | int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, input_ids) for a single sequence.

    preds[i] is the predicted token for moves[i] (position i+2).
    Pass a str for a UCI move or an int to set a raw token ID directly
    (useful for testing special-token predictions like <eos>).
    """
    seq = _seq(tok, result, moves)
    seq_len = len(seq)
    vocab_size = len(tok)
    input_ids = torch.tensor([seq], dtype=torch.long)
    logits = torch.zeros(1, seq_len, vocab_size)
    for i, pred in enumerate(preds):
        t = i + 2
        pred_id = tok.token_to_id(pred) if isinstance(pred, str) else pred
        logits[0, t - 1, pred_id] = 1.0
    return logits, input_ids


# 4-move opening: 2 white + 2 black, all predicted with the actual moves played.
_MOVES = ["e2e4", "e7e5", "g1f3", "b8c6"]
_PREDS_LEGAL = ["e2e4", "e7e5", "g1f3", "b8c6"]


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_legal_predicted_move_counted(
        self, metric: LegalityMetric, tok: ChessTokenizer
    ) -> None:
        """A legal prediction increments both total_positions and total_legal."""
        logits, input_ids = _batch(tok, "1-0", ["e2e4"], ["e2e4"])
        metric.update(logits, input_ids)
        assert int(metric.total_positions) == 1
        assert int(metric.total_legal_predictions) == 1

    def test_illegal_predicted_move_increments_positions_only(
        self, metric: LegalityMetric, tok: ChessTokenizer
    ) -> None:
        """An illegal prediction increments total_positions but not total_legal."""
        # a5a4 is never legal from the starting position (a5 is empty)
        logits, input_ids = _batch(tok, "1-0", ["e2e4"], ["a5a4"])
        metric.update(logits, input_ids)
        assert int(metric.total_positions) == 1
        assert int(metric.total_legal_predictions) == 0

    def test_non_move_token_predicted_counts_as_illegal(
        self, metric: LegalityMetric, tok: ChessTokenizer
    ) -> None:
        """Predicting a special token mid-game (e.g. <eos>) counts as a position
        with no legal prediction — not silently dropped from the denominator."""
        eos_id = SPECIAL_TOKENS["<eos>"]
        logits, input_ids = _batch(tok, "1-0", ["e2e4"], [eos_id])
        metric.update(logits, input_ids)
        assert int(metric.total_positions) == 1
        assert int(metric.total_legal_predictions) == 0

    def test_all_turns_evaluated_regardless_of_result(
        self, metric: LegalityMetric, tok: ChessTokenizer
    ) -> None:
        """Unlike StockfishMetric, both colours are evaluated; the result token
        has no bearing on which positions are counted."""
        logits, input_ids = _batch(tok, "1-0", _MOVES, _PREDS_LEGAL)
        metric.update(logits, input_ids)
        # 4 moves total (2 white, 2 black) — all 4 positions counted
        assert int(metric.total_positions) == 4
        assert int(metric.total_legal_predictions) == 4

    def test_board_advances_on_each_turn(
        self, metric: LegalityMetric, tok: ChessTokenizer
    ) -> None:
        """Board state is updated after each actual move so later evaluations
        reflect the correct position.

        e7e5 is illegal at the initial position (white to move) but becomes
        legal after 1. e4 (black to move). Predicting e7e5 at both positions
        in a two-move sequence should yield exactly one legal prediction.
        """
        logits, input_ids = _batch(tok, "1/2-1/2", ["e2e4", "e7e5"], ["e7e5", "e7e5"])
        metric.update(logits, input_ids)
        assert int(metric.total_positions) == 2
        assert int(metric.total_legal_predictions) == 1

    def test_eos_in_input_terminates_loop(
        self, metric: LegalityMetric, tok: ChessTokenizer
    ) -> None:
        """<eos> in the move region ends evaluation; tokens after it are ignored."""
        bos = SPECIAL_TOKENS["<bos>"]
        draw = SPECIAL_TOKENS["<draw>"]
        eos = SPECIAL_TOKENS["<eos>"]
        e2e4_id = tok.token_to_id("e2e4")
        e7e5_id = tok.token_to_id("e7e5")

        # BOS DRAW e2e4 EOS e7e5 — e7e5 comes after EOS and must not be counted
        seq = [bos, draw, e2e4_id, eos, e7e5_id]
        input_ids = torch.tensor([seq], dtype=torch.long)
        logits = torch.zeros(1, len(seq), len(tok))
        logits[0, 1, e2e4_id] = 1.0  # predict e2e4 for position 2

        metric.update(logits, input_ids)
        assert int(metric.total_positions) == 1

    def test_pad_in_input_terminates_loop(
        self, metric: LegalityMetric, tok: ChessTokenizer
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

        metric.update(logits, input_ids)
        assert int(metric.total_positions) == 1

    def test_accumulates_across_batch(
        self, metric: LegalityMetric, tok: ChessTokenizer
    ) -> None:
        """Counts are summed over all sequences in a batch."""
        seq = _seq(tok, "1/2-1/2", ["e2e4"])  # length 4
        e2e4_id = tok.token_to_id("e2e4")
        a5a4_id = tok.token_to_id("a5a4")

        input_ids = torch.tensor([seq, seq], dtype=torch.long)
        logits = torch.zeros(2, len(seq), len(tok))
        logits[0, 1, e2e4_id] = 1.0  # seq 0: legal prediction
        logits[1, 1, a5a4_id] = 1.0  # seq 1: illegal prediction

        metric.update(logits, input_ids)
        assert int(metric.total_positions) == 2
        assert int(metric.total_legal_predictions) == 1


# ---------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------


class TestCompute:
    def test_no_positions_returns_nan(self, id_to_move: dict[int, str]) -> None:
        m = LegalityMetric(id_to_move=id_to_move)
        assert math.isnan(m.compute().item())

    def test_perfect_legal_rate(self, id_to_move: dict[int, str]) -> None:
        m = LegalityMetric(id_to_move=id_to_move)
        m.total_legal_predictions = torch.tensor(5)
        m.total_positions = torch.tensor(5)
        assert m.compute().item() == pytest.approx(1.0)

    def test_zero_legal_rate(self, id_to_move: dict[int, str]) -> None:
        m = LegalityMetric(id_to_move=id_to_move)
        m.total_legal_predictions = torch.tensor(0)
        m.total_positions = torch.tensor(5)
        assert m.compute().item() == pytest.approx(0.0)

    def test_partial_legal_rate(self, id_to_move: dict[int, str]) -> None:
        m = LegalityMetric(id_to_move=id_to_move)
        m.total_legal_predictions = torch.tensor(3)
        m.total_positions = torch.tensor(4)
        assert m.compute().item() == pytest.approx(0.75)
