from __future__ import annotations

import logging
from typing import Any
from tqdm import tqdm

import chess
import chess.engine
import torch
from torchmetrics import Metric

from chesstf.data.tokenizer import SPECIAL_TOKENS

# Frozenset of special token IDs for O(1) membership checks.
# Sequences contain moves between <bos>/<result> and <eos>/<pad>; hitting any
# special ID in the move region means the game has ended.
_SPECIAL_IDS: frozenset[int] = frozenset(SPECIAL_TOKENS.values())

_W_WIN_ID = SPECIAL_TOKENS["<w_win>"]
_B_WIN_ID = SPECIAL_TOKENS["<b_win>"]


class StockfishMetric(Metric):
    is_differentiable: bool | None = False

    total_cp_loss: torch.Tensor
    total_positions: torch.Tensor
    n_evaluated: torch.Tensor

    def __init__(
        self,
        id_to_move: dict[int, str],
        engine_path: str | None = None,
        analysis_time: float = 0.05,
        max_positions: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.id_to_move = id_to_move
        self.analysis_time = analysis_time
        self.max_positions = max_positions
        self._engine_path = engine_path if engine_path is not None else "/usr/bin/stockfish"

        self.engine: chess.engine.SimpleEngine | None = None
        self._engine_available = self._open_engine()

        self.add_state("total_cp_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_positions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_evaluated", default=torch.tensor(0), dist_reduce_fx="sum")

    def _open_engine(self) -> bool:
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self._engine_path)
            return True
        except FileNotFoundError:
            logging.warning(f"Failed to find stockfish at `{self._engine_path}`, metric will be disabled")
            self.engine = None
            return False

    def _close_engine(self) -> None:
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def __del__(self) -> None:
        self._close_engine()

    def reset(self) -> None:
        self._close_engine()
        super().reset()

    def update(self, logits: torch.Tensor, input_ids: torch.Tensor) -> None:
        """Accumulate centipawn loss for up to *max_positions* positions per call.

        Sequence layout (result-conditioned)::

            position: 0       1         2      3      ...  N      N+1
            token:    <bos>   <result>  move0  move1  ...  moveN  <eos>

        ``logits[:, t, :]`` predicts the token at position ``t+1``, so the
        model's prediction for the move at position ``t`` is ``preds[t-1]``.

        With result conditioning the model learns to play *bad* moves on the
        loser's turns, which is correct behaviour at inference (the model only
        ever plays one colour).  To avoid penalising that, only positions where
        it is the **winner's** turn are evaluated.  For draws (or sequences
        without a result token) both sides are evaluated.
        """
        if not self._engine_available:
            return

        if self.n_evaluated.item() >= self.max_positions:
            return

        # Open engine lazily — reset() closes it; we reopen on first update() of each epoch.
        if self.engine is None and not self._open_engine():
            return

        # predicted_ids[b][t] = model's argmax prediction for position t+1
        predicted_ids = logits.argmax(dim=-1).tolist()
        sequences = input_ids.tolist()

        for seq, preds in tqdm(zip(sequences, predicted_ids, strict=False)):
            if self.n_evaluated.item() >= self.max_positions:
                break

            # Determine which colour to evaluate based on the result token.
            # seq[1] is <w_win>, <b_win>, <draw>, or a move token (no conditioning).
            result_token_id = seq[1]
            if result_token_id == _W_WIN_ID:
                winner_color: chess.Color | None = chess.WHITE
            elif result_token_id == _B_WIN_ID:
                winner_color = chess.BLACK
            else:
                winner_color = None  # draw or not result-conditioned: evaluate all

            board = chess.Board()

            for t in range(2, len(seq)):  # skip <bos> (0) and <result> (1)
                if self.n_evaluated.item() >= self.max_positions:
                    break

                token_id = seq[t]
                if token_id in _SPECIAL_IDS:
                    break  # <eos> or <pad> — end of game sequence

                actual_uci = self.id_to_move.get(token_id)
                if actual_uci is None:
                    break

                # Skip the loser's turns — intentionally weak play is correct
                # behaviour, not a metric failure.
                if winner_color is not None and board.turn != winner_color:
                    try:
                        board.push_uci(actual_uci)
                    except ValueError:
                        break
                    continue

                # preds[t-1] is the model's prediction for the token at position t
                pred_uci = self.id_to_move.get(preds[t - 1])

                if pred_uci is not None:
                    try:
                        pred_move = chess.Move.from_uci(pred_uci)
                    except ValueError:
                        pred_move = None

                    if pred_move is not None and pred_move in board.legal_moves:
                        cp_loss = self._centipawn_loss(board, pred_uci)
                        self.total_cp_loss += cp_loss
                        self.total_positions += 1
                        self.n_evaluated += 1

                # Advance the board with the move actually played in the game
                try:
                    board.push_uci(actual_uci)
                except ValueError:
                    break

    def _centipawn_loss(self, board: chess.Board, pred_uci: str) -> float:
        """Return centipawn loss of *pred_uci* vs. the engine's best move.

        Evaluates from the current player's perspective before the move, then
        from the same player's perspective after *pred_uci* (negating the
        opponent-to-move score).  The loss is clamped to 0 so a better-than-
        engine move never gives negative loss.
        """
        assert self.engine is not None
        limit = chess.engine.Limit(time=self.analysis_time)

        # Best achievable score from the player-to-move's perspective
        info = self.engine.analyse(board, limit)
        best_cp: int = info["score"].relative.score(mate_score=10000) or 0

        # Score after the predicted move — negate because it's now the opponent's turn
        board_after = board.copy()
        board_after.push_uci(pred_uci)
        info_after = self.engine.analyse(board_after, limit)
        pred_cp: int = -(info_after["score"].relative.score(mate_score=10000) or 0)

        return float(max(0, best_cp - pred_cp))

    def compute(self) -> torch.Tensor:
        if self.engine is None or self.total_positions == 0:
            return torch.tensor(torch.nan)
        return self.total_cp_loss / self.total_positions.float()
