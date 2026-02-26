from __future__ import annotations

from typing import Any

import chess
import torch
from torchmetrics import Metric

from chesstf.data.tokenizer import SPECIAL_TOKENS

_SPECIAL_IDS: frozenset[int] = frozenset(SPECIAL_TOKENS.values())


class LegalityMetric(Metric):
    is_differentiable: bool | None = False

    total_legal_predictions: torch.Tensor
    total_positions: torch.Tensor

    def __init__(
        self,
        id_to_move: dict[int, str],
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.id_to_move = id_to_move

        self.add_state("total_legal_predictions", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_positions", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, input_ids: torch.Tensor) -> None:
        predicted_ids = logits.argmax(dim=-1).tolist()
        sequences = input_ids.tolist()

        for seq, preds in zip(sequences, predicted_ids, strict=False):
            # Don't care who won, we always want the legal move to be predicted
            board = chess.Board()
            for t in range(2, len(seq)):  # Skip <bos> and <result>
                token_id = seq[t]
                if token_id in _SPECIAL_IDS:
                    break

                actual_uci = self.id_to_move.get(token_id)
                if actual_uci is None:
                    break

                self.total_positions += 1
                pred_uci = self.id_to_move.get(preds[t - 1])
                if pred_uci is not None:
                    pred_move = chess.Move.from_uci(pred_uci)
                    if pred_move in board.legal_moves:
                        self.total_legal_predictions += 1
                
                try:
                    board.push_uci(actual_uci)
                except ValueError:
                    break

    def compute(self) -> torch.Tensor:
        if self.total_positions == 0:
            return torch.tensor(torch.nan)
        
        return self.total_legal_predictions / self.total_positions.float()