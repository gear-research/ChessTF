"""Chess move tokenizer for UCI-format chess moves."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

# Special token definitions — IDs are fixed and must never change.
SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<w_win>": 3,
    "<b_win>": 4,
    "<draw>": 5,
}

RESULT_TO_TOKEN: dict[str, str] = {
    "1-0": "<w_win>",
    "0-1": "<b_win>",
    "1/2-1/2": "<draw>",
}

_VOCAB_VERSION = "1"


@functools.lru_cache(maxsize=1)
def _enumerate_all_uci_moves() -> list[str]:
    """Enumerate every UCI move string that can appear in a legal chess game.

    UCI format: ``[from_sq][to_sq][promo_piece?]``, e.g. ``e2e4``, ``g1f3``,
    ``e7e8q``.  The vocabulary is derived from actual piece-movement geometry:

    - **Non-pawn pieces** (N, B, R, Q, K): attack squares on an empty board.
      This naturally covers castling (encoded as king's 2-square rook-type move,
      e.g. ``e1g1``) and en-passant capture strings (diagonal bishop-type).
    - **Pawn non-promotion moves** (advances + diagonal captures) coincide with
      rook-type and bishop-type strings already generated above.
    - **Pawn promotions**: the only UCI strings unique to pawns — 5-character
      strings such as ``e7e8q`` — are added separately for both colors.

    The result is cached after the first call.  Vocabulary size is ~1,968 tokens,
    matching the original spec estimate and comfortably within embedding budgets.

    Returns:
        Sorted list of UCI move strings (deterministic across runs).
    """
    import chess

    moves: set[str] = set()
    promo_pieces = "qrbn"

    # ------------------------------------------------------------------
    # Piece attacks on an empty board
    # Covers all king / queen / rook / bishop / knight moves.
    # Pawn advances (same-file) and captures (diagonal) are subsets of
    # rook-type and bishop-type strings respectively.
    # ------------------------------------------------------------------
    board = chess.Board()
    board.clear_board()
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        for src_sq in chess.SQUARES:
            src_name = chess.square_name(src_sq)
            board.set_piece_at(src_sq, chess.Piece(piece_type, chess.WHITE))
            for dst_sq in board.attacks(src_sq):
                moves.add(src_name + chess.square_name(dst_sq))
            board.remove_piece_at(src_sq)

    # ------------------------------------------------------------------
    # Pawn promotions (5-char UCI strings, unique to pawns)
    # White: rank 7 (index 6) → rank 8 (index 7)
    # Black: rank 2 (index 1) → rank 1 (index 0)
    # ------------------------------------------------------------------
    for src_file in range(8):
        # White straight promotion
        src_w = chess.square_name(chess.square(src_file, 6))
        dst_w = chess.square_name(chess.square(src_file, 7))
        for p in promo_pieces:
            moves.add(src_w + dst_w + p)
        # White promotion captures (adjacent files)
        for df in (-1, 1):
            cap_file = src_file + df
            if 0 <= cap_file <= 7:
                dst_cap = chess.square_name(chess.square(cap_file, 7))
                for p in promo_pieces:
                    moves.add(src_w + dst_cap + p)

        # Black straight promotion
        src_b = chess.square_name(chess.square(src_file, 1))
        dst_b = chess.square_name(chess.square(src_file, 0))
        for p in promo_pieces:
            moves.add(src_b + dst_b + p)
        # Black promotion captures (adjacent files)
        for df in (-1, 1):
            cap_file = src_file + df
            if 0 <= cap_file <= 7:
                dst_cap = chess.square_name(chess.square(cap_file, 0))
                for p in promo_pieces:
                    moves.add(src_b + dst_cap + p)

    return sorted(moves)

PAD_ID = SPECIAL_TOKENS["<pad>"]
BOS_ID = SPECIAL_TOKENS["<bos>"]
EOS_ID = SPECIAL_TOKENS["<eos>"]


class ChessTokenizer:
    """Move-level tokenizer for UCI chess moves.

    Special tokens occupy IDs 0-5 (fixed).  Move tokens start at 6 and are
    assigned in alphabetical order for determinism.

    Usage::

        tok = ChessTokenizer.build_complete_vocab()
        tok.save("vocab.json")

        tok2 = ChessTokenizer.load("vocab.json")
        ids = tok2.encode(["e2e4", "e7e5", "g1f3"], add_special=True)
        moves = tok2.decode(ids)
    """

    def __init__(self) -> None:
        # token -> id
        self._token_to_id: dict[str, int] = dict(SPECIAL_TOKENS)
        # id -> token
        self._id_to_token: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}

    # ------------------------------------------------------------------
    # Vocab construction
    # ------------------------------------------------------------------

    @classmethod
    def build_complete_vocab(cls) -> ChessTokenizer:
        """Return a tokenizer pre-populated with every possible UCI move.

        Enumerates all UCI move strings that can appear in a legal chess game
        (piece moves on an empty board + pawn promotions), so the tokenizer can
        encode any move regardless of how unusual the board position is.

        Vocabulary size is ~1,968 move tokens plus 6 special tokens.

        Returns:
            A new :class:`ChessTokenizer` with a complete move vocabulary.
        """
        tok = cls()
        tok.build_vocab([_enumerate_all_uci_moves()])
        return tok

    def build_vocab(self, all_moves: list[list[str]]) -> None:
        """Build vocabulary from a collection of move lists.

        Existing vocabulary is preserved; new moves are appended in sorted
        order so results are deterministic across calls with the same data.

        Args:
            all_moves: Each element is a list of SAN move strings from one game.
        """
        seen: set[str] = set()
        for moves in all_moves:
            seen.update(moves)
        # Remove any already-known tokens
        new_tokens = sorted(seen - set(self._token_to_id))
        next_id = max(self._id_to_token) + 1 if self._id_to_token else len(SPECIAL_TOKENS)
        for token in new_tokens:
            self._token_to_id[token] = next_id
            self._id_to_token[next_id] = token
            next_id += 1

    def add_moves(self, moves: list[str]) -> None:
        """Add individual move tokens that are not yet in the vocabulary."""
        self.build_vocab([moves])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        move_vocab = {
            token: tid
            for token, tid in self._token_to_id.items()
            if token not in SPECIAL_TOKENS
        }
        payload: dict[str, Any] = {
            "version": _VOCAB_VERSION,
            "special_tokens": SPECIAL_TOKENS,
            "vocab": move_vocab,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> ChessTokenizer:
        """Load a tokenizer from a saved JSON file."""
        path = Path(path)
        payload: dict[str, Any] = json.loads(path.read_text())
        if payload.get("version") != _VOCAB_VERSION:
            raise ValueError(
                f"Unsupported vocab version: {payload.get('version')!r}. "
                f"Expected {_VOCAB_VERSION!r}."
            )
        tok = cls()
        for token, tid in payload["vocab"].items():
            tok._token_to_id[token] = tid
            tok._id_to_token[tid] = token
        return tok

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------

    def encode(self, moves: list[str], *, add_special: bool = True) -> list[int]:
        """Encode a list of SAN moves to token IDs.

        Args:
            moves: Ordered list of SAN move strings for one game.
            add_special: If True, prepend ``<bos>`` and append ``<eos>``.

        Returns:
            List of integer token IDs.

        Raises:
            KeyError: If a move is not in the vocabulary.
        """
        ids = [self._token_to_id[m] for m in moves]
        if add_special:
            ids = [BOS_ID, *ids, EOS_ID]
        return ids

    def encode_result(self, result: str) -> int:
        """Return the token ID for a game result string ('1-0', '0-1', '1/2-1/2').

        Raises:
            KeyError: If *result* is not a recognised result string.
        """
        token = RESULT_TO_TOKEN[result]
        return self._token_to_id[token]

    def decode(self, ids: list[int], *, skip_special: bool = True) -> list[str]:
        """Decode token IDs back to SAN move strings.

        Args:
            ids: List of token IDs.
            skip_special: If True, special tokens are omitted from output.

        Returns:
            List of token strings (moves and/or special tokens).
        """
        special_ids = set(SPECIAL_TOKENS.values())
        tokens = []
        for tid in ids:
            token = self._id_to_token.get(tid)
            if token is None:
                raise KeyError(f"Unknown token id: {tid}")
            if skip_special and tid in special_ids:
                continue
            tokens.append(token)
        return tokens

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of tokens including special tokens."""
        return len(self._token_to_id)

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def token_to_id(self, token: str) -> int:
        """Look up a token string. Raises ``KeyError`` if not found."""
        return self._token_to_id[token]

    def id_to_token(self, tid: int) -> str:
        """Look up a token ID. Raises ``KeyError`` if not found."""
        return self._id_to_token[tid]

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    def __len__(self) -> int:
        return self.vocab_size
