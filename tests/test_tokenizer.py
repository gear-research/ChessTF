"""Tests for ChessTokenizer (UCI move format)."""

from __future__ import annotations

import json

import pytest

from chesstf.data.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    ChessTokenizer,
)


class TestSpecialTokens:
    def test_special_token_ids_are_fixed(self, simple_tokenizer: ChessTokenizer) -> None:
        assert simple_tokenizer.token_to_id("<pad>") == 0
        assert simple_tokenizer.token_to_id("<bos>") == 1
        assert simple_tokenizer.token_to_id("<eos>") == 2
        assert simple_tokenizer.token_to_id("<w_win>") == 3
        assert simple_tokenizer.token_to_id("<b_win>") == 4
        assert simple_tokenizer.token_to_id("<draw>") == 5

    def test_properties_match_constants(self, simple_tokenizer: ChessTokenizer) -> None:
        assert simple_tokenizer.pad_id == PAD_ID == 0
        assert simple_tokenizer.bos_id == BOS_ID == 1
        assert simple_tokenizer.eos_id == EOS_ID == 2


class TestVocabBuilding:
    def test_move_tokens_start_at_6(self, simple_tokenizer: ChessTokenizer) -> None:
        for tid in simple_tokenizer._token_to_id.values():
            if simple_tokenizer._id_to_token[tid] not in SPECIAL_TOKENS:
                assert tid >= 6

    def test_move_ids_are_alphabetically_ordered(self) -> None:
        tok = ChessTokenizer()
        tok.build_vocab([["g1f3", "e2e4", "f1c4", "e7e5"]])
        moves_by_id = sorted(
            (tid, token)
            for token, tid in tok._token_to_id.items()
            if token not in SPECIAL_TOKENS
        )
        tokens_in_order = [t for _, t in moves_by_id]
        assert tokens_in_order == sorted(tokens_in_order)

    def test_vocab_size_counts_special_and_move_tokens(self) -> None:
        tok = ChessTokenizer()
        tok.build_vocab([["e2e4", "e7e5"]])
        assert tok.vocab_size == len(SPECIAL_TOKENS) + 2

    def test_build_vocab_is_idempotent(self) -> None:
        tok = ChessTokenizer()
        tok.build_vocab([["e2e4", "e7e5"]])
        size_before = tok.vocab_size
        tok.build_vocab([["e2e4", "e7e5"]])
        assert tok.vocab_size == size_before

    def test_add_new_moves_increments_vocab(self) -> None:
        tok = ChessTokenizer()
        tok.build_vocab([["e2e4", "e7e5"]])
        tok.add_moves(["g1f3"])
        assert "g1f3" in tok
        assert tok.vocab_size == len(SPECIAL_TOKENS) + 3

    def test_contains_dunder(self, simple_tokenizer: ChessTokenizer) -> None:
        assert "e2e4" in simple_tokenizer
        assert "z9z9" not in simple_tokenizer

    def test_len_dunder(self, simple_tokenizer: ChessTokenizer) -> None:
        assert len(simple_tokenizer) == simple_tokenizer.vocab_size


class TestEncodeDecode:
    def test_encode_with_special_tokens(self, simple_tokenizer: ChessTokenizer) -> None:
        ids = simple_tokenizer.encode(["e2e4", "e7e5"], add_special=True)
        assert ids[0] == BOS_ID
        assert ids[-1] == EOS_ID
        assert len(ids) == 4  # bos + 2 moves + eos

    def test_encode_without_special_tokens(self, simple_tokenizer: ChessTokenizer) -> None:
        ids = simple_tokenizer.encode(["e2e4", "e7e5"], add_special=False)
        assert BOS_ID not in ids
        assert EOS_ID not in ids
        assert len(ids) == 2

    def test_encode_unknown_move_raises(self, simple_tokenizer: ChessTokenizer) -> None:
        with pytest.raises(KeyError):
            simple_tokenizer.encode(["z9z9"])

    def test_decode_skips_special_by_default(self, simple_tokenizer: ChessTokenizer) -> None:
        ids = simple_tokenizer.encode(["e2e4", "e7e5"], add_special=True)
        decoded = simple_tokenizer.decode(ids)
        assert decoded == ["e2e4", "e7e5"]

    def test_decode_includes_special_when_requested(
        self, simple_tokenizer: ChessTokenizer
    ) -> None:
        ids = simple_tokenizer.encode(["e2e4"], add_special=True)
        decoded = simple_tokenizer.decode(ids, skip_special=False)
        assert "<bos>" in decoded
        assert "<eos>" in decoded

    def test_decode_unknown_id_raises(self, simple_tokenizer: ChessTokenizer) -> None:
        with pytest.raises(KeyError):
            simple_tokenizer.decode([99999])

    def test_roundtrip(self, simple_tokenizer: ChessTokenizer) -> None:
        moves = ["e2e4", "e7e5", "g1f3", "b8c6"]
        ids = simple_tokenizer.encode(moves, add_special=False)
        assert simple_tokenizer.decode(ids) == moves

    def test_encode_result(self, simple_tokenizer: ChessTokenizer) -> None:
        assert simple_tokenizer.encode_result("1-0") == SPECIAL_TOKENS["<w_win>"]
        assert simple_tokenizer.encode_result("0-1") == SPECIAL_TOKENS["<b_win>"]
        assert simple_tokenizer.encode_result("1/2-1/2") == SPECIAL_TOKENS["<draw>"]

    def test_encode_result_invalid_raises(self, simple_tokenizer: ChessTokenizer) -> None:
        with pytest.raises(KeyError):
            simple_tokenizer.encode_result("*")

    def test_encode_with_result_conditioning(self, simple_tokenizer: ChessTokenizer) -> None:
        ids = simple_tokenizer.encode(["e2e4", "e7e5"], add_special=True, result="1-0")
        # Expected: [BOS, <w_win>, e2e4, e7e5, EOS]
        assert len(ids) == 5
        assert ids[0] == BOS_ID
        assert ids[1] == SPECIAL_TOKENS["<w_win>"]
        assert ids[-1] == EOS_ID

    def test_encode_result_conditioning_ordering(self, simple_tokenizer: ChessTokenizer) -> None:
        """Result token must follow BOS, not precede it."""
        ids = simple_tokenizer.encode(["e2e4"], add_special=True, result="0-1")
        assert ids[0] == BOS_ID, "BOS must be first"
        assert ids[1] == SPECIAL_TOKENS["<b_win>"], "result token must be second"
        assert ids[-1] == EOS_ID

    def test_encode_result_all_outcomes(self, simple_tokenizer: ChessTokenizer) -> None:
        for result_str, token_name in [("1-0", "<w_win>"), ("0-1", "<b_win>"), ("1/2-1/2", "<draw>")]:
            ids = simple_tokenizer.encode(["e2e4"], add_special=True, result=result_str)
            assert ids[1] == SPECIAL_TOKENS[token_name]

    def test_encode_result_without_add_special_ignored(self, simple_tokenizer: ChessTokenizer) -> None:
        """result= has no effect when add_special=False."""
        ids = simple_tokenizer.encode(["e2e4"], add_special=False, result="1-0")
        assert BOS_ID not in ids
        assert SPECIAL_TOKENS["<w_win>"] not in ids

    def test_encode_invalid_result_raises(self, simple_tokenizer: ChessTokenizer) -> None:
        with pytest.raises(KeyError):
            simple_tokenizer.encode(["e2e4"], add_special=True, result="*")


class TestSaveLoad:
    def test_save_creates_valid_json(
        self, simple_tokenizer: ChessTokenizer, tmp_path: pytest.TempPathFactory
    ) -> None:
        vocab_path = tmp_path / "vocab.json"  # type: ignore[operator]
        simple_tokenizer.save(vocab_path)
        payload = json.loads(vocab_path.read_text())
        assert payload["version"] == "1"
        assert "special_tokens" in payload
        assert "vocab" in payload

    def test_save_load_roundtrip(
        self, simple_tokenizer: ChessTokenizer, tmp_path: pytest.TempPathFactory
    ) -> None:
        vocab_path = tmp_path / "vocab.json"  # type: ignore[operator]
        simple_tokenizer.save(vocab_path)
        loaded = ChessTokenizer.load(vocab_path)
        assert loaded.vocab_size == simple_tokenizer.vocab_size
        assert loaded._token_to_id == simple_tokenizer._token_to_id

    def test_load_wrong_version_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        vocab_path = tmp_path / "vocab.json"  # type: ignore[operator]
        vocab_path.write_text(json.dumps({"version": "99", "special_tokens": {}, "vocab": {}}))
        with pytest.raises(ValueError, match="Unsupported vocab version"):
            ChessTokenizer.load(vocab_path)

    def test_save_creates_parent_dirs(
        self, simple_tokenizer: ChessTokenizer, tmp_path: pytest.TempPathFactory
    ) -> None:
        nested = tmp_path / "deep" / "dir" / "vocab.json"  # type: ignore[operator]
        simple_tokenizer.save(nested)
        assert nested.exists()


class TestBuildCompleteVocab:
    def test_returns_tokenizer_instance(self) -> None:
        tok = ChessTokenizer.build_complete_vocab()
        assert isinstance(tok, ChessTokenizer)

    def test_special_tokens_preserved(self) -> None:
        tok = ChessTokenizer.build_complete_vocab()
        assert tok.pad_id == PAD_ID == 0
        assert tok.bos_id == BOS_ID == 1
        assert tok.eos_id == EOS_ID == 2

    def test_common_moves_present(self) -> None:
        tok = ChessTokenizer.build_complete_vocab()
        for move in ["e2e4", "d7d5", "g1f3", "f8c5", "e1g1", "e1c1", "e8g8", "e8c8"]:
            assert move in tok, f"Expected {move!r} in complete vocab"

    def test_promotions_present(self) -> None:
        tok = ChessTokenizer.build_complete_vocab()
        for move in ["e7e8q", "a7a8n", "h2h1r", "g2h1b", "b7c8q"]:
            assert move in tok, f"Expected {move!r} in complete vocab"

    def test_vocab_size_in_expected_range(self) -> None:
        tok = ChessTokenizer.build_complete_vocab()
        # ~1,968 move tokens + 6 special tokens
        assert 1_900 <= tok.vocab_size <= 2_100

    def test_move_tokens_start_at_6(self) -> None:
        tok = ChessTokenizer.build_complete_vocab()
        for token, tid in tok._token_to_id.items():
            if token not in SPECIAL_TOKENS:
                assert tid >= 6

    def test_can_encode_and_decode_moves(self) -> None:
        tok = ChessTokenizer.build_complete_vocab()
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
        ids = tok.encode(moves, add_special=False)
        assert tok.decode(ids) == moves

    def test_covers_game_with_castling_and_promotion(self) -> None:
        """UCI moves from a game that includes castling and underpromotion."""
        import io

        import chess
        import chess.pgn

        pgn_str = (
            "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 5. d3 d6 "
            "6. Nc3 O-O 7. Bg5 h6 8. Bh4 g5 9. Bg3 Nh5 1/2-1/2"
        )
        tok = ChessTokenizer.build_complete_vocab()
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        assert game is not None
        board = game.board()
        for move in game.mainline_moves():
            uci = move.uci()
            assert uci in tok, f"UCI move {uci!r} not in complete vocab"
            board.push(move)
