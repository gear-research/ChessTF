"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from chesstf.data.tokenizer import ChessTokenizer


@pytest.fixture()
def simple_tokenizer() -> ChessTokenizer:
    """A tokenizer pre-loaded with a small set of UCI moves."""
    tok = ChessTokenizer()
    tok.build_vocab([["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]])
    return tok


SAMPLE_PGN = """\
[Event "Test"]
[White "Alice"]
[Black "Bob"]
[WhiteElo "2000"]
[BlackElo "1900"]
[TimeControl "600+0"]
[Result "1-0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3
8. Qb3 Qf6 9. e5 Qg6 10. Re1 Nge7 11. Ba3 b5 12. Qxb5 Rb8 13. Qa4 Bb6
14. Nbd2 Bb7 15. Ne4 Qf5 16. Bxd3 Qh5 17. Nf6+ gxf6 18. exf6 Rg8 19. Rad1
Qxf3 20. Rxe7+ Nxe7 21. Qxd7+ Kxd7 22. Bf5+ Ke8 23. Bd7+ Kf8 24. Bxe7# 1-0

"""

SAMPLE_PGN_LOW_ELO = """\
[Event "Test"]
[White "Alice"]
[Black "Bob"]
[WhiteElo "1400"]
[BlackElo "1200"]
[TimeControl "600+0"]
[Result "1-0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 1-0

"""

SAMPLE_PGN_BULLET = """\
[Event "Test"]
[White "Alice"]
[Black "Bob"]
[WhiteElo "2000"]
[BlackElo "1900"]
[TimeControl "60+0"]
[Result "1-0"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 1-0

"""

SAMPLE_PGN_ABANDONED = """\
[Event "Test"]
[White "Alice"]
[Black "Bob"]
[WhiteElo "2000"]
[BlackElo "1900"]
[TimeControl "600+0"]
[Result "1-0"]
[Termination "Abandoned"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O d3 1-0

"""
