"""Tests for download.py — mocks all HTTP calls."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from chesstf.data.download import _lichess_url, download_lichess_month

if TYPE_CHECKING:
    from pathlib import Path


class TestLichessUrl:
    def test_url_format(self) -> None:
        url = _lichess_url(2023, 1)
        assert "2023-01" in url
        assert url.endswith(".pgn.zst")
        assert url.startswith("https://")

    def test_url_zero_pads_month(self) -> None:
        url = _lichess_url(2024, 6)
        assert "2024-06" in url

    def test_url_double_digit_month(self) -> None:
        url = _lichess_url(2024, 12)
        assert "2024-12" in url


class TestDownloadLichessMonth:
    def _make_mock_response(self, content: bytes) -> MagicMock:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.headers = {"content-length": str(len(content))}
        # iter_content returns chunks
        mock_resp.iter_content = MagicMock(return_value=iter([content]))
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    def test_download_creates_file(self, tmp_path: Path) -> None:
        fake_content = b"fake zst content"
        mock_resp = self._make_mock_response(fake_content)

        with patch("chesstf.data.download.requests.get", return_value=mock_resp):
            dest = download_lichess_month(2023, 1, tmp_path)

        assert dest.exists()
        assert dest.suffix == ".zst"
        assert dest.read_bytes() == fake_content

    def test_download_writes_checksum_file(self, tmp_path: Path) -> None:
        fake_content = b"fake zst content"
        mock_resp = self._make_mock_response(fake_content)

        with patch("chesstf.data.download.requests.get", return_value=mock_resp):
            dest = download_lichess_month(2023, 1, tmp_path)

        checksum_path = dest.with_suffix(dest.suffix + ".sha256")
        assert checksum_path.exists()
        checksum_text = checksum_path.read_text()
        assert dest.name in checksum_text

    def test_skip_if_exists(self, tmp_path: Path) -> None:
        # Pre-create the file
        url = _lichess_url(2023, 1)
        filename = url.rsplit("/", 1)[-1]
        existing = tmp_path / filename
        existing.write_bytes(b"old content")

        with patch("chesstf.data.download.requests.get") as mock_get:
            dest = download_lichess_month(2023, 1, tmp_path, force=False)
            mock_get.assert_not_called()

        assert dest == existing

    def test_force_redownload(self, tmp_path: Path) -> None:
        url = _lichess_url(2023, 1)
        filename = url.rsplit("/", 1)[-1]
        existing = tmp_path / filename
        existing.write_bytes(b"old content")

        new_content = b"new content"
        mock_resp = self._make_mock_response(new_content)

        with patch("chesstf.data.download.requests.get", return_value=mock_resp):
            dest = download_lichess_month(2023, 1, tmp_path, force=True)

        assert dest.read_bytes() == new_content

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "deep" / "dir"
        fake_content = b"data"
        mock_resp = self._make_mock_response(fake_content)

        with patch("chesstf.data.download.requests.get", return_value=mock_resp):
            dest = download_lichess_month(2023, 1, new_dir)

        assert new_dir.exists()
        assert dest.exists()

    def test_http_error_propagates(self, tmp_path: Path) -> None:
        import requests as req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req.HTTPError("404 Not Found")
        mock_resp.headers = {}

        with (
            patch("chesstf.data.download.requests.get", return_value=mock_resp),
            pytest.raises(req.HTTPError),
        ):
            download_lichess_month(2023, 1, tmp_path)

    def test_returned_path_matches_year_month(self, tmp_path: Path) -> None:
        fake_content = b"data"
        mock_resp = self._make_mock_response(fake_content)

        with patch("chesstf.data.download.requests.get", return_value=mock_resp):
            dest = download_lichess_month(2023, 3, tmp_path)

        assert "2023-03" in dest.name
