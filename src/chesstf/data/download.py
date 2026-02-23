"""Download Lichess monthly PGN dump files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

_LICHESS_BASE = "https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
_CHUNK_SIZE = 1 << 20  # 1 MiB


def _lichess_url(year: int, month: int) -> str:
    return _LICHESS_BASE.format(year=year, month=month)


def download_lichess_month(
    year: int,
    month: int,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Download a Lichess monthly PGN dump (zstd-compressed).

    If the file already exists and *force* is False the download is skipped
    and the existing path is returned.

    Args:
        year: Four-digit year, e.g. 2023.
        month: Month number 1–12.
        output_dir: Directory in which the file will be saved.
        force: Re-download even if the file already exists.

    Returns:
        :class:`pathlib.Path` pointing to the downloaded ``.pgn.zst`` file.

    Raises:
        requests.HTTPError: If the server returns an error status.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = _lichess_url(year, month)
    filename = url.rsplit("/", 1)[-1]
    dest = output_dir / filename

    if dest.exists() and not force:
        print(f"Already exists, skipping download: {dest}")
        return dest

    print(f"Downloading {url} → {dest}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    hasher = hashlib.sha256()

    with (
        open(dest, "wb") as fh,
        tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=filename,
            leave=True,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            fh.write(chunk)
            hasher.update(chunk)
            bar.update(len(chunk))

    sha256 = hasher.hexdigest()
    checksum_path = dest.with_suffix(dest.suffix + ".sha256")
    checksum_path.write_text(f"{sha256}  {filename}\n")
    print(f"SHA-256: {sha256}")

    return dest
