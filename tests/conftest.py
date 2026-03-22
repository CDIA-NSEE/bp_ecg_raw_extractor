"""Shared pytest fixtures for bp_ecg_raw_extractor tests.

Provides in-memory PNG images, zstd-compressed versions thereof, and
RegionConfig instances used across multiple test modules.  No files are
written to disk.
"""

from io import BytesIO

import pytest
import zstandard
from PIL import Image

from bp_ecg_raw_extractor.config import RegionConfig

# ---------------------------------------------------------------------------
# Image fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_image() -> Image.Image:
    """Return a small (200 x 100) solid-colour RGB PIL image."""
    img: Image.Image = Image.new("RGB", (200, 100), color=(128, 64, 32))
    return img


@pytest.fixture()
def sample_png_bytes(sample_image: Image.Image) -> bytes:
    """Return *sample_image* encoded as PNG bytes (in memory)."""
    buf: BytesIO = BytesIO()
    sample_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def sample_compressed_bytes(sample_png_bytes: bytes) -> bytes:
    """Return *sample_png_bytes* compressed with zstd level=1."""
    cctx: zstandard.ZstdCompressor = zstandard.ZstdCompressor(level=1)
    return cctx.compress(sample_png_bytes)


# ---------------------------------------------------------------------------
# RegionConfig fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_region() -> RegionConfig:
    """RegionConfig covering the entire image (0,0 → 1,1)."""
    return RegionConfig(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)


@pytest.fixture()
def top_strip_region() -> RegionConfig:
    """RegionConfig covering the top 15 % of the image (OCR region)."""
    return RegionConfig(x_min=0.0, y_min=0.0, x_max=1.0, y_max=0.15)


@pytest.fixture()
def center_region() -> RegionConfig:
    """RegionConfig for a centred crop (10–90 % on each axis)."""
    return RegionConfig(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
