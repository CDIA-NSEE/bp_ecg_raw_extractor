"""Tests for bp_ecg_raw_extractor.decompressor.zstd_reader."""

import pytest
import zstandard
from PIL import Image, UnidentifiedImageError

from bp_ecg_raw_extractor.decompressor.zstd_reader import decompress_to_image


class TestDecompressToImage:
    """Unit tests for :func:`decompress_to_image`."""

    def test_roundtrip_returns_correct_size(
        self,
        sample_compressed_bytes: bytes,
        sample_image: Image.Image,
    ) -> None:
        """Decompress a compressed PNG; resulting image must match original size."""
        result: Image.Image = decompress_to_image(sample_compressed_bytes)
        assert result.size == sample_image.size

    def test_roundtrip_returns_pil_image(self, sample_compressed_bytes: bytes) -> None:
        """decompress_to_image must return a PIL.Image instance."""
        result: Image.Image = decompress_to_image(sample_compressed_bytes)
        assert isinstance(result, Image.Image)

    def test_roundtrip_rgb_mode(self, sample_compressed_bytes: bytes) -> None:
        """Decompressed image should preserve the original RGB colour mode."""
        result: Image.Image = decompress_to_image(sample_compressed_bytes)
        assert result.mode == "RGB"

    def test_empty_bytes_raises_value_error(self) -> None:
        """Empty input must raise ValueError before attempting decompression."""
        with pytest.raises(ValueError, match="empty"):
            decompress_to_image(b"")

    def test_corrupt_bytes_raises_error(self) -> None:
        """Garbage bytes must raise an appropriate exception during decompression."""
        corrupt: bytes = b"\x00\x01\x02\x03corrupt_data_not_zstd"
        with pytest.raises(zstandard.ZstdError):
            decompress_to_image(corrupt)

    def test_valid_zstd_but_not_image_raises_error(self) -> None:
        """Valid zstd payload that is not an image must raise UnidentifiedImageError."""
        cctx: zstandard.ZstdCompressor = zstandard.ZstdCompressor(level=1)
        not_an_image: bytes = cctx.compress(b"this is just plain text, not PNG")
        with pytest.raises(UnidentifiedImageError):
            decompress_to_image(not_an_image)
