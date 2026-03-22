"""Streaming zstd decompression utilities.

Decompresses a zstd-compressed byte payload entirely in memory and returns a
PIL.Image.  No intermediate files are written to disk at any point.
"""

from io import BytesIO

import zstandard
from PIL import Image


def decompress_to_image(compressed_bytes: bytes) -> Image.Image:
    """Decompress a zstd-compressed PNG payload and return a PIL.Image.

    Args:
        compressed_bytes: Raw bytes of a .png.zst file as produced by the
            bp_ecg_file_watcher compressor.

    Returns:
        A PIL Image opened from the decompressed PNG data.

    Raises:
        ValueError: If *compressed_bytes* is empty.
        zstandard.ZstdError: If the bytes are not valid zstd data.
        PIL.UnidentifiedImageError: If the decompressed data is not a valid image.
    """
    if not compressed_bytes:
        raise ValueError("compressed_bytes must not be empty")

    dctx: zstandard.ZstdDecompressor = zstandard.ZstdDecompressor()
    source: BytesIO = BytesIO(compressed_bytes)

    with dctx.stream_reader(source) as reader:
        decompressed: bytes = reader.read()

    return Image.open(BytesIO(decompressed))
