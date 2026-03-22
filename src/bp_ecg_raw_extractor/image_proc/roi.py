"""Region-of-interest (ROI) extraction for OCR input.

Extracts a sub-region of an image defined by relative coordinates and returns
it as raw PNG bytes suitable for passing to an OCR engine.  All operations are
performed in memory — no files are written to disk.
"""

from io import BytesIO

from PIL import Image

from bp_ecg_raw_extractor.config import RegionConfig
from bp_ecg_raw_extractor.image_proc.crop import crop_image


def extract_roi_bytes(image: Image.Image, region: RegionConfig) -> bytes:
    """Extract the OCR region from *image* and return PNG-encoded bytes.

    Args:
        image: The source PIL image (typically the full decompressed page image).
        region: Relative coordinates defining the OCR region (0.0–1.0 per axis).

    Returns:
        PNG-encoded bytes of the extracted ROI, suitable for passing to an OCR
        engine or for further in-memory processing.

    Raises:
        ValueError: Propagated from :func:`crop_image` when the region produces
            a zero-area box.
    """
    roi: Image.Image = crop_image(image, region)
    buf: BytesIO = BytesIO()
    roi.save(buf, format="PNG")
    return buf.getvalue()
