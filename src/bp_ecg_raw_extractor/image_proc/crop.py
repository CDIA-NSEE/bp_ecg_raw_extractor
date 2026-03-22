"""Image cropping utilities using relative region coordinates.

All coordinates are expressed as fractions of the image dimensions (0.0–1.0).
PIL.Image.crop() is used for the actual pixel-level crop operation.
"""

from PIL import Image

from bp_ecg_raw_extractor.config import RegionConfig


def crop_image(image: Image.Image, region: RegionConfig) -> Image.Image:
    """Crop *image* to the area described by *region*.

    Relative coordinates are converted to absolute pixel coordinates before
    passing to :py:meth:`PIL.Image.Image.crop`.  PIL's crop() handles
    out-of-bounds coordinates gracefully by clamping them to the image
    boundaries.

    Args:
        image: The source PIL image.
        region: Relative coordinates defining the crop area (0.0–1.0 per axis).

    Returns:
        A new PIL Image representing the cropped region.

    Raises:
        ValueError: If the resulting crop box has zero or negative width/height
            after pixel conversion (e.g. x_min == x_max after rounding).
    """
    width: int
    height: int
    width, height = image.size

    left: int = int(region.x_min * width)
    upper: int = int(region.y_min * height)
    right: int = int(region.x_max * width)
    lower: int = int(region.y_max * height)

    if right <= left or lower <= upper:
        raise ValueError(
            f"Crop region produces a zero-area box: "
            f"({left}, {upper}, {right}, {lower}) for image size {image.size}."
        )

    return image.crop((left, upper, right, lower))
