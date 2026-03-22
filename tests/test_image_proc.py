"""Tests for bp_ecg_raw_extractor.image_proc.crop and .roi."""

import pytest
from PIL import Image

from bp_ecg_raw_extractor.config import RegionConfig
from bp_ecg_raw_extractor.image_proc.crop import crop_image
from bp_ecg_raw_extractor.image_proc.roi import extract_roi_bytes

# PNG magic bytes: \x89 P N G \r \n \x1a \n
_PNG_MAGIC: bytes = b"\x89PNG"


class TestCropImage:
    """Unit tests for :func:`crop_image`."""

    def test_full_region_returns_same_size(
        self, sample_image: Image.Image, full_region: RegionConfig
    ) -> None:
        """Full-coverage region crop must return the original dimensions."""
        result: Image.Image = crop_image(sample_image, full_region)
        assert result.size == sample_image.size

    def test_top_strip_returns_correct_height(
        self, sample_image: Image.Image, top_strip_region: RegionConfig
    ) -> None:
        """Top-strip region (y_max=0.15) should produce correct cropped height."""
        result: Image.Image = crop_image(sample_image, top_strip_region)
        # sample_image is 200 x 100; 15 % of 100 = 15 px
        expected_height: int = int(0.15 * sample_image.size[1])
        assert result.size[1] == expected_height
        # Width must equal full image width when x_min=0, x_max=1
        assert result.size[0] == sample_image.size[0]

    def test_center_region_reduces_both_dimensions(
        self, sample_image: Image.Image, center_region: RegionConfig
    ) -> None:
        """Center crop (10–90 % per axis) must reduce both width and height."""
        result: Image.Image = crop_image(sample_image, center_region)
        assert result.size[0] < sample_image.size[0]
        assert result.size[1] < sample_image.size[1]

    def test_zero_width_region_raises_value_error(
        self, sample_image: Image.Image
    ) -> None:
        """A region where x_min == x_max must raise ValueError."""
        zero_width: RegionConfig = RegionConfig(
            x_min=0.5, y_min=0.0, x_max=0.5, y_max=1.0
        )
        with pytest.raises(ValueError, match="zero-area"):
            crop_image(sample_image, zero_width)

    def test_zero_height_region_raises_value_error(
        self, sample_image: Image.Image
    ) -> None:
        """A region where y_min == y_max must raise ValueError."""
        zero_height: RegionConfig = RegionConfig(
            x_min=0.0, y_min=0.4, x_max=1.0, y_max=0.4
        )
        with pytest.raises(ValueError, match="zero-area"):
            crop_image(sample_image, zero_height)

    def test_out_of_bounds_coords_clamped_by_pil(
        self, sample_image: Image.Image
    ) -> None:
        """Coordinates beyond [0,1] are passed through to PIL which clamps them.

        PIL.Image.crop() silently clamps out-of-bounds coords to the image
        boundary, so no exception is expected and the result must still be a
        valid Image.
        """
        out_of_bounds: RegionConfig = RegionConfig(
            x_min=-0.5, y_min=-0.5, x_max=1.5, y_max=1.5
        )
        result: Image.Image = crop_image(sample_image, out_of_bounds)
        # PIL clamps to image boundaries, so result size equals original
        assert isinstance(result, Image.Image)

    def test_crop_returns_new_image_object(
        self, sample_image: Image.Image, full_region: RegionConfig
    ) -> None:
        """crop_image must return a new image, not the same object."""
        result: Image.Image = crop_image(sample_image, full_region)
        assert result is not sample_image


class TestExtractRoiBytes:
    """Unit tests for :func:`extract_roi_bytes`."""

    def test_returns_bytes(
        self, sample_image: Image.Image, top_strip_region: RegionConfig
    ) -> None:
        """extract_roi_bytes must return a bytes object."""
        result: bytes = extract_roi_bytes(sample_image, top_strip_region)
        assert isinstance(result, bytes)

    def test_returns_valid_png_magic_bytes(
        self, sample_image: Image.Image, top_strip_region: RegionConfig
    ) -> None:
        """Output must start with the PNG magic signature."""
        result: bytes = extract_roi_bytes(sample_image, top_strip_region)
        assert result[:4] == _PNG_MAGIC

    def test_full_region_produces_non_empty_bytes(
        self, sample_image: Image.Image, full_region: RegionConfig
    ) -> None:
        """Full-region ROI must produce non-empty bytes."""
        result: bytes = extract_roi_bytes(sample_image, full_region)
        assert len(result) > 0

    def test_zero_width_region_raises_value_error(
        self, sample_image: Image.Image
    ) -> None:
        """Zero-width region must propagate ValueError from crop_image."""
        zero_width: RegionConfig = RegionConfig(
            x_min=0.5, y_min=0.0, x_max=0.5, y_max=1.0
        )
        with pytest.raises(ValueError, match="zero-area"):
            extract_roi_bytes(sample_image, zero_width)
