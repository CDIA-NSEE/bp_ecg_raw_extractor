"""Tests for the OCR module.

PaddleOCR is not installed in the dev environment (it is an optional extra).
All PaddleOCR interactions are therefore mocked.
"""

from __future__ import annotations

import builtins
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 20, height: int = 10) -> bytes:
    """Return minimal PNG bytes for a solid-colour image."""
    img: Image.Image = Image.new("RGB", (width, height), color=(200, 200, 200))
    buf: BytesIO = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TestGetOcrInstance
# ---------------------------------------------------------------------------


class TestGetOcrInstance:
    """Tests for the lazy PaddleOCR singleton initialiser."""

    def test_raises_import_error_when_paddleocr_not_installed(self) -> None:
        """ImportError raised when paddleocr package is missing."""
        import bp_ecg_raw_extractor.ocr.paddle_ocr as mod

        # Reset singleton so _get_ocr_instance attempts a fresh import.
        original: Any = mod._ocr_instance
        mod._ocr_instance = None
        original_flag: bool = mod._import_attempted
        mod._import_attempted = False

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "paddleocr":
                raise ImportError("no module named paddleocr")
            return real_import(name, *args, **kwargs)

        try:
            with (
                patch("builtins.__import__", side_effect=mock_import),
                pytest.raises(ImportError, match="paddlepaddle and paddleocr"),
            ):
                mod._get_ocr_instance()
        finally:
            mod._ocr_instance = original
            mod._import_attempted = original_flag

    def test_returns_mock_instance(self) -> None:
        """A PaddleOCR instance is returned when the package is importable."""
        import bp_ecg_raw_extractor.ocr.paddle_ocr as mod

        original: Any = mod._ocr_instance
        mod._ocr_instance = None

        fake_ocr: MagicMock = MagicMock()

        try:
            with patch(
                "bp_ecg_raw_extractor.ocr.paddle_ocr._get_ocr_instance",
                return_value=fake_ocr,
            ):
                # Call through the patch to verify a non-None value is returned.
                result: Any = mod._get_ocr_instance()
                assert result is fake_ocr
        finally:
            mod._ocr_instance = original


# ---------------------------------------------------------------------------
# TestRunOcrSync
# ---------------------------------------------------------------------------


class TestRunOcrSync:
    """Tests for run_ocr_sync."""

    def test_returns_list_of_dicts(self) -> None:
        """run_ocr_sync returns a list of text/confidence dicts."""
        from bp_ecg_raw_extractor.ocr.paddle_ocr import run_ocr_sync

        fake_result: list[list[Any]] = [
            [
                ["word1", ["hello", 0.99]],
                ["word2", ["world", 0.95]],
            ]
        ]
        fake_ocr: MagicMock = MagicMock()
        fake_ocr.ocr.return_value = fake_result

        roi: bytes = _make_png_bytes()

        with patch(
            "bp_ecg_raw_extractor.ocr.paddle_ocr._get_ocr_instance",
            return_value=fake_ocr,
        ):
            results: list[dict[str, Any]] = run_ocr_sync(roi)

        assert isinstance(results, list)
        assert len(results) == 2
        for item in results:
            assert "text" in item
            assert "confidence" in item
        assert results[0]["text"] == "hello"
        assert results[0]["confidence"] == pytest.approx(0.99)
        assert results[1]["text"] == "world"
        assert results[1]["confidence"] == pytest.approx(0.95)

    def test_empty_ocr_result_returns_empty_list(self) -> None:
        """Empty inner block returns an empty list."""
        from bp_ecg_raw_extractor.ocr.paddle_ocr import run_ocr_sync

        fake_ocr: MagicMock = MagicMock()
        fake_ocr.ocr.return_value = [[]]

        roi: bytes = _make_png_bytes()

        with patch(
            "bp_ecg_raw_extractor.ocr.paddle_ocr._get_ocr_instance",
            return_value=fake_ocr,
        ):
            results: list[dict[str, Any]] = run_ocr_sync(roi)

        assert results == []

    def test_none_ocr_result_returns_empty_list(self) -> None:
        """None returned by .ocr() yields an empty list."""
        from bp_ecg_raw_extractor.ocr.paddle_ocr import run_ocr_sync

        fake_ocr: MagicMock = MagicMock()
        fake_ocr.ocr.return_value = None

        roi: bytes = _make_png_bytes()

        with patch(
            "bp_ecg_raw_extractor.ocr.paddle_ocr._get_ocr_instance",
            return_value=fake_ocr,
        ):
            results: list[dict[str, Any]] = run_ocr_sync(roi)

        assert results == []


# ---------------------------------------------------------------------------
# TestGetOcrTextAndConfidence
# ---------------------------------------------------------------------------


class TestGetOcrTextAndConfidence:
    """Tests for get_ocr_text_and_confidence."""

    def test_returns_text_and_mean_confidence(self) -> None:
        """Text is joined and mean confidence is computed correctly."""
        from bp_ecg_raw_extractor.ocr.paddle_ocr import get_ocr_text_and_confidence

        mock_items: list[dict[str, Any]] = [
            {"text": "hello", "confidence": 0.9},
            {"text": "world", "confidence": 0.8},
        ]

        with patch(
            "bp_ecg_raw_extractor.ocr.paddle_ocr.run_ocr_sync",
            return_value=mock_items,
        ):
            text, confidence = get_ocr_text_and_confidence(b"fake_bytes")

        assert text == "hello world"
        assert confidence == pytest.approx(0.85)

    def test_empty_result_returns_empty_string_and_zero(self) -> None:
        """Empty OCR output yields an empty string and 0.0 confidence."""
        from bp_ecg_raw_extractor.ocr.paddle_ocr import get_ocr_text_and_confidence

        with patch(
            "bp_ecg_raw_extractor.ocr.paddle_ocr.run_ocr_sync",
            return_value=[],
        ):
            text, confidence = get_ocr_text_and_confidence(b"fake_bytes")

        assert text == ""
        assert confidence == 0.0
