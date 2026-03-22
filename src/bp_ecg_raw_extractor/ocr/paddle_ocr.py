"""PaddleOCR wrapper for ECG image text recognition.

PaddleOCR and paddlepaddle are optional dependencies.  They are not installed
in the default dev environment — install the ``ocr`` extra to enable them::

    uv pip install 'bp-ecg-raw-extractor[ocr]'

At runtime this module is executed inside a ``ProcessPoolExecutor`` worker
(spawned, not forked) so that PaddleOCR's C++ internals cannot corrupt the
asyncio event loop.  The OCR model is initialised lazily — once per worker
process — via a module-level singleton.
"""

from __future__ import annotations

import logging
from io import BytesIO
from typing import Any

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Module-level singleton; None until first call to _get_ocr_instance().
_ocr_instance: Any = None

# Sentinel so the import is attempted only once per process even on failure.
_import_attempted: bool = False


def _get_ocr_instance(use_gpu: bool = False, lang: str = "pt") -> Any:
    """Return (or lazily create) the process-level PaddleOCR singleton.

    Args:
        use_gpu: Whether to enable GPU inference.
        lang: Language code passed to PaddleOCR.

    Returns:
        An initialised ``PaddleOCR`` instance.

    Raises:
        ImportError: If ``paddlepaddle`` / ``paddleocr`` are not installed.
    """
    global _ocr_instance, _import_attempted

    if _ocr_instance is not None:
        return _ocr_instance

    _import_attempted = True
    try:
        from paddleocr import PaddleOCR  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "paddlepaddle and paddleocr are required for OCR. "
            "Install with: uv pip install 'bp-ecg-raw-extractor[ocr]'"
        ) from exc

    logger.info("initialising_paddleocr", use_gpu=use_gpu, lang=lang)
    # Suppress verbose PaddleOCR logging at the Python level.
    logging.getLogger("ppocr").setLevel(logging.WARNING)
    _ocr_instance = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
    return _ocr_instance


def run_ocr_sync(
    roi_bytes: bytes,
    use_gpu: bool = False,
    lang: str = "pt",
) -> list[dict[str, Any]]:
    """Run OCR on *roi_bytes* and return structured results.

    This function is designed to be called from a ``ProcessPoolExecutor``
    worker (spawned context).  The PaddleOCR model is initialised lazily on
    the first call and reused for every subsequent call within the same worker
    process.

    Args:
        roi_bytes: PNG-encoded bytes of the region of interest.
        use_gpu: Whether to use GPU inference.
        lang: Language code for PaddleOCR.

    Returns:
        A list of ``{"text": str, "confidence": float}`` dicts — one entry per
        recognised text line.  Returns an empty list when OCR finds nothing.
    """
    import numpy as np  # transitive dep of paddlepaddle; safe inside worker
    from PIL import Image

    ocr: Any = _get_ocr_instance(use_gpu=use_gpu, lang=lang)

    img: Image.Image = Image.open(BytesIO(roi_bytes))
    img_array: np.ndarray[Any, np.dtype[Any]] = np.array(img)

    raw: Any = ocr.ocr(img_array, cls=True)

    results: list[dict[str, Any]] = []
    if not raw:
        return results

    for block in raw:
        if not block:
            continue
        for line in block:
            if not line or len(line) < 2:
                continue
            text_info: Any = line[1]
            if not text_info or len(text_info) < 2:
                continue
            text: str = str(text_info[0])
            confidence: float = float(text_info[1])
            results.append({"text": text, "confidence": confidence})

    return results


def get_ocr_text_and_confidence(
    roi_bytes: bytes,
    use_gpu: bool = False,
    lang: str = "pt",
) -> tuple[str, float]:
    """Return the concatenated OCR text and mean confidence for *roi_bytes*.

    Args:
        roi_bytes: PNG-encoded bytes of the region of interest.
        use_gpu: Whether to use GPU inference.
        lang: Language code for PaddleOCR.

    Returns:
        A ``(joined_text, mean_confidence)`` tuple where *joined_text* is all
        recognised words joined with a space and *mean_confidence* is the
        arithmetic mean of per-line confidence scores.  Returns ``("", 0.0)``
        when no text is detected.
    """
    items: list[dict[str, Any]] = run_ocr_sync(roi_bytes, use_gpu=use_gpu, lang=lang)
    if not items:
        return "", 0.0

    texts: list[str] = [item["text"] for item in items]
    confidences: list[float] = [item["confidence"] for item in items]
    joined_text: str = " ".join(texts)
    mean_confidence: float = sum(confidences) / len(confidences)
    return joined_text, mean_confidence
