"""PDF text extraction using pdfplumber.

All extraction is performed on in-memory bytes — no files are written to disk.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pdfplumber
import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def extract_text(pdf_bytes: bytes) -> str:
    """Extract and concatenate text from all pages of a PDF.

    Pages are joined with ``"\\n---\\n"`` as a separator.  Leading and trailing
    whitespace is stripped from the final result.

    Args:
        pdf_bytes: Raw PDF file contents.

    Returns:
        The extracted text string.  Returns an empty string when no text is
        found or when the bytes cannot be parsed as a valid PDF.
    """
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            page_texts: list[str] = []
            for page in pdf.pages:
                text: str | None = page.extract_text()
                if text:
                    page_texts.append(text)
            joined: str = "\n---\n".join(page_texts)
            return joined.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pdf_text_extraction_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return ""


def get_pdf_metadata(pdf_bytes: bytes) -> dict[str, str | None]:
    """Extract producer and creator metadata from a PDF.

    Args:
        pdf_bytes: Raw PDF file contents.

    Returns:
        A dict with keys ``"producer"`` and ``"creator"``, each holding the
        corresponding metadata value as a string or ``None`` when absent.
    """
    result: dict[str, str | None] = {"producer": None, "creator": None}
    try:
        with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
            meta: dict[str, Any] | None = pdf.metadata
            if meta:
                raw_producer: Any = meta.get("Producer")
                raw_creator: Any = meta.get("Creator")
                result["producer"] = (
                    str(raw_producer) if raw_producer is not None else None
                )
                result["creator"] = (
                    str(raw_creator) if raw_creator is not None else None
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "pdf_metadata_extraction_failed",
            error=str(exc),
            error_type=type(exc).__name__,
        )
    return result
