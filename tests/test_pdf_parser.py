"""Tests for the PDF parser module.

pdfplumber IS installed as a regular dependency.  PDF fixtures are generated
programmatically via reportlab — no binary PDF files are committed to the repo.
"""

from __future__ import annotations

from io import BytesIO

from reportlab.pdfgen import canvas

from bp_ecg_raw_extractor.pdf_parser.pdfplumber_parser import (
    extract_text,
    get_pdf_metadata,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_pdf(text: str) -> bytes:
    """Return a single-page PDF with *text* drawn at position (100, 750).

    Args:
        text: The string to embed on the first page.

    Returns:
        Raw PDF bytes (in memory, never written to disk).
    """
    buf: BytesIO = BytesIO()
    c: canvas.Canvas = canvas.Canvas(buf)
    c.drawString(100, 750, text)
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_multipage_pdf(texts: list[str]) -> bytes:
    """Return a multi-page PDF with one text string per page.

    Args:
        texts: List of strings; each entry becomes one PDF page.

    Returns:
        Raw PDF bytes.
    """
    buf: BytesIO = BytesIO()
    c: canvas.Canvas = canvas.Canvas(buf)
    for t in texts:
        c.drawString(100, 750, t)
        c.showPage()
    c.save()
    return buf.getvalue()


def _make_empty_pdf() -> bytes:
    """Return a single-page PDF with no drawn text content."""
    buf: BytesIO = BytesIO()
    c: canvas.Canvas = canvas.Canvas(buf)
    c.showPage()
    c.save()
    return buf.getvalue()


# ---------------------------------------------------------------------------
# TestExtractText
# ---------------------------------------------------------------------------


class TestExtractText:
    """Tests for extract_text()."""

    def test_extracts_text_from_pdf(self) -> None:
        """Text embedded in a PDF is extracted correctly."""
        pdf_bytes: bytes = _make_pdf("hello world")
        result: str = extract_text(pdf_bytes)
        assert "hello world" in result

    def test_multipage_pdf_text_joined(self) -> None:
        """Text from multiple pages is joined with the '---' separator."""
        pdf_bytes: bytes = _make_multipage_pdf(["first page", "second page"])
        result: str = extract_text(pdf_bytes)
        assert "first page" in result
        assert "second page" in result
        assert "---" in result

    def test_empty_pdf_returns_empty_string(self) -> None:
        """A PDF with no text content returns an empty or whitespace-only string."""
        pdf_bytes: bytes = _make_empty_pdf()
        result: str = extract_text(pdf_bytes)
        assert result.strip() == ""

    def test_corrupt_bytes_returns_empty_string(self) -> None:
        """Corrupt bytes that cannot be parsed as PDF return an empty string."""
        result: str = extract_text(b"not a pdf")
        assert result == ""


# ---------------------------------------------------------------------------
# TestGetPdfMetadata
# ---------------------------------------------------------------------------


class TestGetPdfMetadata:
    """Tests for get_pdf_metadata()."""

    def test_returns_dict_with_producer_and_creator_keys(self) -> None:
        """The returned dict always contains 'producer' and 'creator' keys."""
        pdf_bytes: bytes = _make_pdf("metadata test")
        result: dict[str, str | None] = get_pdf_metadata(pdf_bytes)
        assert "producer" in result
        assert "creator" in result

    def test_producer_is_string_or_none(self) -> None:
        """The 'producer' value is either a str or None."""
        pdf_bytes: bytes = _make_pdf("producer test")
        result: dict[str, str | None] = get_pdf_metadata(pdf_bytes)
        assert isinstance(result["producer"], (str, type(None)))

    def test_creator_is_string_or_none(self) -> None:
        """The 'creator' value is either a str or None."""
        pdf_bytes: bytes = _make_pdf("creator test")
        result: dict[str, str | None] = get_pdf_metadata(pdf_bytes)
        assert isinstance(result["creator"], (str, type(None)))
