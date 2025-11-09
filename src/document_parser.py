"""
Document parsing utilities to extract text from PDF and Word documents.

Supports:
- PDF (typed text via PyPDF2; OCR fallback via pdf2image + pytesseract)
- DOCX (via python-docx)

Notes for OCR on Windows:
- Requires Tesseract OCR installed and available in PATH, or set pytesseract.pytesseract.tesseract_cmd
- For PDF OCR, pdf2image requires Poppler (install and add to PATH)
"""

import io
import os
from typing import Tuple

from pathlib import Path

def _safe_imports():
    modules = {}
    try:
        import docx  # python-docx
        modules['docx'] = docx
    except Exception:
        modules['docx'] = None
    # Prefer pypdf, fall back to PyPDF2 for compatibility
    try:
        from pypdf import PdfReader  # type: ignore
        modules['PdfReader'] = PdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            modules['PdfReader'] = PdfReader
        except Exception:
            modules['PdfReader'] = None
    try:
        from pdf2image import convert_from_bytes
        modules['convert_from_bytes'] = convert_from_bytes
    except Exception:
        modules['convert_from_bytes'] = None
    try:
        import pytesseract
        modules['pytesseract'] = pytesseract
    except Exception:
        modules['pytesseract'] = None
    try:
        from PIL import Image
        modules['Image'] = Image
    except Exception:
        modules['Image'] = None
    return modules


def _read_docx(file_bytes: bytes) -> str:
    mods = _safe_imports()
    if mods['docx'] is None:
        raise RuntimeError("python-docx is not installed. Please add 'python-docx' to requirements and install it.")
    document = mods['docx'].Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs]
    return "\n".join([p for p in paragraphs if p and p.strip()])


def _read_pdf_typed(file_bytes: bytes) -> str:
    mods = _safe_imports()
    if mods['PdfReader'] is None:
        raise RuntimeError("PDF parser not available. Install 'pypdf' (preferred) or 'PyPDF2'.")
    reader = mods['PdfReader'](io.BytesIO(file_bytes))
    texts = []
    for page in getattr(reader, 'pages', []):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            texts.append(txt)
    return "\n".join(texts).strip()


def _read_pdf_ocr(file_bytes: bytes) -> Tuple[str, str]:
    mods = _safe_imports()
    if mods['convert_from_bytes'] is None or mods['pytesseract'] is None:
        raise RuntimeError("OCR dependencies missing. Install 'pdf2image', 'pytesseract', and ensure Poppler & Tesseract are available.")
    images = mods['convert_from_bytes'](file_bytes)
    ocr_texts = []
    for img in images:
        try:
            text = mods['pytesseract'].image_to_string(img)
        except Exception:
            text = ""
        if text:
            ocr_texts.append(text)
    return "\n".join(ocr_texts).strip(), _tesseract_hint()


def _tesseract_hint() -> str:
    return (
        "If OCR text seems empty: Ensure Tesseract is installed (Windows: install Tesseract OCR) "
        "and Poppler is available for pdf2image. Add their bin folders to PATH."
    )


def extract_text_from_file(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    """
    Extract text and return (text, note). Note includes hints/warnings (possibly empty).

    Args:
        file_bytes: Raw file content
        filename: Original filename to infer type

    Returns:
        (text, note)
    """
    suffix = Path(filename).suffix.lower()
    note = ""
    if suffix in [".docx"]:
        text = _read_docx(file_bytes)
        return text, note
    if suffix in [".pdf"]:
        text = _read_pdf_typed(file_bytes)
        if len(text.strip()) >= 50:
            return text, note
        # Fallback to OCR for scanned PDFs
        try:
            ocr_text, hint = _read_pdf_ocr(file_bytes)
            note = hint
            return ocr_text or text, note
        except Exception as e:
            # Return whatever we have and the error note
            return text, f"OCR fallback unavailable: {e}"
    raise ValueError("Unsupported file type. Supported: PDF (.pdf) and Word (.docx)")


