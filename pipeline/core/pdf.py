"""Shared PDF utility for text extraction."""

import io
import os
import tempfile
from typing import Any, Optional

import requests


def extract_text_from_pdf(url: str, logger: Optional[Any] = None) -> Optional[str]:
    """Download and extract text content from a PDF.

    Uses pypdf first (fast), falls back to doctr OCR for scanned documents.
    """

    def _log(msg: str, level: str = "info"):
        if logger:
            getattr(logger, level)(msg)

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "EU-Lobby-Pipeline/1.0 (EU Parliament Transparency Project)"},
            timeout=60,
        )
        response.raise_for_status()
        content = response.content

        # Try pypdf first (fast, works for text-based PDFs)
        try:
            from pypdf import PdfReader

            pdf_file = io.BytesIO(content)
            reader = PdfReader(pdf_file)

            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)

            text_content = "\n\n".join(text_parts)
            if text_content.strip():
                _log(f"Extracted {len(text_content)} chars from PDF (pypdf): {url}")
                return text_content

        except ImportError:
            _log("pypdf not installed, skipping direct text extraction", "warning")
        except Exception as e:
            _log(f"pypdf failed: {e}", "warning")

        # Fallback to doctr OCR for scanned/image-based PDFs
        try:
            from doctr.io import DocumentFile
            from doctr.models import ocr_predictor

            _log(f"PDF appears scanned or pypdf failed, using OCR: {url}")

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(content)
                temp_path = f.name

            try:
                model = ocr_predictor(pretrained=True)
                doc = DocumentFile.from_pdf(temp_path)
                result = model(doc)
                text_content = result.render()

                _log(f"Extracted {len(text_content)} chars from PDF (OCR): {url}")
                return text_content if text_content.strip() else None
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except ImportError:
            _log("doctr not installed, cannot OCR scanned PDF", "warning")
            return None

    except Exception as e:
        _log(f"Error extracting PDF content from {url}: {e}", "warning")
        return None
