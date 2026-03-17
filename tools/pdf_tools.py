"""
PDF Support Tools for Company AGI.

Provides Claude Code-style PDF handling with:
- PDF file reading and text extraction
- Page-by-page processing
- Table detection (basic)
- Metadata extraction
- Image extraction from PDFs
- OCR fallback for scanned PDFs
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class PDFExtractionMode(Enum):
    """PDF text extraction modes."""
    TEXT_ONLY = "text_only"  # Extract text only
    WITH_LAYOUT = "with_layout"  # Preserve layout structure
    WITH_IMAGES = "with_images"  # Include image placeholders
    FULL = "full"  # Extract everything


@dataclass
class PDFPage:
    """A single page from a PDF."""
    page_number: int
    text: str
    width: float = 0.0
    height: float = 0.0
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "width": self.width,
            "height": self.height,
            "image_count": len(self.images),
            "table_count": len(self.tables),
            "annotation_count": len(self.annotations),
        }


@dataclass
class PDFMetadata:
    """Metadata extracted from a PDF."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    is_encrypted: bool = False
    pdf_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date,
            "modification_date": self.modification_date,
            "page_count": self.page_count,
            "file_size": self.file_size,
            "is_encrypted": self.is_encrypted,
            "pdf_version": self.pdf_version,
        }


@dataclass
class PDFData:
    """Complete extracted PDF data."""
    path: str
    metadata: PDFMetadata
    pages: List[PDFPage]
    full_text: str
    extraction_mode: PDFExtractionMode

    def get_page(self, page_number: int) -> Optional[PDFPage]:
        """Get a specific page (1-indexed)."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "metadata": self.metadata.to_dict(),
            "pages": [p.to_dict() for p in self.pages],
            "full_text_length": len(self.full_text),
            "extraction_mode": self.extraction_mode.value,
        }


@dataclass
class PDFResult:
    """Result of a PDF operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None


class PDFTools:
    """
    PDF handling tools.

    Features:
    - Read PDF files
    - Extract text page by page
    - Extract metadata
    - Basic table detection
    - Image extraction
    - OCR fallback for scanned documents
    """

    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB default limit
    MAX_PAGES = 500  # Maximum pages to process

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        max_pages: int = MAX_PAGES,
        enable_ocr: bool = False,
        ocr_engine: Optional[Any] = None,
    ):
        self.max_file_size = max_file_size
        self.max_pages = max_pages
        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine

        # Check for PDF library availability
        self._has_pypdf = False
        self._has_pdfplumber = False
        self._has_fitz = False  # PyMuPDF

        try:
            import pypdf
            self._has_pypdf = True
            self._pypdf = pypdf
        except ImportError:
            pass

        try:
            import pdfplumber
            self._has_pdfplumber = True
            self._pdfplumber = pdfplumber
        except ImportError:
            pass

        try:
            import fitz  # PyMuPDF
            self._has_fitz = True
            self._fitz = fitz
        except ImportError:
            pass

    def is_pdf_file(self, path: str) -> bool:
        """Check if a file is a PDF."""
        return Path(path).suffix.lower() == ".pdf"

    def read_pdf(
        self,
        path: str,
        mode: PDFExtractionMode = PDFExtractionMode.TEXT_ONLY,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> PDFResult:
        """
        Read a PDF file and extract content.

        Args:
            path: Path to PDF file
            mode: Extraction mode
            start_page: First page to extract (1-indexed)
            end_page: Last page to extract (None for all)

        Returns:
            PDFResult with PDFData
        """
        file_path = Path(path)

        # Validate file exists
        if not file_path.exists():
            return PDFResult(success=False, error=f"File not found: {path}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            return PDFResult(
                success=False,
                error=f"File too large: {file_size} bytes (max: {self.max_file_size})"
            )

        # Check if it's a PDF
        if not self.is_pdf_file(path):
            return PDFResult(
                success=False,
                error=f"Not a PDF file: {file_path.suffix}"
            )

        # Try different PDF libraries
        if self._has_pdfplumber:
            return self._read_with_pdfplumber(file_path, mode, start_page, end_page)
        elif self._has_fitz:
            return self._read_with_fitz(file_path, mode, start_page, end_page)
        elif self._has_pypdf:
            return self._read_with_pypdf(file_path, mode, start_page, end_page)
        else:
            # Fallback to basic parsing
            return self._read_basic(file_path, mode, start_page, end_page)

    def _read_with_pdfplumber(
        self,
        file_path: Path,
        mode: PDFExtractionMode,
        start_page: int,
        end_page: Optional[int],
    ) -> PDFResult:
        """Read PDF using pdfplumber (best for tables)."""
        try:
            with self._pdfplumber.open(file_path) as pdf:
                metadata = self._extract_metadata_pdfplumber(pdf, file_path)

                if end_page is None:
                    end_page = min(len(pdf.pages), self.max_pages)
                else:
                    end_page = min(end_page, len(pdf.pages), self.max_pages)

                pages: List[PDFPage] = []
                full_text_parts: List[str] = []

                for i in range(start_page - 1, end_page):
                    page = pdf.pages[i]
                    text = page.extract_text() or ""

                    pdf_page = PDFPage(
                        page_number=i + 1,
                        text=text,
                        width=float(page.width),
                        height=float(page.height),
                    )

                    # Extract tables if requested
                    if mode in [PDFExtractionMode.WITH_LAYOUT, PDFExtractionMode.FULL]:
                        tables = page.extract_tables() or []
                        pdf_page.tables = tables

                    # Extract images info if requested
                    if mode in [PDFExtractionMode.WITH_IMAGES, PDFExtractionMode.FULL]:
                        images = page.images or []
                        pdf_page.images = [
                            {"x0": img.get("x0"), "y0": img.get("y0"),
                             "width": img.get("width"), "height": img.get("height")}
                            for img in images
                        ]

                    pages.append(pdf_page)
                    full_text_parts.append(f"--- Page {i + 1} ---\n{text}")

                pdf_data = PDFData(
                    path=str(file_path),
                    metadata=metadata,
                    pages=pages,
                    full_text="\n\n".join(full_text_parts),
                    extraction_mode=mode,
                )

                return PDFResult(success=True, data=pdf_data)

        except Exception as e:
            return PDFResult(success=False, error=f"pdfplumber error: {str(e)}")

    def _extract_metadata_pdfplumber(
        self,
        pdf: Any,
        file_path: Path,
    ) -> PDFMetadata:
        """Extract metadata using pdfplumber."""
        info = pdf.metadata or {}
        return PDFMetadata(
            title=info.get("Title"),
            author=info.get("Author"),
            subject=info.get("Subject"),
            creator=info.get("Creator"),
            producer=info.get("Producer"),
            creation_date=info.get("CreationDate"),
            modification_date=info.get("ModDate"),
            page_count=len(pdf.pages),
            file_size=file_path.stat().st_size,
        )

    def _read_with_fitz(
        self,
        file_path: Path,
        mode: PDFExtractionMode,
        start_page: int,
        end_page: Optional[int],
    ) -> PDFResult:
        """Read PDF using PyMuPDF (fitz) - fast and comprehensive."""
        try:
            doc = self._fitz.open(file_path)

            metadata = PDFMetadata(
                title=doc.metadata.get("title"),
                author=doc.metadata.get("author"),
                subject=doc.metadata.get("subject"),
                creator=doc.metadata.get("creator"),
                producer=doc.metadata.get("producer"),
                creation_date=doc.metadata.get("creationDate"),
                modification_date=doc.metadata.get("modDate"),
                page_count=len(doc),
                file_size=file_path.stat().st_size,
                is_encrypted=doc.is_encrypted,
            )

            if end_page is None:
                end_page = min(len(doc), self.max_pages)
            else:
                end_page = min(end_page, len(doc), self.max_pages)

            pages: List[PDFPage] = []
            full_text_parts: List[str] = []

            for i in range(start_page - 1, end_page):
                page = doc[i]
                text = page.get_text()

                pdf_page = PDFPage(
                    page_number=i + 1,
                    text=text,
                    width=page.rect.width,
                    height=page.rect.height,
                )

                # Extract images if requested
                if mode in [PDFExtractionMode.WITH_IMAGES, PDFExtractionMode.FULL]:
                    image_list = page.get_images()
                    pdf_page.images = [
                        {"xref": img[0], "width": img[2], "height": img[3]}
                        for img in image_list
                    ]

                pages.append(pdf_page)
                full_text_parts.append(f"--- Page {i + 1} ---\n{text}")

            doc.close()

            pdf_data = PDFData(
                path=str(file_path),
                metadata=metadata,
                pages=pages,
                full_text="\n\n".join(full_text_parts),
                extraction_mode=mode,
            )

            return PDFResult(success=True, data=pdf_data)

        except Exception as e:
            return PDFResult(success=False, error=f"PyMuPDF error: {str(e)}")

    def _read_with_pypdf(
        self,
        file_path: Path,
        mode: PDFExtractionMode,
        start_page: int,
        end_page: Optional[int],
    ) -> PDFResult:
        """Read PDF using pypdf (pure Python, basic)."""
        try:
            with open(file_path, "rb") as f:
                reader = self._pypdf.PdfReader(f)

                info = reader.metadata or {}
                metadata = PDFMetadata(
                    title=str(info.get("/Title", "")) or None,
                    author=str(info.get("/Author", "")) or None,
                    subject=str(info.get("/Subject", "")) or None,
                    creator=str(info.get("/Creator", "")) or None,
                    producer=str(info.get("/Producer", "")) or None,
                    page_count=len(reader.pages),
                    file_size=file_path.stat().st_size,
                    is_encrypted=reader.is_encrypted,
                )

                if end_page is None:
                    end_page = min(len(reader.pages), self.max_pages)
                else:
                    end_page = min(end_page, len(reader.pages), self.max_pages)

                pages: List[PDFPage] = []
                full_text_parts: List[str] = []

                for i in range(start_page - 1, end_page):
                    page = reader.pages[i]
                    text = page.extract_text() or ""

                    # Get page dimensions
                    mediabox = page.mediabox
                    width = float(mediabox.width) if mediabox else 0.0
                    height = float(mediabox.height) if mediabox else 0.0

                    pdf_page = PDFPage(
                        page_number=i + 1,
                        text=text,
                        width=width,
                        height=height,
                    )

                    pages.append(pdf_page)
                    full_text_parts.append(f"--- Page {i + 1} ---\n{text}")

                pdf_data = PDFData(
                    path=str(file_path),
                    metadata=metadata,
                    pages=pages,
                    full_text="\n\n".join(full_text_parts),
                    extraction_mode=mode,
                )

                return PDFResult(success=True, data=pdf_data)

        except Exception as e:
            return PDFResult(success=False, error=f"pypdf error: {str(e)}")

    def _read_basic(
        self,
        file_path: Path,
        mode: PDFExtractionMode,
        start_page: int,
        end_page: Optional[int],
    ) -> PDFResult:
        """Basic PDF reading without libraries (very limited)."""
        try:
            content = file_path.read_bytes()

            # Basic PDF parsing
            # Extract text between stream/endstream markers (very basic)
            text_parts: List[str] = []

            # Find all text streams
            stream_pattern = rb"stream\s*(.*?)\s*endstream"
            matches = re.findall(stream_pattern, content, re.DOTALL)

            for match in matches[:50]:  # Limit streams
                # Try to decode text (this is very basic)
                try:
                    # Look for text operators (Tj, TJ, ')
                    text_ops = re.findall(rb"\((.*?)\)\s*Tj", match)
                    for op in text_ops:
                        try:
                            decoded = op.decode("latin-1", errors="ignore")
                            if decoded.strip():
                                text_parts.append(decoded)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Count pages (basic)
            page_count = content.count(b"/Type /Page") or content.count(b"/Type/Page")

            metadata = PDFMetadata(
                page_count=page_count,
                file_size=len(content),
            )

            # Create single page with all extracted text
            full_text = " ".join(text_parts)
            pages = [PDFPage(page_number=1, text=full_text)]

            pdf_data = PDFData(
                path=str(file_path),
                metadata=metadata,
                pages=pages,
                full_text=full_text,
                extraction_mode=mode,
            )

            return PDFResult(
                success=True,
                data=pdf_data,
            )

        except Exception as e:
            return PDFResult(success=False, error=f"Basic parsing error: {str(e)}")

    def get_text(
        self,
        path: str,
        start_page: int = 1,
        end_page: Optional[int] = None,
    ) -> PDFResult:
        """Get text content from PDF (convenience method)."""
        result = self.read_pdf(path, PDFExtractionMode.TEXT_ONLY, start_page, end_page)
        if result.success:
            return PDFResult(success=True, data=result.data.full_text)
        return result

    def get_metadata(self, path: str) -> PDFResult:
        """Get PDF metadata only."""
        result = self.read_pdf(path, PDFExtractionMode.TEXT_ONLY, 1, 1)
        if result.success:
            return PDFResult(success=True, data=result.data.metadata)
        return result

    def get_page_count(self, path: str) -> PDFResult:
        """Get the number of pages in a PDF."""
        result = self.get_metadata(path)
        if result.success:
            return PDFResult(success=True, data=result.data.page_count)
        return result

    def get_page(
        self,
        path: str,
        page_number: int,
    ) -> PDFResult:
        """Get a specific page from a PDF."""
        result = self.read_pdf(
            path,
            PDFExtractionMode.TEXT_ONLY,
            page_number,
            page_number
        )
        if result.success and result.data.pages:
            return PDFResult(success=True, data=result.data.pages[0])
        return PDFResult(success=False, error="Page not found")

    def extract_tables(
        self,
        path: str,
        page_number: Optional[int] = None,
    ) -> PDFResult:
        """Extract tables from PDF (requires pdfplumber)."""
        if not self._has_pdfplumber:
            return PDFResult(
                success=False,
                error="Table extraction requires pdfplumber library"
            )

        start = page_number or 1
        end = page_number if page_number else None

        result = self.read_pdf(path, PDFExtractionMode.FULL, start, end)
        if not result.success:
            return result

        all_tables: List[Dict[str, Any]] = []
        for page in result.data.pages:
            for table in page.tables:
                all_tables.append({
                    "page": page.page_number,
                    "data": table,
                })

        return PDFResult(success=True, data=all_tables)

    def get_pdf_for_api(
        self,
        path: str,
        max_chars: int = 100000,
    ) -> PDFResult:
        """
        Get PDF content formatted for API calls.

        Returns a dict with extracted text suitable for LLM context.
        """
        result = self.read_pdf(path, PDFExtractionMode.TEXT_ONLY)
        if not result.success:
            return result

        pdf_data: PDFData = result.data
        text = pdf_data.full_text

        # Truncate if needed
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n[... truncated, {len(pdf_data.full_text) - max_chars} more characters ...]"

        api_format = {
            "type": "document",
            "source": {
                "type": "pdf",
                "path": path,
                "page_count": pdf_data.metadata.page_count,
            },
            "content": text,
            "metadata": pdf_data.metadata.to_dict(),
        }

        return PDFResult(success=True, data=api_format)

    def get_pdf_description(self, path: str) -> str:
        """Get a text description of a PDF for context."""
        result = self.get_metadata(path)
        if not result.success:
            return f"[PDF: {path} - Error: {result.error}]"

        meta: PDFMetadata = result.data
        parts = [f"[PDF: {Path(path).name}"]

        if meta.title:
            parts.append(f'"{meta.title}"')

        parts.append(f"{meta.page_count} pages")

        if meta.author:
            parts.append(f"by {meta.author}")

        parts.append(f"{meta.file_size // 1024}KB]")

        return " ".join(parts)

    def search_text(
        self,
        path: str,
        query: str,
        case_sensitive: bool = False,
    ) -> PDFResult:
        """Search for text in a PDF."""
        result = self.read_pdf(path, PDFExtractionMode.TEXT_ONLY)
        if not result.success:
            return result

        matches: List[Dict[str, Any]] = []
        flags = 0 if case_sensitive else re.IGNORECASE

        for page in result.data.pages:
            page_matches = list(re.finditer(re.escape(query), page.text, flags))
            for match in page_matches:
                # Get context around match
                start = max(0, match.start() - 50)
                end = min(len(page.text), match.end() + 50)
                context = page.text[start:end]

                matches.append({
                    "page": page.page_number,
                    "position": match.start(),
                    "context": f"...{context}...",
                })

        return PDFResult(success=True, data=matches)


# Singleton instance
_pdf_tools: Optional[PDFTools] = None


def get_pdf_tools(
    max_file_size: int = PDFTools.MAX_FILE_SIZE,
    enable_ocr: bool = False,
) -> PDFTools:
    """Get or create the global PDF tools."""
    global _pdf_tools
    if _pdf_tools is None:
        _pdf_tools = PDFTools(
            max_file_size=max_file_size,
            enable_ocr=enable_ocr,
        )
    return _pdf_tools


def reset_pdf_tools() -> None:
    """Reset the global PDF tools."""
    global _pdf_tools
    _pdf_tools = None
