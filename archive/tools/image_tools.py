"""
Image Support Tools for Company AGI.

Provides Claude Code-style image handling with:
- Image file reading (PNG, JPG, GIF, WebP)
- Screenshot support
- Base64 encoding for API
- Image metadata extraction
- Optional OCR support
"""

import base64
import io
import mimetypes
import struct
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ImageFormat(Enum):
    """Supported image formats."""
    PNG = "png"
    JPEG = "jpeg"
    GIF = "gif"
    WEBP = "webp"
    BMP = "bmp"
    TIFF = "tiff"
    UNKNOWN = "unknown"


@dataclass
class ImageMetadata:
    """Metadata extracted from an image."""
    format: ImageFormat
    width: int
    height: int
    file_size: int
    color_depth: Optional[int] = None
    has_alpha: bool = False
    is_animated: bool = False
    frame_count: int = 1
    mime_type: str = "image/unknown"


@dataclass
class ImageData:
    """Processed image data."""
    path: str
    metadata: ImageMetadata
    base64_data: str
    thumbnail_base64: Optional[str] = None
    text_content: Optional[str] = None  # OCR result if available


@dataclass
class ImageResult:
    """Result of an image operation."""
    success: bool
    data: Any = None
    error: Optional[str] = None


class ImageTools:
    """
    Image handling tools.

    Features:
    - Read various image formats
    - Extract metadata
    - Base64 encoding for API calls
    - Optional OCR text extraction
    """

    SUPPORTED_EXTENSIONS = {
        ".png": ImageFormat.PNG,
        ".jpg": ImageFormat.JPEG,
        ".jpeg": ImageFormat.JPEG,
        ".gif": ImageFormat.GIF,
        ".webp": ImageFormat.WEBP,
        ".bmp": ImageFormat.BMP,
        ".tiff": ImageFormat.TIFF,
        ".tif": ImageFormat.TIFF,
    }

    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB default limit

    def __init__(
        self,
        max_file_size: int = MAX_FILE_SIZE,
        enable_ocr: bool = False,
        ocr_engine: Optional[Any] = None,
    ):
        self.max_file_size = max_file_size
        self.enable_ocr = enable_ocr
        self.ocr_engine = ocr_engine

        # Check for PIL availability
        self._has_pil = False
        try:
            from PIL import Image
            self._has_pil = True
            self._pil_image = Image
        except ImportError:
            pass

    def is_image_file(self, path: str) -> bool:
        """Check if a file is a supported image."""
        ext = Path(path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def get_format(self, path: str) -> ImageFormat:
        """Get the image format from file extension."""
        ext = Path(path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, ImageFormat.UNKNOWN)

    def read_image(self, path: str) -> ImageResult:
        """
        Read an image file and return processed data.

        Returns ImageData with:
        - Base64 encoded content
        - Metadata (dimensions, format, etc.)
        - OCR text if enabled
        """
        file_path = Path(path)

        # Validate file exists
        if not file_path.exists():
            return ImageResult(success=False, error=f"File not found: {path}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            return ImageResult(
                success=False,
                error=f"File too large: {file_size} bytes (max: {self.max_file_size})"
            )

        # Check if it's a supported image
        if not self.is_image_file(path):
            return ImageResult(
                success=False,
                error=f"Unsupported image format: {file_path.suffix}"
            )

        try:
            # Read file content
            content = file_path.read_bytes()

            # Get metadata
            metadata = self._extract_metadata(content, file_path)

            # Encode to base64
            base64_data = base64.b64encode(content).decode('utf-8')

            # Optional OCR
            text_content = None
            if self.enable_ocr:
                text_content = self._extract_text(content)

            image_data = ImageData(
                path=str(file_path),
                metadata=metadata,
                base64_data=base64_data,
                text_content=text_content,
            )

            return ImageResult(success=True, data=image_data)

        except Exception as e:
            return ImageResult(success=False, error=str(e))

    def _extract_metadata(
        self,
        content: bytes,
        file_path: Path,
    ) -> ImageMetadata:
        """Extract metadata from image content."""
        file_size = len(content)
        fmt = self.get_format(str(file_path))
        mime_type = mimetypes.guess_type(str(file_path))[0] or "image/unknown"

        # Try PIL first if available
        if self._has_pil:
            try:
                return self._extract_metadata_pil(content, fmt, file_size, mime_type)
            except Exception:
                pass

        # Fallback to manual parsing
        return self._extract_metadata_manual(content, fmt, file_size, mime_type)

    def _extract_metadata_pil(
        self,
        content: bytes,
        fmt: ImageFormat,
        file_size: int,
        mime_type: str,
    ) -> ImageMetadata:
        """Extract metadata using PIL."""
        img = self._pil_image.open(io.BytesIO(content))

        width, height = img.size
        has_alpha = img.mode in ('RGBA', 'LA', 'PA')

        # Check for animation
        is_animated = False
        frame_count = 1
        try:
            frame_count = getattr(img, 'n_frames', 1)
            is_animated = frame_count > 1
        except Exception:
            pass

        # Color depth
        color_depth = None
        mode_depths = {
            '1': 1, 'L': 8, 'P': 8, 'RGB': 24, 'RGBA': 32,
            'CMYK': 32, 'YCbCr': 24, 'LAB': 24, 'HSV': 24,
            'I': 32, 'F': 32, 'LA': 16, 'PA': 16,
        }
        color_depth = mode_depths.get(img.mode)

        return ImageMetadata(
            format=fmt,
            width=width,
            height=height,
            file_size=file_size,
            color_depth=color_depth,
            has_alpha=has_alpha,
            is_animated=is_animated,
            frame_count=frame_count,
            mime_type=mime_type,
        )

    def _extract_metadata_manual(
        self,
        content: bytes,
        fmt: ImageFormat,
        file_size: int,
        mime_type: str,
    ) -> ImageMetadata:
        """Extract metadata without PIL (basic parsing)."""
        width = height = 0
        has_alpha = False
        is_animated = False
        color_depth = None

        try:
            if fmt == ImageFormat.PNG:
                width, height, color_depth, has_alpha = self._parse_png_header(content)
            elif fmt == ImageFormat.JPEG:
                width, height = self._parse_jpeg_header(content)
                color_depth = 24
            elif fmt == ImageFormat.GIF:
                width, height, is_animated = self._parse_gif_header(content)
                color_depth = 8
            elif fmt == ImageFormat.BMP:
                width, height, color_depth = self._parse_bmp_header(content)
        except Exception:
            pass

        return ImageMetadata(
            format=fmt,
            width=width,
            height=height,
            file_size=file_size,
            color_depth=color_depth,
            has_alpha=has_alpha,
            is_animated=is_animated,
            frame_count=1,
            mime_type=mime_type,
        )

    def _parse_png_header(self, content: bytes) -> Tuple[int, int, int, bool]:
        """Parse PNG header for dimensions and color info."""
        # PNG signature: 89 50 4E 47 0D 0A 1A 0A
        if content[:8] != b'\x89PNG\r\n\x1a\n':
            raise ValueError("Invalid PNG signature")

        # IHDR chunk starts at byte 8
        # Length (4) + Type (4) + Width (4) + Height (4) + Bit depth (1) + Color type (1)
        width = struct.unpack('>I', content[16:20])[0]
        height = struct.unpack('>I', content[20:24])[0]
        bit_depth = content[24]
        color_type = content[25]

        # Color depth calculation
        samples_per_pixel = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color_type, 1)
        color_depth = bit_depth * samples_per_pixel

        # Alpha channel present in color types 4 and 6
        has_alpha = color_type in (4, 6)

        return width, height, color_depth, has_alpha

    def _parse_jpeg_header(self, content: bytes) -> Tuple[int, int]:
        """Parse JPEG header for dimensions."""
        # JPEG starts with FF D8
        if content[:2] != b'\xff\xd8':
            raise ValueError("Invalid JPEG signature")

        # Find SOF0 marker (FF C0) or SOF2 (FF C2)
        i = 2
        while i < len(content) - 8:
            if content[i] != 0xff:
                i += 1
                continue

            marker = content[i + 1]

            # SOF0, SOF1, SOF2 markers
            if marker in (0xc0, 0xc1, 0xc2):
                height = struct.unpack('>H', content[i + 5:i + 7])[0]
                width = struct.unpack('>H', content[i + 7:i + 9])[0]
                return width, height

            # Skip marker
            if marker == 0xd9:  # EOI
                break
            if marker in (0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0x01):
                i += 2
            else:
                length = struct.unpack('>H', content[i + 2:i + 4])[0]
                i += 2 + length

        return 0, 0

    def _parse_gif_header(self, content: bytes) -> Tuple[int, int, bool]:
        """Parse GIF header for dimensions and animation."""
        # GIF signature: GIF87a or GIF89a
        if content[:3] != b'GIF':
            raise ValueError("Invalid GIF signature")

        width = struct.unpack('<H', content[6:8])[0]
        height = struct.unpack('<H', content[8:10])[0]

        # Check for animation (multiple image blocks)
        # This is a simple heuristic
        is_animated = content.count(b'\x00\x2c') > 1

        return width, height, is_animated

    def _parse_bmp_header(self, content: bytes) -> Tuple[int, int, int]:
        """Parse BMP header for dimensions."""
        # BMP signature: BM
        if content[:2] != b'BM':
            raise ValueError("Invalid BMP signature")

        # Width and height are at offset 18 and 22 (4 bytes each, little-endian)
        width = struct.unpack('<I', content[18:22])[0]
        height = abs(struct.unpack('<i', content[22:26])[0])  # Can be negative
        color_depth = struct.unpack('<H', content[28:30])[0]

        return width, height, color_depth

    def _extract_text(self, content: bytes) -> Optional[str]:
        """Extract text from image using OCR."""
        if not self.enable_ocr:
            return None

        # Try custom OCR engine
        if self.ocr_engine:
            try:
                if hasattr(self.ocr_engine, 'extract_text'):
                    result = self.ocr_engine.extract_text(content)
                    return str(result) if result else None
                elif callable(self.ocr_engine):
                    result = self.ocr_engine(content)
                    return str(result) if result else None
            except Exception:
                pass

        # Try pytesseract if available
        try:
            import pytesseract  # type: ignore[import-not-found]
            if self._has_pil:
                img = self._pil_image.open(io.BytesIO(content))
                return str(pytesseract.image_to_string(img))
        except ImportError:
            pass
        except Exception:
            pass

        return None

    def get_image_for_api(
        self,
        path: str,
        max_dimension: Optional[int] = None,
    ) -> ImageResult:
        """
        Get image data formatted for API calls (like Claude API).

        Returns a dict with:
        - type: "image"
        - source: { type: "base64", media_type: "...", data: "..." }
        """
        result = self.read_image(path)
        if not result.success:
            return result

        image_data: ImageData = result.data

        # Resize if needed and PIL is available
        base64_data = image_data.base64_data
        if max_dimension and self._has_pil:
            try:
                content = base64.b64decode(base64_data)
                img = self._pil_image.open(io.BytesIO(content))

                # Check if resize needed
                if img.width > max_dimension or img.height > max_dimension:
                    # Calculate new size maintaining aspect ratio
                    ratio = min(max_dimension / img.width, max_dimension / img.height)
                    new_size = (int(img.width * ratio), int(img.height * ratio))
                    # Use LANCZOS resampling (works in both old and new PIL)
                    resample = getattr(self._pil_image, 'LANCZOS', getattr(self._pil_image, 'Resampling', {}).get('LANCZOS', 1))
                    img = img.resize(new_size, resample)

                    # Re-encode
                    buffer = io.BytesIO()
                    img.save(buffer, format=image_data.metadata.format.value.upper())
                    base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception:
                pass

        api_format = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_data.metadata.mime_type,
                "data": base64_data,
            }
        }

        return ImageResult(success=True, data=api_format)

    def get_image_description(self, path: str) -> str:
        """Get a text description of an image for context."""
        result = self.read_image(path)
        if not result.success:
            return f"[Image: {path} - Error: {result.error}]"

        image_data: ImageData = result.data
        meta = image_data.metadata

        desc_parts = [
            f"[Image: {Path(path).name}",
            f"{meta.width}x{meta.height}",
            meta.format.value.upper(),
        ]

        if meta.has_alpha:
            desc_parts.append("with alpha")
        if meta.is_animated:
            desc_parts.append(f"animated ({meta.frame_count} frames)")

        desc_parts.append(f"{meta.file_size // 1024}KB]")

        return " ".join(desc_parts)

    def list_images(
        self,
        directory: str,
        recursive: bool = False,
    ) -> ImageResult:
        """List all image files in a directory."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            return ImageResult(success=False, error=f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        images: List[Dict[str, Any]] = []

        for path in dir_path.glob(pattern):
            if path.is_file() and self.is_image_file(str(path)):
                try:
                    file_size = path.stat().st_size
                    images.append({
                        "path": str(path),
                        "name": path.name,
                        "format": self.get_format(str(path)).value,
                        "size": file_size,
                    })
                except Exception:
                    continue

        return ImageResult(success=True, data=images)


# Singleton instance
_image_tools: Optional[ImageTools] = None


def get_image_tools(
    max_file_size: int = ImageTools.MAX_FILE_SIZE,
    enable_ocr: bool = False,
) -> ImageTools:
    """Get or create the global image tools."""
    global _image_tools
    if _image_tools is None:
        _image_tools = ImageTools(
            max_file_size=max_file_size,
            enable_ocr=enable_ocr,
        )
    return _image_tools


def reset_image_tools() -> None:
    """Reset the global image tools."""
    global _image_tools
    _image_tools = None
