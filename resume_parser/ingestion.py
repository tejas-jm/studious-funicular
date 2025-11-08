"""Document ingestion utilities for the resume parsing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pdfplumber
from PIL import Image

try:
    from pdf2image import convert_from_path  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    convert_from_path = None

try:  # pragma: no cover - optional dependency
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None

try:  # pragma: no cover - optional dependency
    import docx  # python-docx
except Exception:
    docx = None

try:  # pragma: no cover - optional dependency
    import docx2txt  # type: ignore
except Exception:
    docx2txt = None

from .types import BoundingBox, DocumentContent, PageContent, PageMetadata, Token

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}


@dataclass
class IngestionConfig:
    """Configuration options for ingestion."""

    max_pages: Optional[int] = None
    bbox_scale: int = 1000
    ocr_language: str = "en"
    keep_images: bool = False


def detect_file_type(file_path: str) -> str:
    """Return the canonical file extension for *file_path*.

    Raises:
        ValueError: If the file extension is not supported.
    """

    extension = Path(file_path).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")
    return extension


def normalize_bbox(
    bbox: Tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    scale: int = 1000,
) -> BoundingBox:
    """Convert a bounding box into a normalized coordinate system."""

    x0, top, x1, bottom = bbox
    # Ensure values are within bounds and convert to LayoutLM scale (0-1000)
    x0 = max(min(x0, page_width), 0)
    x1 = max(min(x1, page_width), 0)
    top = max(min(top, page_height), 0)
    bottom = max(min(bottom, page_height), 0)

    return BoundingBox(
        x0=int(scale * x0 / page_width) if page_width else 0,
        y0=int(scale * top / page_height) if page_height else 0,
        x1=int(scale * x1 / page_width) if page_width else 0,
        y1=int(scale * bottom / page_height) if page_height else 0,
    )


def _post_process_tokens(tokens: Iterable[Token]) -> List[Token]:
    cleaned_tokens: List[Token] = []
    for token in tokens:
        text = token.text.strip()
        if not text:
            continue
        # unify whitespace
        text = " ".join(text.split())
        token.text = text
        cleaned_tokens.append(token)
    return cleaned_tokens


def extract_docx_content(file_path: str, config: IngestionConfig) -> DocumentContent:
    """Extract tokens from a DOCX/DOC file using python-docx.

    DOC files are converted using *docx2txt* when possible.
    Bounding boxes are heuristically generated because DOCX does not
    expose absolute positioning. The pipeline later refines these boxes.
    """

    if docx is None:
        raise ImportError("python-docx is required to parse DOCX files")

    extension = Path(file_path).suffix.lower()
    if extension == ".doc" and docx2txt is not None:
        LOGGER.info("Converting legacy .doc file via docx2txt")
        text = docx2txt.process(file_path)
        paragraphs = [line for line in text.splitlines() if line.strip()]
    else:
        document = docx.Document(file_path)  # type: ignore
        paragraphs = [para.text for para in document.paragraphs if para.text.strip()]

    tokens: List[Token] = []
    raw_text_lines: List[str] = []

    page_width, page_height = 8.5 * 72, 11 * 72  # assume letter size in points
    line_height = page_height / max(len(paragraphs), 1)
    for line_idx, line in enumerate(paragraphs):
        raw_text_lines.append(line)
        words = line.split()
        for word_idx, word in enumerate(words):
            left = (word_idx / max(len(words), 1)) * page_width
            right = ((word_idx + 1) / max(len(words), 1)) * page_width
            top = line_idx * line_height
            bottom = top + line_height
            bbox = normalize_bbox((left, top, right, bottom), page_width, page_height, config.bbox_scale)
            tokens.append(Token(text=word, bbox=bbox, page=0, metadata={"line": line_idx}))

    page_metadata = PageMetadata(width=int(page_width), height=int(page_height), number=1)
    page_content = PageContent(metadata=page_metadata, tokens=_post_process_tokens(tokens))

    return DocumentContent(pages=[page_content], raw_text="\n".join(raw_text_lines), file_path=file_path)


def _perform_easyocr(page_image: Image.Image, config: IngestionConfig, page_number: int) -> List[Token]:
    if easyocr is None:
        raise ImportError("easyocr is required for OCR on scanned PDFs")

    reader = easyocr.Reader([config.ocr_language], gpu=False)  # heavy operation; cache outside in prod
    width, height = page_image.size
    ocr_results = reader.readtext(page_image, detail=1, paragraph=False)
    tokens: List[Token] = []
    for bbox, text, confidence in ocr_results:
        # easyocr bounding boxes are list of four points; convert to bounding rectangle
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        normalized = normalize_bbox((x0, y0, x1, y1), width, height, config.bbox_scale)
        tokens.append(
            Token(
                text=text,
                bbox=normalized,
                page=page_number,
                metadata={"confidence": confidence},
            )
        )
    return tokens


def extract_pdf_content(file_path: str, config: IngestionConfig) -> DocumentContent:
    """Extract tokens from a PDF file using pdfplumber.

    Falls back to OCR for scanned PDFs when no text is detected.
    """

    pages: List[PageContent] = []
    raw_text_lines: List[str] = []

    with pdfplumber.open(file_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            if config.max_pages and page_number > config.max_pages:
                break

            words = page.extract_words(x_tolerance=1, y_tolerance=1)
            page_width = float(page.width or 1)
            page_height = float(page.height or 1)
            page_tokens: List[Token]
            image_meta_path: Optional[str] = None

            if words:
                page_tokens = []
                for word in words:
                    text = word.get("text", "").strip()
                    if not text:
                        continue
                    bbox = normalize_bbox(
                        (
                            float(word["x0"]),
                            float(word["top"]),
                            float(word["x1"]),
                            float(word["bottom"]),
                        ),
                        page_width,
                        page_height,
                        config.bbox_scale,
                    )
                    page_tokens.append(
                        Token(
                            text=text,
                            bbox=bbox,
                            page=page_number - 1,
                            metadata={"upright": word.get("upright", True)},
                        )
                    )
                    raw_text_lines.append(text)
            else:
                LOGGER.info("No selectable text on page %s; running OCR", page_number)
                if convert_from_path is None:
                    raise RuntimeError("pdf2image is required to OCR scanned PDFs")
                images = convert_from_path(file_path, first_page=page_number, last_page=page_number)
                page_tokens = []
                for image in images:
                    tokens = _perform_easyocr(image, config, page_number - 1)
                    page_tokens.extend(tokens)
                    if config.keep_images:
                        image_path = Path(file_path).with_suffix(f".page{page_number}.png")
                        image.save(image_path)
                        image.close()
                        image_meta_path = str(image_path)

            page_content = PageContent(
                metadata=PageMetadata(
                    width=int(page_width),
                    height=int(page_height),
                    number=page_number,
                    image_path=image_meta_path,
                ),
                tokens=_post_process_tokens(page_tokens),
            )
            pages.append(page_content)

    return DocumentContent(pages=pages, raw_text="\n".join(raw_text_lines), file_path=file_path)


def ingest_document(file_path: str, config: Optional[IngestionConfig] = None) -> DocumentContent:
    """Read *file_path* and return the normalized document content."""

    if config is None:
        config = IngestionConfig()

    file_type = detect_file_type(file_path)
    if file_type in {".doc", ".docx"}:
        return extract_docx_content(file_path, config)
    if file_type == ".pdf":
        return extract_pdf_content(file_path, config)

    raise ValueError(f"Unsupported file type: {file_type}")
