"""Common data structures used across the resume parsing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class BoundingBox:
    """Bounding box in a 0-1000 normalized coordinate space."""

    x0: int
    y0: int
    x1: int
    y1: int

    def as_tuple(self) -> Sequence[int]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Token:
    """Represents a single token/word extracted from the document."""

    text: str
    bbox: BoundingBox
    page: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PageMetadata:
    """General metadata per page (size, rotation, etc.)."""

    width: int
    height: int
    number: int
    rotation: int = 0
    image_path: Optional[str] = None


@dataclass
class PageContent:
    """Collection of tokens and metadata for a single page."""

    metadata: PageMetadata
    tokens: List[Token]


@dataclass
class DocumentContent:
    """Structured representation of the raw document ready for inference."""

    pages: List[PageContent]
    raw_text: str
    file_path: str

    @property
    def tokens(self) -> List[Token]:
        tokens: List[Token] = []
        for page in self.pages:
            tokens.extend(page.tokens)
        return tokens


@dataclass
class ParsedSection:
    """Represents a parsed section with optional confidence."""

    label: str
    content: Dict[str, Any]
    confidence: Optional[float] = None


ParsedResume = Dict[str, Any]


@dataclass
class TokenEmbedding:
    """Container linking a token to its contextual embedding."""

    token: Token
    embedding: List[float]
    logits: Optional[List[float]] = None
