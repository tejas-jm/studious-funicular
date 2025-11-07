"""Layout heuristics for multi-column resumes."""

from __future__ import annotations

import logging
from typing import List

from .types import DocumentContent, Token

LOGGER = logging.getLogger(__name__)


def _token_center_x(token: Token) -> float:
    return (token.bbox.x0 + token.bbox.x1) / 2.0


def assign_columns(document: DocumentContent, max_columns: int = 3, column_gap: int = 120) -> DocumentContent:
    """Assign a column identifier to each token based on x-position clustering."""

    for page in document.pages:
        tokens = page.tokens
        if not tokens:
            continue
        columns: List[dict[str, float]] = []
        for token in sorted(tokens, key=_token_center_x):
            center = _token_center_x(token)
            assigned = False
            for idx, column in enumerate(columns):
                if abs(center - column["center"]) <= column_gap:
                    count = column["count"] + 1
                    column["center"] = (column["center"] * column["count"] + center) / count
                    column["count"] = count
                    token.metadata["column_id"] = idx
                    assigned = True
                    break
            if assigned:
                continue
            if len(columns) < max_columns:
                columns.append({"center": center, "count": 1})
                token.metadata["column_id"] = len(columns) - 1
            else:
                nearest_idx = min(range(len(columns)), key=lambda i: abs(center - columns[i]["center"]))
                column = columns[nearest_idx]
                count = column["count"] + 1
                column["center"] = (column["center"] * column["count"] + center) / count
                column["count"] = count
                token.metadata["column_id"] = nearest_idx
        LOGGER.debug(
            "Assigned %s columns on page %s", len(columns), page.metadata.number
        )
    return document


def sort_tokens_reading_order(document: DocumentContent) -> List[Token]:
    """Return tokens sorted by reading order (page → column → y → x)."""

    sorted_tokens = sorted(
        document.tokens,
        key=lambda token: (
            token.page,
            token.metadata.get("column_id", 0),
            token.bbox.y0,
            token.bbox.x0,
        ),
    )
    return sorted_tokens


def reorder_document_tokens(document: DocumentContent) -> DocumentContent:
    """Sort tokens in-place on each page to follow reading order."""

    for page in document.pages:
        page.tokens = sorted(
            page.tokens,
            key=lambda token: (
                token.metadata.get("column_id", 0),
                token.bbox.y0,
                token.bbox.x0,
            ),
        )
    return document


__all__ = ["assign_columns", "sort_tokens_reading_order", "reorder_document_tokens"]
