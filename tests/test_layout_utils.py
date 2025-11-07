import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import resume_parser.layout_utils as layout_utils
from resume_parser.ingestion import remove_headers_footers
from resume_parser.types import BoundingBox, DocumentContent, PageContent, PageMetadata, Token


def _make_page(tokens, number=1):
    metadata = PageMetadata(width=1000, height=1000, number=number)
    return PageContent(metadata=metadata, tokens=tokens)


def _make_token(text, x0, y0, x1, y1, page):
    return Token(text=text, bbox=BoundingBox(x0, y0, x1, y1), page=page, metadata={})


def build_document() -> DocumentContent:
    left_tokens = [
        _make_token("Left", 50, 100, 200, 150, 0),
        _make_token("Column", 60, 160, 220, 210, 0),
    ]
    right_tokens = [
        _make_token("Right", 700, 120, 850, 170, 0),
    ]
    header_tokens = [_make_token("Header", 100, 20, 200, 40, 0)]
    page1 = _make_page(header_tokens + left_tokens + right_tokens, number=1)

    body_tokens = [
        _make_token("Body", 80, 400, 220, 450, 1),
        _make_token("Footer", 90, 960, 200, 990, 1),
    ]
    page2 = _make_page(body_tokens, number=2)
    raw_text = "Header\nLeft Column\nRight\nBody\nFooter"
    return DocumentContent(pages=[page1, page2], raw_text=raw_text, file_path="dummy.pdf")


def test_assign_columns_and_sorting():
    document = build_document()
    layout_utils.assign_columns(document)
    layout_utils.reorder_document_tokens(document)

    column_ids = {token.text: token.metadata.get("column_id") for token in document.tokens}
    assert column_ids["Left"] == column_ids["Column"]
    assert column_ids["Right"] != column_ids["Left"]

    ordered = layout_utils.sort_tokens_reading_order(document)
    texts = [token.text for token in ordered]
    assert texts.index("Left") < texts.index("Right")


def test_remove_headers_and_footers():
    document = build_document()
    remove_headers_footers(document, region_height=100, min_repeats=1)

    remaining = [token.text for token in document.tokens]
    assert "Header" not in remaining
    assert document.raw_text and "Header" not in document.raw_text
