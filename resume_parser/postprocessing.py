"""Post-processing utilities for the resume parsing pipeline."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from .types import DocumentContent, ParsedResume, Token, TokenEmbedding

LOGGER = logging.getLogger(__name__)

DATE_PATTERNS = [
    re.compile(r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(?P<year>\d{4})", re.I),
    re.compile(r"(?P<year>\d{4})(?:[-/](?P<month>\d{1,2}))?"),
]
DATE_RANGE_PATTERN = re.compile(r"(?P<start>[^-–]+)[-–](?P<end>.+)")
SECTION_KEYWORDS = {
    "education": ["education", "academic", "university", "college"],
    "work_experience": ["experience", "employment", "career", "work history"],
    "skills": ["skills", "technologies", "technical", "tools"],
    "certifications": ["certification", "licenses", "license"],
    "projects": ["project", "portfolio"],
    "publications": ["publication", "papers", "articles"],
    "languages": ["languages", "language"],
}
DEFAULT_SCHEMA = {
    "contact": {},
    "education": [],
    "work_experience": [],
    "skills": [],
    "certifications": [],
    "projects": [],
    "publications": [],
    "languages": [],
    "other_sections": [],
}


def normalize_date(text: str) -> Optional[str]:
    text = text.strip()
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        year = match.group("year")
        month = match.groupdict().get("month")
        if month:
            try:
                month_number = datetime.strptime(month[:3], "%b").month
                return f"{year}-{month_number:02d}"
            except ValueError:
                continue
        return year
    if any(keyword in text.lower() for keyword in ["present", "current"]):
        return "present"
    return None


def normalize_date_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    match = DATE_RANGE_PATTERN.search(text)
    if match:
        start = normalize_date(match.group("start"))
        end = normalize_date(match.group("end"))
        return start, end
    normalized = normalize_date(text)
    return normalized, None


def compute_duration(start: Optional[str], end: Optional[str]) -> Optional[int]:
    if not start or start == "present":
        return None
    if end in (None, "present"):
        end_date = datetime.utcnow()
    else:
        if len(end) == 4:
            end += "-01"
        end_date = datetime.strptime(end, "%Y-%m")
    if len(start) == 4:
        start += "-01"
    start_date = datetime.strptime(start, "%Y-%m")
    months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    return max(months, 0)


def _join_tokens(tokens: Iterable[Token]) -> str:
    return " ".join(token.text for token in tokens)


def extract_contact(tokens: List[Token]) -> Dict[str, str]:
    text = _join_tokens(tokens)
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"(\+?\d[\d\s().-]{7,}\d)", text)
    url_match = re.search(r"https?://\S+", text)
    return {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "website": url_match.group(0) if url_match else "",
        "raw": text,
    }


def group_tokens_by_line(tokens: Iterable[Token]) -> Dict[int, List[Token]]:
    lines: Dict[int, List[Token]] = defaultdict(list)
    for token in tokens:
        line_index = token.metadata.get("line") if isinstance(token.metadata, dict) else None
        if line_index is None:
            # approximate line index from bbox
            line_index = token.bbox.y0 // 10
        lines[int(line_index)].append(token)
    return dict(sorted(lines.items()))


def detect_sections(document: DocumentContent) -> Dict[str, List[Token]]:
    sections: Dict[str, List[Token]] = defaultdict(list)
    current_section = "other_sections"
    keywords = {section: [kw.lower() for kw in words] for section, words in SECTION_KEYWORDS.items()}
    for token in document.tokens:
        word_lower = token.text.lower().strip(":")
        matched_section = None
        for section, section_keywords in keywords.items():
            if word_lower in section_keywords:
                matched_section = section
                break
        if matched_section:
            current_section = matched_section
            continue
        sections[current_section].append(token)
    return sections


def build_work_entries(tokens: List[Token]) -> List[Dict[str, object]]:
    lines = group_tokens_by_line(tokens)
    entries: List[Dict[str, object]] = []
    current_entry: Dict[str, object] = {}
    for _, line_tokens in lines.items():
        line_text = _join_tokens(line_tokens)
        start, end = normalize_date_range(line_text)
        if start or end:
            if current_entry:
                entries.append(current_entry)
            current_entry = {
                "organization": line_text,
                "start_date": start,
                "end_date": end,
                "duration_months": compute_duration(start, end) if start else None,
                "description": [],
            }
        else:
            current_entry.setdefault("description", []).append(line_text)
    if current_entry:
        entries.append(current_entry)
    return entries


def build_education_entries(tokens: List[Token]) -> List[Dict[str, object]]:
    text = _join_tokens(tokens)
    entries: List[Dict[str, object]] = []
    for segment in re.split(r"\b(?:Degree|Diploma|Certificate)\b", text, flags=re.I):
        segment = segment.strip()
        if not segment:
            continue
        start, end = normalize_date_range(segment)
        entries.append(
            {
                "institution": segment,
                "degree": "",
                "field_of_study": "",
                "start_date": start,
                "end_date": end,
                "grade": "",
            }
        )
    return entries


def deduplicate_skills(tokens: List[Token]) -> List[str]:
    words = [_join_tokens(tokens).replace(";", ",")]
    skill_candidates = []
    for chunk in words:
        skill_candidates.extend([skill.strip() for skill in chunk.split(",") if skill.strip()])
    normalized = set()
    result: List[str] = []
    synonyms = {"js": "javascript"}
    for skill in skill_candidates:
        key = skill.lower()
        key = synonyms.get(key, key)
        if key in normalized:
            continue
        normalized.add(key)
        result.append(skill)
    return result


def build_resume(tokens: Dict[str, List[Token]]) -> ParsedResume:
    resume: ParsedResume = {section: value for section, value in DEFAULT_SCHEMA.items()}
    resume = {k: (v.copy() if isinstance(v, list) else dict(v)) for k, v in resume.items()}

    resume["contact"] = extract_contact(tokens.get("contact", [])) if tokens.get("contact") else {}
    resume["education"] = build_education_entries(tokens.get("education", []))
    resume["work_experience"] = build_work_entries(tokens.get("work_experience", []))
    resume["skills"] = deduplicate_skills(tokens.get("skills", []))
    resume["certifications"] = [{"name": _join_tokens(tokens.get("certifications", []))}] if tokens.get("certifications") else []
    resume["projects"] = [{"name": _join_tokens(tokens.get("projects", []))}] if tokens.get("projects") else []
    resume["publications"] = [{"title": _join_tokens(tokens.get("publications", []))}] if tokens.get("publications") else []
    resume["languages"] = deduplicate_skills(tokens.get("languages", []))
    resume["other_sections"] = [
        {
            "label": "other",
            "content": _join_tokens(tokens.get("other_sections", [])),
        }
    ] if tokens.get("other_sections") else []
    return resume


def link_entities(document: DocumentContent, embeddings: Optional[List[TokenEmbedding]] = None) -> ParsedResume:
    sections = detect_sections(document)
    if not sections.get("contact") and document.pages:
        # assume first 5 lines as contact fallback
        first_page_tokens = document.pages[0].tokens[:50]
        sections["contact"] = first_page_tokens
    resume = build_resume(sections)
    if embeddings:
        resume.setdefault("meta", {})["token_embeddings"] = len(embeddings)
    return resume
