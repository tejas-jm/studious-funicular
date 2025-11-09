"""Post-processing utilities for the resume parsing pipeline."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from .schema import (
    Certification,
    Contact,
    Education,
    Language,
    Meta,
    OtherSection,
    Project,
    Publication,
    ResumeOutput,
    Skill,
    WorkExperience,
)
from .types import DocumentContent, Token, TokenEmbedding

LOGGER = logging.getLogger(__name__)

MONTH_NAMES = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
)
DATE_PATTERNS = [
    re.compile(r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(?P<year>\d{4})", re.I),
    re.compile(r"(?P<year>\d{4})(?:[-/](?P<month>\d{1,2}))?"),
]
DATE_RANGE_PATTERN = re.compile(r"(?P<start>[^-–]+)[-–](?P<end>.+)")
DEGREE_PATTERN = re.compile(
    r"(Bachelor(?:'s)?|Master(?:'s)?|B\.\s?Sc|M\.\s?Sc|B\.\s?Eng|M\.\s?Eng|MBA|Ph\.?D)",
    re.I,
)
FIELD_MARKER_PATTERN = re.compile(r"(?:in|of)\s+([A-Za-z&\s]+)", re.I)
BULLET_PATTERN = re.compile(r"^[•\-\u2022\u2023\u25E6\*]+\s*")
SECTION_KEYWORDS = {
    "education": ["education", "academic", "university", "college"],
    "work_experience": ["experience", "employment", "career", "work history"],
    "skills": ["skills", "technologies", "technical", "tools"],
    "certifications": ["certification", "certifications", "licenses", "license"],
    "projects": ["project", "projects", "portfolio"],
    "publications": ["publication", "publications", "papers", "articles"],
    "languages": ["languages", "language"],
}


def normalize_date(text: str) -> Optional[str]:
    """Normalize a date string to ISO (YYYY or YYYY-MM) if possible."""

    text = text.strip()
    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        year = match.group("year")
        month = match.groupdict().get("month")
        if month:
            try:
                if len(month) == 2 and month.isdigit():
                    month_number = int(month)
                else:
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
    return " ".join(token.text for token in tokens).strip()


def group_tokens_by_line(tokens: Iterable[Token]) -> Dict[int, List[Token]]:
    lines: Dict[int, List[Token]] = defaultdict(list)
    for token in tokens:
        line_index = None
        if isinstance(token.metadata, dict):
            line_index = token.metadata.get("line")
        if line_index is None:
            line_index = token.bbox.y0 // 10
        lines[int(line_index)].append(token)
    return dict(sorted(lines.items()))


def tokens_to_lines(tokens: Iterable[Token]) -> List[str]:
    grouped = group_tokens_by_line(tokens)
    lines: List[str] = []
    for _, line_tokens in grouped.items():
        line_text = _join_tokens(line_tokens)
        if line_text:
            lines.append(line_text)
    return lines


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


def _extract_name(lines: List[str]) -> Optional[str]:
    for line in lines[:3]:
        if not line:
            continue
        candidate = line.strip()
        if any(symbol in candidate for symbol in ("@", "+", "http")):
            continue
        words = candidate.split()
        if 1 < len(words) <= 5:
            return candidate
    return None


def extract_contact(document: DocumentContent, tokens: List[Token]) -> Contact:
    lines = tokens_to_lines(tokens)
    if not lines and document.pages:
        first_page_tokens = document.pages[0].tokens[:50]
        lines = tokens_to_lines(first_page_tokens)
    contact_text = "\n".join(lines) if lines else None
    combined_text = " ".join(lines) if lines else document.raw_text
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", combined_text)
    phone_match = re.search(r"(\+?\d[\d\s().-]{7,}\d)", combined_text)
    url_match = re.search(r"https?://\S+", combined_text)
    location = None
    for line in reversed(lines):
        if re.search(r"\d+\s+[A-Za-z]", line):
            location = line.strip()
            break
    name = _extract_name(lines)
    return Contact(
        name=name,
        email=email_match.group(0) if email_match else None,
        phone=phone_match.group(0) if phone_match else None,
        website=url_match.group(0) if url_match else None,
        location=location,
        raw=contact_text,
    )


def _split_role_company(text: str) -> Tuple[Optional[str], Optional[str]]:
    lowered = text.lower()
    if " at " in lowered:
        idx = lowered.index(" at ")
        position = text[:idx].strip(" ,-|•")
        company = text[idx + 4 :].strip(" ,-|•")
        return company or None, position or None
    parts = [part.strip(" ,-|•") for part in re.split(r"[,|]", text) if part.strip()]
    if len(parts) >= 2:
        return parts[1], parts[0]
    return text.strip(" ,-|•") or None, None


def _clean_bullet(text: str) -> str:
    return BULLET_PATTERN.sub("", text).strip()


def build_work_entries(tokens: List[Token]) -> List[WorkExperience]:
    lines = tokens_to_lines(tokens)
    entries: List[WorkExperience] = []
    current: Optional[WorkExperience] = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        start, end = normalize_date_range(line)
        has_dates = bool(start or end)
        if has_dates:
            heading = line
            date_match = DATE_RANGE_PATTERN.search(line)
            if date_match:
                heading = heading[: date_match.start()].strip(" ,-•")
            else:
                for month in MONTH_NAMES:
                    month_idx = heading.lower().rfind(month.lower())
                    if month_idx != -1:
                        heading = heading[:month_idx].strip(" ,-•")
                        break
            company, position = _split_role_company(heading)
            if current:
                entries.append(current)
            current = WorkExperience(
                company=company,
                position=position,
                start_date=start,
                end_date=end,
                duration_months=compute_duration(start, end),
                description=[],
            )
            continue
        if current is None:
            company, position = _split_role_company(line)
            current = WorkExperience(
                company=company,
                position=position,
                description=[],
            )
            continue
        for bullet in re.split(r"[\n]", line):
            cleaned = _clean_bullet(bullet)
            if cleaned:
                current.description.append(cleaned)
    if current:
        entries.append(current)
    return entries


def _parse_education_chunk(chunk: str) -> Education:
    start, end = normalize_date_range(chunk)
    text = chunk
    match = DATE_RANGE_PATTERN.search(text)
    if match:
        text = (text[: match.start()] + text[match.end() :]).strip(" ,-•")
    degree = None
    degree_match = DEGREE_PATTERN.search(text)
    if degree_match:
        degree = degree_match.group(0).strip()
    field_of_study = None
    field_match = FIELD_MARKER_PATTERN.search(text)
    if field_match:
        field_of_study = field_match.group(1).strip()
    institution = text
    if degree_match:
        institution = text[: degree_match.start()].strip(" ,-•") or institution
    parts = [part.strip() for part in re.split(r"[,|]", institution) if part.strip()]
    if parts:
        institution = parts[0]
    grade = None
    grade_match = re.search(r"(GPA|Grade)[:\s]+([0-9.]+)", chunk, re.I)
    if grade_match:
        grade = grade_match.group(2)
    return Education(
        institution=institution or None,
        degree=degree,
        field_of_study=field_of_study,
        start_date=start,
        end_date=end,
        grade=grade,
    )


def build_education_entries(tokens: List[Token]) -> List[Education]:
    text = _join_tokens(tokens)
    if not text:
        return []
    segments = [segment.strip() for segment in re.split(r"\n{2,}|•", text) if segment.strip()]
    if not segments:
        segments = [text]
    entries = [_parse_education_chunk(segment) for segment in segments]
    return entries


def extract_skills(tokens: List[Token]) -> List[Skill]:
    text = _join_tokens(tokens)
    if not text:
        return []
    candidates = re.split(r"[,;/\n]|•", text)
    skills: List[Skill] = []
    seen = set()
    for candidate in candidates:
        skill = _clean_bullet(candidate).strip()
        if not skill or len(skill) > 60:
            continue
        key = skill.lower()
        if key in seen:
            continue
        seen.add(key)
        skills.append(Skill(name=skill))
    return skills


def build_certifications(tokens: List[Token]) -> List[Certification]:
    lines = tokens_to_lines(tokens)
    certifications: List[Certification] = []
    for line in lines:
        if not line:
            continue
        name = _clean_bullet(line)
        start, end = normalize_date_range(name)
        date = end or start
        if date:
            name = DATE_RANGE_PATTERN.sub("", name).strip(" ,-•")
        certifications.append(Certification(name=name or None, date=date))
    return certifications


def build_projects(tokens: List[Token]) -> List[Project]:
    text = _join_tokens(tokens)
    if not text:
        return []
    segments = [segment.strip() for segment in re.split(r"\n{2,}|•", text) if segment.strip()]
    projects: List[Project] = []
    for segment in segments:
        name = segment.split(" - ")[0].strip()
        remainder = segment[len(name) :].strip(" -") if len(segment) > len(name) else ""
        role = None
        description = remainder or None
        technologies = [
            tech.strip()
            for tech in re.split(r"[,/|]", remainder)
            if tech.strip() and len(tech.strip()) <= 40
        ]
        projects.append(
            Project(
                name=name or None,
                role=role,
                description=description,
                technologies=technologies,
            )
        )
    return projects


def build_publications(tokens: List[Token]) -> List[Publication]:
    lines = tokens_to_lines(tokens)
    publications: List[Publication] = []
    for line in lines:
        cleaned = _clean_bullet(line)
        if not cleaned:
            continue
        date_start, date_end = normalize_date_range(cleaned)
        date = date_end or date_start
        title = cleaned
        if date:
            title = DATE_RANGE_PATTERN.sub("", title).strip(" ,-•")
        publications.append(Publication(title=title or None, date=date))
    return publications


def build_languages(tokens: List[Token]) -> List[Language]:
    text = _join_tokens(tokens)
    if not text:
        return []
    entries = [entry.strip() for entry in re.split(r"[,/\n]|•", text) if entry.strip()]
    languages: List[Language] = []
    for entry in entries:
        if "-" in entry:
            language, fluency = [part.strip() for part in entry.split("-", 1)]
        elif ":" in entry:
            language, fluency = [part.strip() for part in entry.split(":", 1)]
        else:
            language, fluency = entry.strip(), None
        languages.append(Language(name=language or None, proficiency=fluency or None))
    return languages


def build_other_sections(tokens: List[Token]) -> List[OtherSection]:
    if not tokens:
        return []
    text = _join_tokens(tokens)
    if not text:
        return []
    return [OtherSection(label="other", content=text)]


def link_entities(
    document: DocumentContent,
    embeddings: Optional[List[TokenEmbedding]] = None,
) -> ResumeOutput:
    """Convert tokens and embeddings into a structured resume output."""

    sections = detect_sections(document)
    if not sections.get("contact") and document.pages:
        first_page_tokens = document.pages[0].tokens[:50]
        sections["contact"] = first_page_tokens

    contact = extract_contact(document, sections.get("contact", []))
    education = build_education_entries(sections.get("education", []))
    work_experience = build_work_entries(sections.get("work_experience", []))
    skills = extract_skills(sections.get("skills", []))
    certifications = build_certifications(sections.get("certifications", []))
    projects = build_projects(sections.get("projects", []))
    publications = build_publications(sections.get("publications", []))
    languages = build_languages(sections.get("languages", []))
    other_sections = build_other_sections(sections.get("other_sections", []))

    try:
        resume = ResumeOutput(
            contact=contact,
            education=education,
            work_experience=work_experience,
            skills=skills,
            certifications=certifications,
            projects=projects,
            publications=publications,
            languages=languages,
            other_sections=other_sections,
            meta=Meta(source=document.file_path),
        )
        resume.validate()
    except (TypeError, ValueError) as error:
        LOGGER.error("Resume output validation failed: %s", error)
        raise

    if embeddings:
        meta_payload = resume.meta.to_dict()
        existing_notes = meta_payload.get("notes")
        note = f"token_embeddings={len(embeddings)}"
        if existing_notes:
            note = f"{existing_notes}; {note}"
        meta_payload["notes"] = note
        resume.meta = Meta(**meta_payload)

    return resume
