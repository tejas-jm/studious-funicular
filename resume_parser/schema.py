"""Output schema definitions for the resume parsing pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _normalize_present(value: str) -> str:
    if value.lower() == "present":
        return "Present"
    return value


def _validate_date(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() == "present":
        return "Present"
    if len(normalized) == 4:
        datetime.strptime(normalized, "%Y")
        return normalized
    datetime.strptime(normalized, "%Y-%m")
    return normalized


def _ensure_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass
class Contact:
    """Basic contact details extracted from a resume."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    location: Optional[str] = None
    raw: Optional[str] = None


@dataclass
class Education:
    """Education entry with institution and degree information."""

    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    grade: Optional[str] = None
    location: Optional[str] = None

    def __post_init__(self) -> None:
        if self.start_date:
            self.start_date = _validate_date(self.start_date)
        if self.end_date:
            if isinstance(self.end_date, str):
                self.end_date = _normalize_present(self.end_date)
            self.end_date = _validate_date(self.end_date)


@dataclass
class WorkExperience:
    """Professional experience item."""

    company: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    location: Optional[str] = None
    description: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.start_date:
            self.start_date = _validate_date(self.start_date)
        if self.end_date:
            if isinstance(self.end_date, str):
                self.end_date = _normalize_present(self.end_date)
            self.end_date = _validate_date(self.end_date)
        self.description = [item.strip() for item in self.description if item and item.strip()]


@dataclass
class Skill:
    """Skill entry with optional categorisation and proficiency."""

    name: Optional[str] = None
    category: Optional[str] = None
    proficiency: Optional[str] = None


@dataclass
class Certification:
    """Certification or license entry."""

    name: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None

    def __post_init__(self) -> None:
        if self.date:
            try:
                self.date = _validate_date(self.date)
            except ValueError:
                self.date = self.date


@dataclass
class Project:
    """Project entry including role, dates, and technologies."""

    name: Optional[str] = None
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.start_date:
            self.start_date = _validate_date(self.start_date)
        if self.end_date:
            if isinstance(self.end_date, str):
                self.end_date = _normalize_present(self.end_date)
            self.end_date = _validate_date(self.end_date)
        self.technologies = [tech.strip() for tech in self.technologies if tech and tech.strip()]


@dataclass
class Publication:
    """Publication entry for articles, papers, etc."""

    title: Optional[str] = None
    venue: Optional[str] = None
    date: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if self.date:
            try:
                self.date = _validate_date(self.date)
            except ValueError:
                self.date = self.date


@dataclass
class Language:
    """Language proficiency entry."""

    name: Optional[str] = None
    proficiency: Optional[str] = None


@dataclass
class OtherSection:
    """Catch-all section for unparsed or custom resume segments."""

    label: Optional[str] = None
    content: Optional[str] = None


@dataclass
class Meta:
    """Metadata produced during parsing and refinement."""

    source: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class ResumeOutput:
    """Top-level structured resume representation."""

    contact: Contact = field(default_factory=Contact)
    education: List[Education] = field(default_factory=list)
    work_experience: List[WorkExperience] = field(default_factory=list)
    skills: List[Skill] = field(default_factory=list)
    certifications: List[Certification] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    publications: List[Publication] = field(default_factory=list)
    languages: List[Language] = field(default_factory=list)
    other_sections: List[OtherSection] = field(default_factory=list)
    meta: Meta = field(default_factory=Meta)

    def __post_init__(self) -> None:
        self.education = [item if isinstance(item, Education) else Education(**item) for item in self.education]
        self.work_experience = [item if isinstance(item, WorkExperience) else WorkExperience(**item) for item in self.work_experience]
        self.skills = [item if isinstance(item, Skill) else Skill(**item) for item in self.skills]
        self.certifications = [item if isinstance(item, Certification) else Certification(**item) for item in self.certifications]
        self.projects = [item if isinstance(item, Project) else Project(**item) for item in self.projects]
        self.publications = [item if isinstance(item, Publication) else Publication(**item) for item in self.publications]
        self.languages = [item if isinstance(item, Language) else Language(**item) for item in self.languages]
        self.other_sections = [item if isinstance(item, OtherSection) else OtherSection(**item) for item in self.other_sections]
        if not isinstance(self.meta, Meta):
            self.meta = Meta(**(self.meta or {}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact": asdict(self.contact),
            "education": [asdict(item) for item in self.education],
            "work_experience": [asdict(item) for item in self.work_experience],
            "skills": [asdict(item) for item in self.skills],
            "certifications": [asdict(item) for item in self.certifications],
            "projects": [asdict(item) for item in self.projects],
            "publications": [asdict(item) for item in self.publications],
            "languages": [asdict(item) for item in self.languages],
            "other_sections": [asdict(item) for item in self.other_sections],
            "meta": self.meta.to_dict(),
        }

    def json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ResumeOutput":
        payload = dict(payload)
        return cls(
            contact=Contact(**payload.get("contact", {})),
            education=payload.get("education", []),
            work_experience=payload.get("work_experience", []),
            skills=payload.get("skills", []),
            certifications=payload.get("certifications", []),
            projects=payload.get("projects", []),
            publications=payload.get("publications", []),
            languages=payload.get("languages", []),
            other_sections=payload.get("other_sections", []),
            meta=payload.get("meta", {}),
        )

    def validate(self) -> None:
        for item in self.education:
            _ = item.start_date
            _ = item.end_date
        for item in self.work_experience:
            _ = item.start_date
            _ = item.end_date
        if not isinstance(self.skills, list):
            raise TypeError("skills must be a list")


__all__ = [
    "Contact",
    "Education",
    "WorkExperience",
    "Skill",
    "Certification",
    "Project",
    "Publication",
    "Language",
    "OtherSection",
    "Meta",
    "ResumeOutput",
]
