"""Output schema definitions for the resume parsing pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


def _validate_date(value: Optional[str]) -> Optional[str]:
    if value is None or value == "present":
        return value
    if len(value) == 4:
        datetime.strptime(value, "%Y")
        return value
    datetime.strptime(value, "%Y-%m")
    return value


def _ensure_dict(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass
class Contact:
    """Basic contact details extracted from a resume."""

    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    address: Optional[str] = None
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
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_date:
            self.start_date = _validate_date(self.start_date)
        if self.end_date:
            self.end_date = _validate_date(self.end_date)
        self.extra = _ensure_dict(self.extra)


@dataclass
class WorkExperience:
    """Professional experience item."""

    company: Optional[str] = None
    position: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    duration_months: Optional[int] = None
    description: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start_date:
            self.start_date = _validate_date(self.start_date)
        if self.end_date:
            self.end_date = _validate_date(self.end_date)
        self.extra = _ensure_dict(self.extra)
        self.description = [item.strip() for item in self.description if item]


@dataclass
class Certification:
    """Certification or license entry."""

    name: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.date:
            try:
                self.date = _validate_date(self.date)
            except ValueError:
                pass
        self.extra = _ensure_dict(self.extra)


@dataclass
class Project:
    """Project entry including technologies and description."""

    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = field(default_factory=list)
    date: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.date:
            try:
                self.date = _validate_date(self.date)
            except ValueError:
                pass
        self.extra = _ensure_dict(self.extra)
        self.technologies = [tech.strip() for tech in self.technologies if tech.strip()]


@dataclass
class Publication:
    """Publication entry for articles, papers, etc."""

    title: Optional[str] = None
    publication: Optional[str] = None
    date: Optional[str] = None
    url: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.date:
            try:
                self.date = _validate_date(self.date)
            except ValueError:
                pass
        self.extra = _ensure_dict(self.extra)


@dataclass
class Language:
    """Language proficiency entry."""

    language: Optional[str] = None
    fluency: Optional[str] = None


@dataclass
class OtherSection:
    """Catch-all section for unparsed or custom resume segments."""

    label: Optional[str] = None
    content: Optional[str] = None


@dataclass
class ResumeOutput:
    """Top-level structured resume representation."""

    contact: Contact = field(default_factory=Contact)
    education: List[Education] = field(default_factory=list)
    work_experience: List[WorkExperience] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    certifications: List[Certification] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    publications: List[Publication] = field(default_factory=list)
    languages: List[Language] = field(default_factory=list)
    other_sections: List[OtherSection] = field(default_factory=list)
    raw_text: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.education = [item if isinstance(item, Education) else Education(**item) for item in self.education]
        self.work_experience = [item if isinstance(item, WorkExperience) else WorkExperience(**item) for item in self.work_experience]
        self.certifications = [item if isinstance(item, Certification) else Certification(**item) for item in self.certifications]
        self.projects = [item if isinstance(item, Project) else Project(**item) for item in self.projects]
        self.publications = [item if isinstance(item, Publication) else Publication(**item) for item in self.publications]
        self.languages = [item if isinstance(item, Language) else Language(**item) for item in self.languages]
        self.other_sections = [item if isinstance(item, OtherSection) else OtherSection(**item) for item in self.other_sections]
        self.skills = [skill.strip() for skill in self.skills if skill]
        self.meta = _ensure_dict(self.meta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contact": asdict(self.contact),
            "education": [asdict(item) for item in self.education],
            "work_experience": [asdict(item) for item in self.work_experience],
            "skills": list(self.skills),
            "certifications": [asdict(item) for item in self.certifications],
            "projects": [asdict(item) for item in self.projects],
            "publications": [asdict(item) for item in self.publications],
            "languages": [asdict(item) for item in self.languages],
            "other_sections": [asdict(item) for item in self.other_sections],
            "raw_text": self.raw_text,
            "meta": dict(self.meta),
        }

    def json(self, indent: int = 2, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=ensure_ascii)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ResumeOutput":
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
            raw_text=payload.get("raw_text"),
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
        if not isinstance(self.meta, dict):
            raise TypeError("meta must be a dictionary")


__all__ = [
    "Contact",
    "Education",
    "WorkExperience",
    "Certification",
    "Project",
    "Publication",
    "Language",
    "OtherSection",
    "ResumeOutput",
]
