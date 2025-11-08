import importlib.util
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "resume_parser" / "schema.py"

spec = importlib.util.spec_from_file_location("resume_parser.schema", SCHEMA_PATH)
schema = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules.setdefault(spec.name, schema)
spec.loader.exec_module(schema)

Contact = schema.Contact
Education = schema.Education
ResumeOutput = schema.ResumeOutput
WorkExperience = schema.WorkExperience


def test_resume_output_serialization_roundtrip():
    resume = ResumeOutput(
        contact=Contact(email="test@example.com", name="Test User"),
        education=[Education(institution="Test University", start_date="2020-01", end_date="2022-05")],
        work_experience=[
            WorkExperience(
                company="Example Corp",
                position="Engineer",
                start_date="2019-01",
                end_date="2020-01",
                duration_months=12,
                description=["Worked on projects."],
            )
        ],
        skills=["Python", "Data Analysis"],
    )

    payload = json.loads(resume.json())
    assert payload["contact"]["email"] == "test@example.com"
    assert payload["education"][0]["institution"] == "Test University"
    assert payload["work_experience"][0]["duration_months"] == 12
    assert payload["projects"] == []
    assert payload["raw_text"] is None


def test_resume_output_missing_optional_fields():
    payload = {
        "contact": {},
        "education": [],
        "work_experience": [],
        "skills": [],
        "certifications": [],
        "projects": [],
        "publications": [],
        "languages": [],
        "other_sections": [],
        "raw_text": None,
        "meta": {},
    }

    resume = ResumeOutput.from_dict(payload)
    assert resume.contact.email is None
    assert resume.skills == []
    assert resume.meta == {}
