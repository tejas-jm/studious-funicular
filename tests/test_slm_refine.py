import json

from resume_parser.slm_refine import build_slm_prompt, refine_resume_json, sanitize_resume_payload


def test_sanitize_resume_payload_normalizes_sections():
    raw = {
        "contact": {"name": "Alice", "phone": 1234567890, "extra": "ignored"},
        "education": [
            {
                "institution": "Example University",
                "start_date": 2018,
                "end_date": "2022-06",
                "field_of_study": "Computer Science",
            }
        ],
        "work_experience": [
            {
                "company": "ACME",
                "description": "Did things",
                "start_date": "2019-01",
                "end_date": "Present",
                "duration_months": "24",
            }
        ],
        "skills": ["Python", {"name": "SQL", "category": "Database"}],
        "certifications": [
            {"name": "AWS", "issuer": "Amazon", "date": 2021}
        ],
        "projects": [
            {"name": "Parser", "technologies": "Python"}
        ],
        "publications": [
            {"title": "Paper", "venue": "Journal", "date": 2020}
        ],
        "languages": ["English", {"name": "Spanish", "proficiency": "Advanced"}],
        "other_sections": ["Volunteer work"],
        "meta": {"source": "file.pdf", "notes": 123},
    }

    sanitized = sanitize_resume_payload(raw)
    assert sanitized["contact"]["name"] == "Alice"
    assert sanitized["contact"]["phone"] == "1234567890"
    assert sanitized["education"][0]["start_date"] == "2018"
    assert sanitized["work_experience"][0]["description"] == ["Did things"]
    assert sanitized["work_experience"][0]["duration_months"] == 24
    assert sanitized["skills"][0]["name"] == "Python"
    assert sanitized["certifications"][0]["date"] == "2021"
    assert sanitized["projects"][0]["technologies"] == ["Python"]
    assert sanitized["languages"][0]["name"] == "English"
    assert sanitized["other_sections"][0]["content"] == "Volunteer work"
    assert sanitized["meta"]["notes"] == "123"


def test_refine_resume_json_returns_baseline_when_disabled():
    raw = {
        "contact": {"name": "Test"},
        "education": [],
        "work_experience": [],
        "skills": [],
        "certifications": [],
        "projects": [],
        "publications": [],
        "languages": [],
        "other_sections": [],
        "meta": {},
    }

    refined = refine_resume_json(raw, "sample text")
    assert refined == sanitize_resume_payload(raw)


def test_build_slm_prompt_includes_serialized_inputs():
    payload = {"skills": [{"name": "Python"}]}
    prompt = build_slm_prompt(payload, 'Line with """ triple quotes')
    assert json.dumps(payload, ensure_ascii=False, indent=2) in prompt
    assert 'Line with' in prompt
    assert '\"\"\"' in prompt
