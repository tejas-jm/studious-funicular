"""Small language model (SLM) based refinement utilities."""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Dict, Iterable, Optional
from urllib import request

from .schema import ResumeOutput

LOGGER = logging.getLogger(__name__)

PROMPT_TEMPLATE = textwrap.dedent(
    """
    You are a strict JSON normalization engine for resume data.

    INPUTS:
    1. raw_json: a first-pass extracted JSON produced by a resume parsing pipeline.
    2. raw_text: (optional) the full raw text of the resume.

    GOAL:
    - Produce a NEW JSON object that:
      - Is WELL-FORMED and VALID JSON.
      - Is PROPERLY NESTED according to the required schema.
      - Uses ONLY the allowed keys.
      - Does NOT hallucinate information that is not supported by raw_json or raw_text.
      - Fixes obvious structural issues (e.g., wrong nesting, flattened fields).

    REQUIRED TOP-LEVEL SCHEMA:
    {{
      "contact": {{
        "name": string (optional),
        "email": string (optional),
        "phone": string (optional),
        "website": string (optional),
        "location": string (optional),
        "raw": string (optional)
      }},
      "education": [
        {{
          "institution": string,
          "degree": string (optional),
          "field_of_study": string (optional),
          "start_date": string (YYYY or YYYY-MM, optional),
          "end_date": string (YYYY or YYYY-MM or "Present", optional),
          "grade": string (optional),
          "location": string (optional)
        }}
      ],
      "work_experience": [
        {{
          "company": string,
          "position": string (optional),
          "start_date": string (YYYY or YYYY-MM, optional),
          "end_date": string (YYYY or YYYY-MM or "Present", optional),
          "duration_months": integer (optional),
          "location": string (optional),
          "description": [string]
        }}
      ],
      "skills": [
        {{
          "name": string,
          "category": string (optional),
          "proficiency": string (optional)
        }}
      ],
      "certifications": [
        {{
          "name": string,
          "issuer": string (optional),
          "date": string (YYYY or YYYY-MM, optional)
        }}
      ],
      "projects": [
        {{
          "name": string,
          "role": string (optional),
          "start_date": string (YYYY or YYYY-MM, optional),
          "end_date": string (YYYY or YYYY-MM or "Present", optional),
          "description": string (optional),
          "technologies": [string] (optional)
        }}
      ],
      "publications": [
        {{
          "title": string,
          "venue": string (optional),
          "date": string (YYYY or YYYY-MM, optional),
          "description": string (optional)
        }}
      ],
      "languages": [
        {{
          "name": string,
          "proficiency": string (optional)
        }}
      ],
      "other_sections": [
        {{
          "label": string,
          "content": string
        }}
      ],
      "meta": {{
        "source": string (optional),
        "notes": string (optional)
      }}
    }}

    KEY RULES (STRICT):
    1. DO NOT add any new top-level keys beyond:
       - "contact", "education", "work_experience", "skills",
         "certifications", "projects", "publications",
         "languages", "other_sections", "meta".
    2. All arrays MUST be present, even if empty.
    3. If a field is unknown or not confidently supported, either:
       - omit that field, or
       - use an empty string "" (for strings) or empty list [] (for arrays).
    4. DO NOT fabricate entities (companies, degrees, dates, skills, projects, etc.)
       that do NOT appear in either raw_json or raw_text.
    5. You MAY fix structure:
       - Example: move wrongly-placed education info from "skills" into "education".
       - Example: split a combined string into structured fields.
    6. You MUST ensure that every scalar value in the final JSON is:
       - directly present in raw_json/raw_text, OR
       - a normalization/cleaning/trimming/joining of such values (e.g., normalized dates, split names).
    7. NEVER invent model IDs, embeddings, scores, or unrelated metadata.

    VALIDATION REQUIREMENTS:
    - Ensure:
      - "education", "work_experience", "skills",
        "certifications", "projects", "publications",
        "languages", "other_sections"
        are ALWAYS arrays (use [] if nothing).
      - "contact" and "meta" are ALWAYS objects (use {{}} if nothing).
      - Ensure all keys are spelled exactly as in the schema.
      - Ensure JSON is syntactically valid (double quotes, commas, etc.).

    OUTPUT:
    - Return ONLY the final JSON object.
    - No markdown.
    - No comments.
    - No explanations.

    Provided raw_json:
    {raw_json}

    Provided raw_text (may be empty):
    \"\"\"{raw_text}\"\"\"
    """
)

ALLOWED_TOP_LEVEL_KEYS = {
    "contact",
    "education",
    "work_experience",
    "skills",
    "certifications",
    "projects",
    "publications",
    "languages",
    "other_sections",
    "meta",
}

CONTACT_KEYS = {"name", "email", "phone", "website", "location", "raw"}
EDUCATION_KEYS = {
    "institution",
    "degree",
    "field_of_study",
    "start_date",
    "end_date",
    "grade",
    "location",
}
WORK_KEYS = {
    "company",
    "position",
    "start_date",
    "end_date",
    "duration_months",
    "location",
    "description",
}
SKILL_KEYS = {"name", "category", "proficiency"}
CERTIFICATION_KEYS = {"name", "issuer", "date"}
PROJECT_KEYS = {
    "name",
    "role",
    "start_date",
    "end_date",
    "description",
    "technologies",
}
PUBLICATION_KEYS = {"title", "venue", "date", "description"}
LANGUAGE_KEYS = {"name", "proficiency"}
OTHER_SECTION_KEYS = {"label", "content"}
META_KEYS = {"source", "notes"}
ARRAY_KEYS = {
    "education",
    "work_experience",
    "skills",
    "certifications",
    "projects",
    "publications",
    "languages",
    "other_sections",
}

ENABLE_SLM = os.getenv("ENABLE_SLM_REFINER", "0").lower() in {"1", "true", "yes"}
_TIMEOUT = float(os.getenv("SLM_REFINER_TIMEOUT", "60"))


def build_slm_prompt(raw_json: Dict[str, Any], raw_text: Optional[str]) -> str:
    """Build the prompt that is sent to the SLM."""

    json_payload = json.dumps(raw_json, ensure_ascii=False, indent=2, sort_keys=True)
    text_payload = (raw_text or "").replace('"""', '\\"\\"\\"')
    return PROMPT_TEMPLATE.format(raw_json=json_payload, raw_text=text_payload)


def _http_call(prompt: str) -> Optional[str]:
    endpoint = os.getenv("SLM_REFINER_ENDPOINT")
    if not endpoint:
        return None
    body = json.dumps({"prompt": prompt}).encode("utf-8")
    req = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=_TIMEOUT) as response:
            content_type = response.headers.get("Content-Type", "")
            payload = response.read().decode("utf-8")
    except Exception as error:  # pragma: no cover - network failure
        LOGGER.error("SLM HTTP call failed: %s", error)
        return None
    if "application/json" in content_type:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return None
        if isinstance(data, dict):
            return str(data.get("output") or data.get("response") or "")
    return payload


_GENERATOR = None


def _local_generator(prompt: str) -> Optional[str]:
    model_name = os.getenv("SLM_REFINER_MODEL")
    if not model_name:
        return None
    global _GENERATOR
    if _GENERATOR is None:
        try:  # pragma: no cover - heavy dependency branch
            from transformers import pipeline
        except Exception as error:  # pragma: no cover
            LOGGER.error("Unable to import transformers pipeline: %s", error)
            return None
        try:
            _GENERATOR = pipeline(
                "text-generation",
                model=model_name,
                device_map="auto" if os.getenv("SLM_REFINER_USE_GPU") else None,
            )
        except Exception as error:  # pragma: no cover
            LOGGER.error("Failed to load SLM model %s: %s", model_name, error)
            return None
    try:
        outputs = _GENERATOR(
            prompt,
            max_new_tokens=int(os.getenv("SLM_REFINER_MAX_TOKENS", "512")),
            temperature=float(os.getenv("SLM_REFINER_TEMPERATURE", "0.1")),
            return_full_text=False,
        )
    except Exception as error:  # pragma: no cover
        LOGGER.error("SLM generation failed: %s", error)
        return None
    if not outputs:
        return None
    result = outputs[0]
    if isinstance(result, dict):
        return str(result.get("generated_text", ""))
    return str(result)


def _call_slm(prompt: str) -> Optional[str]:
    """Call the configured SLM endpoint or model."""

    response = _http_call(prompt)
    if response:
        return response
    return _local_generator(prompt)


def _filter_dict(source: Any, allowed_keys: Iterable[str]) -> Dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {key: source.get(key) for key in allowed_keys if key in source}


def _sanitize_array(items: Any, allowed_keys: Iterable[str]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return sanitized
    for entry in items:
        if isinstance(entry, dict):
            sanitized.append(_filter_dict(entry, allowed_keys))
        elif allowed_keys == SKILL_KEYS and isinstance(entry, str):
            sanitized.append({"name": entry})
        elif allowed_keys == LANGUAGE_KEYS and isinstance(entry, str):
            sanitized.append({"name": entry})
        elif allowed_keys == OTHER_SECTION_KEYS and isinstance(entry, str):
            sanitized.append({"label": "other", "content": entry})
    return sanitized


def _stringify_fields(entry: Dict[str, Any], string_keys: Iterable[str]) -> None:
    for key in list(entry.keys()):
        value = entry[key]
        if key not in string_keys or value in (None, ""):
            continue
        if isinstance(value, list):
            entry[key] = [str(item).strip() for item in value if str(item).strip()]
        elif not isinstance(value, str):
            entry[key] = str(value).strip()


def sanitize_resume_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize payload to match the ResumeOutput schema."""

    sanitized: Dict[str, Any] = {key: [] if key in ARRAY_KEYS else {} for key in ALLOWED_TOP_LEVEL_KEYS}
    sanitized["contact"] = _filter_dict(payload.get("contact"), CONTACT_KEYS)
    _stringify_fields(sanitized["contact"], CONTACT_KEYS)

    sanitized["education"] = _sanitize_array(payload.get("education"), EDUCATION_KEYS)
    for entry in sanitized["education"]:
        _stringify_fields(entry, EDUCATION_KEYS)

    work_entries = _sanitize_array(payload.get("work_experience"), WORK_KEYS)
    for entry in work_entries:
        description = entry.get("description")
        if isinstance(description, list):
            entry["description"] = [str(item).strip() for item in description if str(item).strip()]
        else:
            entry["description"] = [str(description).strip()] if description else []
        _stringify_fields(entry, WORK_KEYS - {"duration_months", "description"})
        duration = entry.get("duration_months")
        if duration not in (None, ""):
            try:
                entry["duration_months"] = int(duration)
            except (ValueError, TypeError):
                entry["duration_months"] = None
    sanitized["work_experience"] = work_entries

    sanitized["skills"] = _sanitize_array(payload.get("skills"), SKILL_KEYS)
    for entry in sanitized["skills"]:
        _stringify_fields(entry, SKILL_KEYS)

    sanitized["certifications"] = _sanitize_array(payload.get("certifications"), CERTIFICATION_KEYS)
    for entry in sanitized["certifications"]:
        _stringify_fields(entry, CERTIFICATION_KEYS)

    sanitized["projects"] = _sanitize_array(payload.get("projects"), PROJECT_KEYS)
    for entry in sanitized["projects"]:
        _stringify_fields(entry, PROJECT_KEYS - {"technologies"})
        technologies = entry.get("technologies")
        if isinstance(technologies, list):
            entry["technologies"] = [str(item).strip() for item in technologies if str(item).strip()]
        elif technologies:
            entry["technologies"] = [str(technologies).strip()]
        else:
            entry["technologies"] = []

    sanitized["publications"] = _sanitize_array(payload.get("publications"), PUBLICATION_KEYS)
    for entry in sanitized["publications"]:
        _stringify_fields(entry, PUBLICATION_KEYS)

    sanitized["languages"] = _sanitize_array(payload.get("languages"), LANGUAGE_KEYS)
    for entry in sanitized["languages"]:
        _stringify_fields(entry, LANGUAGE_KEYS)

    sanitized["other_sections"] = _sanitize_array(payload.get("other_sections"), OTHER_SECTION_KEYS)
    for entry in sanitized["other_sections"]:
        _stringify_fields(entry, OTHER_SECTION_KEYS)

    meta_payload = _filter_dict(payload.get("meta"), META_KEYS)
    _stringify_fields(meta_payload, META_KEYS)
    sanitized["meta"] = meta_payload

    resume = ResumeOutput.from_dict(sanitized)
    resume.validate()
    return resume.to_dict()


def refine_resume_json(raw_json: Dict[str, Any], raw_text: Optional[str] = None) -> Dict[str, Any]:
    """Refine resume JSON using an optional SLM step."""

    baseline = sanitize_resume_payload(raw_json)
    if not ENABLE_SLM:
        return baseline

    prompt = build_slm_prompt(raw_json, raw_text)
    response_text = _call_slm(prompt)
    if not response_text:
        LOGGER.info("SLM refinement skipped or returned no data; using baseline payload")
        return baseline

    candidate_text = response_text.strip()
    if candidate_text.startswith("```"):
        candidate_text = candidate_text.strip("`")
        candidate_text = candidate_text.replace("json", "", 1).strip()

    try:
        candidate_json = json.loads(candidate_text)
    except json.JSONDecodeError as error:
        LOGGER.warning("Failed to parse SLM response as JSON: %s", error)
        return baseline

    try:
        refined = sanitize_resume_payload(candidate_json)
    except Exception as error:
        LOGGER.warning("SLM output failed validation; falling back to baseline: %s", error)
        return baseline

    baseline_str = json.dumps(baseline, sort_keys=True)
    refined_str = json.dumps(refined, sort_keys=True)
    if baseline_str != refined_str:
        LOGGER.info("SLM refinement updated the resume payload")
    return refined


__all__ = ["build_slm_prompt", "refine_resume_json", "sanitize_resume_payload"]
