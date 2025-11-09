# LayoutLMv3 Resume Parser

An end-to-end resume parsing pipeline that ingests DOCX/DOC/PDF (including scanned PDFs), runs LayoutLMv3 for contextual embeddings, and maps the content into a structured JSON schema.

## Features
- File-type detection with dedicated ingestion paths for DOC/DOCX (python-docx/docx2txt) and PDF (pdfplumber with OCR fallback).
- Normalization of word-level bounding boxes for LayoutLMv3 compatibility.
- LayoutLMv3 inference without fine-tuning for contextual embeddings and downstream heuristics.
- Heuristic post-processing for section segmentation, date normalization, skill deduplication, and structured field extraction.
- Dataclass-backed schema describing the deeply nested JSON output (contact, education, experience, skills, certifications, projects, publications, languages, and other sections).
- Optional SLM-powered refinement step that normalizes the heuristic output into schema-valid JSON without hallucinating new data.

- Notebook and CLI examples for running the pipeline end-to-end.

## Installation

```bash
pip install -r requirements.txt  # create this file based on your environment
```

Ensure optional dependencies for OCR are installed when processing scanned PDFs:

```bash
pip install easyocr pdf2image
sudo apt-get install poppler-utils  # required by pdf2image
```

## Usage

### Python API

```python
from resume_parser import parse_resume

result = parse_resume("/path/to/resume.pdf")
print(result)
```

### Command-Line Interface

```bash
python examples/run_pipeline.py /path/to/resume.pdf --output parsed.json
```

### Notebook Demo

Open `notebooks/resume_parser_demo.ipynb` (or upload it to Google Colab) and follow the instructions to upload sample resumes and inspect the JSON output produced by the schema-aware pipeline and optional SLM normalizer.


## JSON Schema

The pipeline produces an object matching the schema below. Optional fields may be omitted when not available, but arrays are always included.

```json
{
  "contact": {
    "name": "Ada Lovelace",
    "email": "ada@example.com",
    "phone": "+44 20 1234 5678",
    "website": "https://adalovelace.dev",
    "location": "12 Analytical Engine Way, London",
    "raw": "Ada Lovelace\nada@example.com\n+44 20 1234 5678"
  },
  "education": [
    {
      "institution": "University of London",
      "degree": "Bachelor of Science",
      "field_of_study": "Mathematics",
      "start_date": "1833-01",
      "end_date": "1835-12",
      "grade": null,
      "location": "London"
    }
  ],
  "work_experience": [
    {
      "company": "Analytical Engines Ltd",
      "position": "Algorithm Designer",
      "start_date": "1842-01",
      "end_date": "1845-06",
      "duration_months": 41,
      "location": "London",
      "description": [
        "Drafted the first published algorithm.",
        "Collaborated with Charles Babbage on engine design notes."
      ]
    }
  ],
  "skills": [
    {"name": "Mathematics"},
    {"name": "Analytical Engine", "category": "Hardware"},
    {"name": "Technical Writing"}
  ],
  "certifications": [
    {
      "name": "Royal Society Fellowship",
      "issuer": "Royal Society",
      "date": "1843-01"
    }
  ],
  "projects": [
    {
      "name": "Engine Notes",
      "role": "Researcher",
      "description": "Annotated plans and notes for the Analytical Engine.",
      "technologies": [
        "Mathematics",
        "Mechanical Computing"
      ]
    }
  ],
  "publications": [
    {
      "title": "Sketch of the Analytical Engine",
      "venue": "Scientific Memoirs",
      "date": "1843-10",
      "description": "English translation and expansion of Menabrea's paper."
    }
  ],
  "languages": [
    {
      "name": "English",
      "proficiency": "Native"
    }
  ],
  "other_sections": [
    {
      "label": "Volunteer Work",
      "content": "Advocated for early computing research."
    }
  ],
  "meta": {
    "source": "ada_lovelace.pdf",
    "notes": "token_embeddings=1234"
  }
}
```

The schema is implemented with Python dataclasses that perform lightweight runtime validation of dates, lists, and optional fields. The optional SLM refinement step can be enabled by setting `ENABLE_SLM_REFINER=1` and configuring an endpoint or local model; when disabled, the pipeline returns the heuristic baseline already coerced into this schema.

### Enabling the SLM Refinement Step

Set the following environment variables to activate the SLM-based normalizer:

| Variable | Description |
| --- | --- |
| `ENABLE_SLM_REFINER` | Toggle (`1`/`true`) to enable the SLM call. |
| `SLM_REFINER_ENDPOINT` | Optional HTTP endpoint that accepts `{"prompt": ...}` JSON payloads and returns a JSON or text response. |
| `SLM_REFINER_MODEL` | Optional local Hugging Face model ID to load via `transformers.pipeline` when no endpoint is provided. |
| `SLM_REFINER_MAX_TOKENS` / `SLM_REFINER_TEMPERATURE` | Generation parameters when using a local model. |

If neither an endpoint nor a model is configured, the pipeline automatically falls back to the baseline heuristic output.

## Limitations & Future Work
- The LayoutLMv3 model is used without fine-tuning; expect noisy predictions.
- DOC ingestion lacks precise layout metadata because DOCX does not expose absolute positions.
- OCR quality directly affects downstream accuracy for scanned PDFs.
- Post-processing relies on heuristics; consider integrating learned classifiers or rule-based engines.
- Future enhancements include custom fine-tuning, richer section detection, and support for multilingual resumes.
