# LayoutLMv3 Resume Parser

An end-to-end resume parsing pipeline that ingests DOCX/DOC/PDF (including scanned PDFs), runs LayoutLMv3 for contextual embeddings, and maps the content into a structured JSON schema.

## Features
- File-type detection with dedicated ingestion paths for DOC/DOCX (python-docx/docx2txt) and PDF (pdfplumber with OCR fallback).
- Normalization of word-level bounding boxes for LayoutLMv3 compatibility.
- LayoutLMv3 inference without fine-tuning for contextual embeddings and downstream heuristics.
- Heuristic post-processing for section segmentation, date normalization, and skill deduplication.
- JSON output schema covering contact information, education, work experience, skills, certifications, projects, publications, languages, and other sections.
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

Open `notebooks/resume_parser_demo.ipynb` (or upload it to Google Colab) and follow the instructions to upload sample resumes and inspect the JSON output.

## JSON Schema

The pipeline produces an object matching the schema below. Optional fields may be omitted when not available, but arrays are always included.

```json
{
  "contact": {
    "email": "",
    "phone": "",
    "website": "",
    "raw": ""
  },
  "education": [
    {
      "institution": "",
      "degree": "",
      "field_of_study": "",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM",
      "grade": ""
    }
  ],
  "work_experience": [
    {
      "organization": "",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM",
      "duration_months": 0,
      "description": [
        ""
      ]
    }
  ],
  "skills": [""],
  "certifications": [
    {"name": "", "issuer": "", "date": ""}
  ],
  "projects": [
    {"name": "", "description": "", "technologies": []}
  ],
  "publications": [
    {"title": "", "date": "", "link": ""}
  ],
  "languages": [""],
  "other_sections": [
    {"label": "", "content": ""}
  ],
  "meta": {}
}
```

## Limitations & Future Work
- The LayoutLMv3 model is used without fine-tuning; expect noisy predictions.
- DOC ingestion lacks precise layout metadata because DOCX does not expose absolute positions.
- OCR quality directly affects downstream accuracy for scanned PDFs.
- Post-processing relies on heuristics; consider integrating learned classifiers or rule-based engines.
- Future enhancements include custom fine-tuning, richer section detection, and support for multilingual resumes.
