"""Command-line helper to run the resume parsing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from resume_parser import parse_resume

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a resume into structured JSON")
    parser.add_argument("file", type=Path, help="Path to the resume file (PDF/DOCX/DOC)")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the JSON output")
    args = parser.parse_args()

    resume_json = parse_resume(str(args.file))
    print(json.dumps(resume_json, indent=2, ensure_ascii=False))

    if args.output:
        args.output.write_text(json.dumps(resume_json, indent=2, ensure_ascii=False))
        logging.info("Saved output to %s", args.output)


if __name__ == "__main__":
    main()
