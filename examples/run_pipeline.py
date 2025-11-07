"""Command-line helper to run the resume parsing pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from resume_parser import ResumeOutput, parse_resume

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse a resume into structured JSON")
    parser.add_argument("file", type=Path, help="Path to the resume file (PDF/DOCX/DOC)")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save the JSON output")
    args = parser.parse_args()

    resume_output: ResumeOutput = parse_resume(str(args.file))
    json_payload = resume_output.json(indent=2, ensure_ascii=False)
    print(json_payload)

    if args.output:
        args.output.write_text(json_payload)
        logging.info("Saved output to %s", args.output)


if __name__ == "__main__":
    main()
