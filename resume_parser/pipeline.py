"""End-to-end resume parsing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from . import ingestion, inference, postprocessing
from .types import DocumentContent, ParsedResume

LOGGER = logging.getLogger(__name__)


class ResumeParser:
    """High-level orchestrator for the resume parsing pipeline."""

    def __init__(
        self,
        ingestion_config: Optional[ingestion.IngestionConfig] = None,
        inference_config: Optional[inference.InferenceConfig] = None,
    ) -> None:
        self.ingestion_config = ingestion_config or ingestion.IngestionConfig()
        self.inference_config = inference_config or inference.InferenceConfig()
        self.inference_engine = inference.LayoutLMv3Inference(self.inference_config)

    def load_document(self, file_path: str) -> DocumentContent:
        LOGGER.info("Ingesting document %s", file_path)
        document = ingestion.ingest_document(file_path, self.ingestion_config)
        LOGGER.debug("Loaded %s tokens", len(document.tokens))
        return document

    def run_inference(self, document: DocumentContent):
        LOGGER.info("Running LayoutLMv3 inference")
        embeddings = self.inference_engine.predict(document)
        LOGGER.debug("Obtained %s token embeddings", len(embeddings))
        return embeddings

    def post_process(self, document: DocumentContent, embeddings) -> ParsedResume:
        LOGGER.info("Linking entities into structured resume")
        resume = postprocessing.link_entities(document, embeddings)
        return resume

    def parse(self, file_path: str) -> ParsedResume:
        document = self.load_document(file_path)
        embeddings = self.run_inference(document)
        return self.post_process(document, embeddings)


def parse_resume(file_path: str) -> ParsedResume:
    """Convenience function to parse a resume into structured JSON."""

    parser = ResumeParser()
    return parser.parse(str(Path(file_path).expanduser().resolve()))
