"""LayoutLMv3 inference utilities for the resume parsing pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from PIL import Image
from transformers import LayoutLMv3Model, LayoutLMv3Processor

from .types import DocumentContent, PageContent, TokenEmbedding

LOGGER = logging.getLogger(__name__)

DEFAULT_LABELS = [
    "O",
    "B-CONTACT",
    "I-CONTACT",
    "B-EDUCATION",
    "I-EDUCATION",
    "B-WORK",
    "I-WORK",
    "B-SKILL",
    "I-SKILL",
]


@dataclass
class InferenceConfig:
    """Configuration for LayoutLMv3 inference."""

    model_name: str = "microsoft/layoutlmv3-base"
    device: Optional[str] = None
    max_length: int = 512
    chunk_overlap: int = 32


class LayoutLMv3Inference:
    """Run LayoutLMv3 to obtain contextual embeddings for resume tokens."""

    def __init__(self, config: Optional[InferenceConfig] = None) -> None:
        self.config = config or InferenceConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("Using device %s for LayoutLMv3", self.device)
        self.processor = LayoutLMv3Processor.from_pretrained(self.config.model_name, apply_ocr=False)
        self.model = LayoutLMv3Model.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _load_page_image(self, page: PageContent) -> Image.Image:
        if page.metadata.image_path:
            try:
                return Image.open(page.metadata.image_path).convert("RGB")
            except FileNotFoundError:
                LOGGER.warning("Image %s not found; creating blank canvas", page.metadata.image_path)
        width = max(page.metadata.width, 1)
        height = max(page.metadata.height, 1)
        return Image.new("RGB", (width, height), color="white")

    def _chunk_tokens(self, page: PageContent) -> Iterable[List[int]]:
        token_indices = list(range(len(page.tokens)))
        max_len = self.config.max_length
        overlap = self.config.chunk_overlap
        if len(token_indices) <= max_len:
            yield token_indices
            return
        start = 0
        while start < len(token_indices):
            end = min(start + max_len, len(token_indices))
            yield token_indices[start:end]
            if end == len(token_indices):
                break
            start = end - overlap
            if start < 0:
                start = 0

    def _encode_chunk(self, page: PageContent, indices: List[int]) -> Dict[str, torch.Tensor]:
        image = self._load_page_image(page)
        words = [page.tokens[i].text for i in indices]
        boxes = [page.tokens[i].bbox.as_tuple() for i in indices]
        encoding = self.processor(
            images=image,
            text=words,
            boxes=boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_length,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        return encoding

    def predict_page(self, page: PageContent) -> List[TokenEmbedding]:
        token_embeddings: List[TokenEmbedding] = []
        for indices in self._chunk_tokens(page):
            encoding = self._encode_chunk(page, indices)
            with torch.no_grad():
                outputs = self.model(**encoding)
            hidden_states = outputs.last_hidden_state.cpu()
            for i, token_index in enumerate(indices):
                if i >= hidden_states.shape[1]:
                    break
                embedding = hidden_states[0, i].tolist()
                token_embeddings.append(TokenEmbedding(token=page.tokens[token_index], embedding=embedding))
        return token_embeddings

    def predict(self, document: DocumentContent) -> List[TokenEmbedding]:
        results: List[TokenEmbedding] = []
        for page in document.pages:
            results.extend(self.predict_page(page))
        return results
