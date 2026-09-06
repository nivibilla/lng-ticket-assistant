"""Jina retrieval embeddings shared by indexing and search."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

DEFAULT_EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
EMBEDDING_DIMENSION = 1024
RETRIEVAL_TASK = "retrieval"
QUERY_PROMPT = "query"
DOCUMENT_PROMPT = "document"


class EmbeddingError(RuntimeError):
    """Raised when the embedding model returns unusable output."""


class JinaRetrievalEmbedder:
    """Lazily load Jina and encode asymmetric retrieval vectors."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        model: Any | None = None,
    ) -> None:
        self.model_id = (
            model_id
            or os.getenv("EMBEDDING_MODEL_ID")
            or DEFAULT_EMBEDDING_MODEL_ID
        )
        self._model = model

    @property
    def model(self) -> Any:
        """Return the loaded model, downloading it on first use."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        import torch
        from transformers import AutoModel

        if torch.cuda.is_available():
            device = torch.device("cuda")
            load_kwargs: dict[str, Any] = {"dtype": torch.bfloat16}
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            load_kwargs = {"dtype": torch.float32}
        else:
            device = torch.device("cpu")
            load_kwargs = {"dtype": torch.float32}

        model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **load_kwargs,
        )
        model = model.to(device=device)
        model.eval()
        return model

    def encode_query(self, text: str) -> np.ndarray:
        """Encode one search query using Jina's retrieval query prompt."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("The search query cannot be empty.")
        return self._encode([clean_text], prompt_name=QUERY_PROMPT)[0]

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Encode knowledge-base chunks using Jina's document prompt."""
        clean_texts = [str(text).strip() for text in texts]
        if not clean_texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        if any(not text for text in clean_texts):
            raise ValueError("Knowledge-base documents cannot be empty.")
        return self._encode(clean_texts, prompt_name=DOCUMENT_PROMPT)

    def _encode(self, texts: list[str], *, prompt_name: str) -> np.ndarray:
        raw_embeddings = self.model.encode(
            texts=texts,
            task=RETRIEVAL_TASK,
            prompt_name=prompt_name,
        )
        if hasattr(raw_embeddings, "detach"):
            raw_embeddings = raw_embeddings.detach().cpu().float().numpy()

        embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        expected_shape = (len(texts), EMBEDDING_DIMENSION)
        if embeddings.shape != expected_shape:
            raise EmbeddingError(
                "Unexpected embedding shape "
                f"{embeddings.shape}; expected {expected_shape}."
            )
        if not np.isfinite(embeddings).all():
            raise EmbeddingError("Embedding output contains non-finite values.")
        return embeddings
