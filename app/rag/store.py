from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.errors import LLMUnavailableError


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    section: str
    text: str


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...


def chunk_markdown(path: Path) -> list[Chunk]:
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^# (.+)$", text, flags=re.M)
    title = title_match.group(1).strip() if title_match else path.stem
    chunks = []
    for i, block in enumerate(re.split(r"^## ", text, flags=re.M)[1:]):
        header, _, body = block.partition("\n")
        body = body.strip()
        if body:
            chunks.append(Chunk(id=f"{path.stem}#{i}", source=path.name, title=title,
                                section=header.strip(), text=body))
    return chunks


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class HashingEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        t = "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))
        words = re.findall(r"[a-z0-9]+", t)
        feats = [w for w in words if len(w) > 2]
        feats += [w[:5] for w in words if len(w) > 5]
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                m[i, h % self.dim] += 1.0
        return _normalize(m)


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str, base_url: str | None, timeout_s: float):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or "not-needed", base_url=base_url, timeout=timeout_s, max_retries=1)
        self.model = model
        self.last_usage = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        import openai

        try:
            resp = self.client.embeddings.create(model=self.model, input=texts)
        except openai.APIError as e:
            raise LLMUnavailableError(f"Fallo del proveedor de embeddings: {type(e).__name__}") from e
        self.last_usage = resp.usage.total_tokens if resp.usage else 0
        return _normalize(np.array([d.embedding for d in resp.data], dtype=np.float32))


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, embedder: Embedder, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No hay documentos que indexar")
        self.embedder = embedder
        self.chunks = chunks
        self.matrix = embedder.embed([f"{c.title}. {c.section}. {c.text}" for c in chunks])

    @classmethod
    def from_dir(cls, embedder: Embedder, docs_dir: Path) -> VectorStore:
        chunks = [c for p in sorted(docs_dir.glob("*.md")) for c in chunk_markdown(p)]
        return cls(embedder, chunks)

    def search(self, query: str, k: int = 4, min_score: float = 0.05) -> list[Hit]:
        q = self.embedder.embed([query])[0]
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]
