"""ค้นมาตราที่เกี่ยวข้องด้วย FAISS (ความหมาย) + BM25 (คำตรงตัว) แล้วรวมอันดับด้วย RRF"""
from __future__ import annotations

from dataclasses import dataclass

# ต้อง import torch (ผ่าน sentence_transformers) ก่อน faiss — ถ้าสลับลำดับ libomp สองชุดชนกันแล้ว segfault บน macOS
from sentence_transformers import SentenceTransformer  # isort: skip
import faiss  # isort: skip
import numpy as np
from pythainlp.tokenize import word_tokenize
from pythainlp.util import thai_digit_to_arabic_digit
from rank_bm25 import BM25Okapi

from rag.loader import Chunk

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
# ปรับจากผล evaluate.py (Task 7) — ค่าสุดท้ายบันทึกใน README
DEFAULT_THRESHOLD = 0.80
CANDIDATES = 8
RRF_K = 60


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    rrf_score: float
    cosine: float


def tokenize(text: str) -> list[str]:
    text = thai_digit_to_arabic_digit(text).lower()
    return [t for t in word_tokenize(text, engine="newmm", keep_whitespace=False) if t.strip()]


def _index_text(chunk: Chunk) -> str:
    return f"{chunk.law_name} {chunk.chapter}\n{chunk.text}"


class Retriever:
    def __init__(self, chunks: list[Chunk], model_name: str = DEFAULT_MODEL):
        if not chunks:
            raise ValueError("ไม่มี chunk ให้สร้าง index — ตรวจโฟลเดอร์ data/")
        self.chunks = chunks
        self.model = SentenceTransformer(model_name, device="cpu")
        vectors = self._encode([f"passage: {_index_text(c)}" for c in chunks])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.bm25 = BM25Okapi([tokenize(_index_text(c)) for c in chunks])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype="float32")

    def search(self, query: str, k: int = 4, laws: set[str] | None = None) -> list[Hit]:
        query = thai_digit_to_arabic_digit(query).strip()
        allowed = [i for i, c in enumerate(self.chunks) if laws is None or c.law in laws]
        if not query or not allowed or k <= 0:
            return []

        # FAISS: คลังเล็ก (ไม่กี่ร้อย chunk) จึงค้นทั้งหมดแล้วกรองตามกฎหมายที่เลือก
        scores, ids = self.index.search(self._encode([f"query: {query}"]), self.index.ntotal)
        cosine = np.zeros(len(self.chunks), dtype="float32")
        cosine[ids[0]] = scores[0]
        allowed_set = set(allowed)
        vector_rank = [int(i) for i in ids[0] if int(i) in allowed_set][:CANDIDATES]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_rank = [
            i for i in sorted(allowed, key=lambda i: bm25_scores[i], reverse=True)
            if bm25_scores[i] > 0
        ][:CANDIDATES]

        fused: dict[int, float] = {}
        for ranking in (vector_rank, bm25_rank):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [Hit(self.chunks[i], fused[i], float(cosine[i])) for i in top]


def is_relevant(hits: list[Hit], threshold: float = DEFAULT_THRESHOLD) -> bool:
    return bool(hits) and max(h.cosine for h in hits) >= threshold
