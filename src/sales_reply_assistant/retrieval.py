"""Semantic search over the knowledge base.

The knowledge base is small (tens of entries), so an in-memory matrix with cosine similarity
is both the simplest and the fastest option — no vector database needed.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
import numpy.typing as npt

from sales_reply_assistant.knowledge import EntryType, KnowledgeBase, KnowledgeEntry

Matrix = npt.NDArray[np.float32]


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class MatchReason(StrEnum):
    SEARCH = "search"  # found by similarity to the client's message
    RELATED = "related"  # complements a found product
    PINNED = "pinned"  # always included


@dataclass(frozen=True, slots=True)
class RetrievedEntry:
    entry: KnowledgeEntry
    reason: MatchReason
    score: float | None = None


class Retriever:
    """Finds the entries relevant to a query and expands them with upsell material.

    Products and store information are ranked separately, each with its own ``top_k`` quota.
    Otherwise product names in a question ("can I take your probiotic while pregnant?") push
    the policy that actually answers it out of the context.

    The result is ordered: search hits by score, then products related to the hits,
    then pinned entries. Each entry appears once.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedder: Embedder,
        *,
        top_k: int = 3,
        min_score: float = 0.4,
        max_related: int = 3,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._entries = knowledge_base.entries
        self._embedder = embedder
        self._top_k = top_k
        self._min_score = min_score
        self._max_related = max_related
        self._groups = [
            np.array([i for i, e in enumerate(self._entries) if e.type is EntryType.PRODUCT]),
            np.array([i for i, e in enumerate(self._entries) if e.type is not EntryType.PRODUCT]),
        ]
        self._matrix: Matrix | None = None
        self._lock = asyncio.Lock()

    async def warm_up(self) -> None:
        """Embed the knowledge base once. Safe to call concurrently and repeatedly."""
        await self._index()

    async def retrieve(self, query: str) -> list[RetrievedEntry]:
        matrix = await self._index()
        query_vector = _normalize(np.asarray(await self._embedder.embed([query]), np.float32))[0]
        scores = matrix @ query_vector

        hits = [
            RetrievedEntry(self._entries[index], MatchReason.SEARCH, float(scores[index]))
            for group in self._groups
            if group.size
            for index in group[np.argsort(-scores[group])[: self._top_k]]
            if scores[index] >= self._min_score
        ]
        hits.sort(key=lambda hit: hit.score or 0, reverse=True)
        found = {hit.entry.id: hit for hit in hits}

        for entry in self._related_products(hits):
            found[entry.id] = RetrievedEntry(entry, MatchReason.RELATED)

        for entry in self._knowledge_base.pinned:
            found.setdefault(entry.id, RetrievedEntry(entry, MatchReason.PINNED))

        return list(found.values())

    def _related_products(self, hits: Sequence[RetrievedEntry]) -> list[KnowledgeEntry]:
        seen = {hit.entry.id for hit in hits}
        related: list[KnowledgeEntry] = []
        for hit in hits:
            for related_id in hit.entry.related:
                entry = self._knowledge_base.get(related_id)
                if entry is None or entry.id in seen or entry.type is not EntryType.PRODUCT:
                    continue
                if len(related) >= self._max_related:
                    return related
                seen.add(entry.id)
                related.append(entry)
        return related

    async def _index(self) -> Matrix:
        if self._matrix is None:
            async with self._lock:
                if self._matrix is None:
                    texts = [entry.search_text for entry in self._entries]
                    vectors = await self._embedder.embed(texts)
                    self._matrix = _normalize(np.asarray(vectors, np.float32))
        return self._matrix


def _normalize(matrix: Matrix) -> Matrix:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32)
