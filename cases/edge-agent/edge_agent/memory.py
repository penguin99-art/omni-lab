"""Three-tier memory system: SOUL.md, USER.md, MEMORY.md + session history.

Storage and retrieval live here. Prompt / context assembly now lives under
`edge_agent.context`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


class MemoryStore:
    """
    File layout (convention over configuration):
        memory/
        ├── SOUL.md       identity definition (read-only)
        ├── USER.md       user profile (read-only)
        ├── MEMORY.md     long-term facts (AI writes)
        └── vectors.npz   embedding vectors (auto-managed)
    """

    def __init__(self, base_dir: str = "./memory", max_turns: int = 100) -> None:
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self._turns: list[dict] = []
        self._max_turns = max_turns
        self._embedder = None
        self._embedder_checked = False
        self._vectors: np.ndarray | None = None
        self._load_vectors()

    # -- Embedding model (lazy load) -------------------------------------------

    def _get_embedder(self):
        if self._embedder is not None:
            return self._embedder
        if self._embedder_checked:
            return None
        self._embedder_checked = True
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
            log.info("Embedding model loaded: BAAI/bge-small-zh-v1.5")
            return self._embedder
        except ImportError:
            log.info("sentence-transformers not installed, semantic search disabled")
            return None
        except Exception as e:
            log.warning("Failed to load embedding model: %s", e)
            return None

    def _load_vectors(self) -> None:
        vec_path = self.base / "vectors.npz"
        if vec_path.exists():
            try:
                data = np.load(vec_path)
                self._vectors = data["vectors"]
                log.info("Loaded %d memory vectors", self._vectors.shape[0])
            except Exception:
                log.warning("Failed to load vectors.npz, will rebuild on demand")
                self._vectors = None

    # -- Read identity & memory -----------------------------------------------

    def soul(self) -> str:
        return self._read_md("SOUL.md")

    def user_profile(self) -> str:
        return self._read_md("USER.md")

    def long_term_memory(self) -> str:
        return self._read_md("MEMORY.md")

    # -- Short-term memory (conversation turns) --------------------------------

    def append_turn(self, role: str, text: str) -> None:
        self._turns.append({"role": role, "text": text, "ts": time.time()})
        if len(self._turns) > self._max_turns:
            self._turns = self._turns[-self._max_turns // 2:]

    def add_user(self, text: str) -> None:
        """Backward-compatible alias used by the conversation runtime."""
        self.append_turn("user", text)

    def add_assistant(self, text: str) -> None:
        """Backward-compatible alias used by the conversation runtime."""
        self.append_turn("assistant", text)

    def recent_turns(self, n: int = 20) -> list[dict]:
        return self._turns[-n:]

    # -- Long-term memory write ------------------------------------------------

    def save_fact(self, fact: str) -> None:
        mem_path = self.base / "MEMORY.md"
        header = ""
        if not mem_path.exists():
            header = "# 长期记忆\n"
        with open(mem_path, "a", encoding="utf-8") as f:
            if header:
                f.write(header)
            f.write(f"\n- [{time.strftime('%Y-%m-%d %H:%M')}] {fact}")

        embedder = self._get_embedder()
        if embedder is not None:
            try:
                vec = embedder.encode([fact], normalize_embeddings=True)[0]
                if self._vectors is None:
                    self._vectors = vec.reshape(1, -1)
                else:
                    self._vectors = np.vstack([self._vectors, vec])
                np.savez(self.base / "vectors.npz", vectors=self._vectors)
            except Exception:
                log.warning("Failed to embed and save vector for fact")

    # -- Semantic search -------------------------------------------------------

    def _parse_facts(self) -> list[str]:
        """Parse MEMORY.md into individual fact strings."""
        content = self._read_md("MEMORY.md")
        if not content:
            return []
        facts = []
        for line in content.split("\n"):
            line = line.strip()
            if not line.startswith("- "):
                continue
            entry = line[2:].strip()
            if entry.startswith("[") and "]" in entry:
                bracket_end = entry.index("]")
                entry = entry[bracket_end + 1:].strip()
            if entry:
                facts.append(entry)
        return facts

    def parse_facts(self) -> list[str]:
        """Public accessor for parsed long-term facts."""
        return self._parse_facts()

    def search_memory(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Semantic search over stored memories. Returns (fact, similarity) pairs."""
        facts = self._parse_facts()
        if not facts:
            return []

        embedder = self._get_embedder()
        if embedder is None or self._vectors is None:
            return self._keyword_search(query, facts, top_k)

        if self._vectors.shape[0] != len(facts):
            self._rebuild_vectors(facts)
            if self._vectors is None:
                return self._keyword_search(query, facts, top_k)

        try:
            q_vec = embedder.encode([query], normalize_embeddings=True)[0]
            sims = self._vectors @ q_vec
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [
                (facts[i], float(sims[i]))
                for i in top_indices
                if i < len(facts) and sims[i] > 0.1
            ]
        except Exception:
            log.warning("Semantic search failed, falling back to keyword search")
            return self._keyword_search(query, facts, top_k)

    def _keyword_search(self, query: str, facts: list[str], top_k: int) -> list[tuple[str, float]]:
        q_lower = query.lower()
        scored = []
        for fact in facts:
            f_lower = fact.lower()
            score = sum(1 for word in q_lower.split() if word in f_lower)
            if score > 0:
                scored.append((fact, score / max(len(q_lower.split()), 1)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _rebuild_vectors(self, facts: list[str]) -> None:
        embedder = self._get_embedder()
        if embedder is None or not facts:
            self._vectors = None
            return
        try:
            log.info("Rebuilding vectors for %d facts...", len(facts))
            self._vectors = embedder.encode(facts, normalize_embeddings=True)
            np.savez(self.base / "vectors.npz", vectors=self._vectors)
            log.info("Vectors rebuilt successfully")
        except Exception:
            log.warning("Failed to rebuild vectors")
            self._vectors = None

    # -- Lifecycle -------------------------------------------------------------

    def flush(self) -> None:
        """Persist any in-memory state that hasn't been written yet."""
        if self._vectors is not None:
            try:
                np.savez(self.base / "vectors.npz", vectors=self._vectors)
            except Exception:
                log.warning("Failed to flush vectors to disk")

    # -- Helpers ---------------------------------------------------------------

    def _read_md(self, filename: str) -> str:
        path = self.base / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""
