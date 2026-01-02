from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    if vec.size == 0:
        raise ValueError("empty feature vector")
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


class SidQueryStore:
    def __init__(self, db_path: Path, index_path: Path) -> None:
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._index: Optional[faiss.IndexIDMap2] = None
        self._dim: Optional[int] = None

        self._init_db()
        self._load_index()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id INTEGER PRIMARY KEY,
                    metadata  TEXT NOT NULL,
                    vector    BLOB NOT NULL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def _get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )

    def _count_entities(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM entities").fetchone()
        return int(row["cnt"]) if row else 0

    def _create_index(self, dim: int) -> faiss.IndexIDMap2:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def _persist_index(self) -> None:
        if self._index is None:
            return
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        faiss.write_index(self._index, str(tmp_path))
        tmp_path.replace(self.index_path)

    def _load_index(self) -> None:
        with self._lock:
            meta_dim = self._get_meta("vector_dim")
            self._dim = int(meta_dim) if meta_dim else None

            if self.index_path.exists():
                index = faiss.read_index(str(self.index_path))
                index_dim = int(index.d)
                if self._dim is None:
                    self._dim = index_dim
                    self._set_meta("vector_dim", str(index_dim))
                if self._dim == index_dim:
                    if isinstance(index, faiss.IndexIDMap):
                        self._index = index
                    else:
                        self._index = faiss.IndexIDMap2(index)

            if self._index is None or self._index.ntotal != self._count_entities():
                self._rebuild_index_from_db()

    def _rebuild_index_from_db(self) -> None:
        rows = self._conn.execute(
            "SELECT entity_id, vector FROM entities ORDER BY entity_id"
        ).fetchall()
        if not rows:
            self._index = None
            return

        if self._dim is None:
            first_vec = rows[0]["vector"]
            self._dim = len(first_vec) // 4
            self._set_meta("vector_dim", str(self._dim))

        index = self._create_index(self._dim)
        ids: List[int] = []
        vectors: List[np.ndarray] = []
        for row in rows:
            vec = np.frombuffer(row["vector"], dtype=np.float32)
            if vec.size != self._dim:
                raise ValueError(f"vector dim mismatch for entity {row['entity_id']}")
            vectors.append(vec)
            ids.append(int(row["entity_id"]))

        index.add_with_ids(np.vstack(vectors), np.asarray(ids, dtype=np.int64))
        self._index = index
        self._persist_index()

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        vec = _normalize_vector(vector)
        if self._dim is None:
            self._dim = int(vec.size)
            with self._conn:
                self._set_meta("vector_dim", str(self._dim))
        if vec.size != self._dim:
            raise ValueError(f"vector dim {vec.size} != expected {self._dim}")
        return vec

    def register(self, vector: np.ndarray, metadata: Dict[str, str], entity_id: Optional[int]) -> int:
        vec = self._prepare_vector(vector)
        metadata_json = json.dumps(metadata, ensure_ascii=True)

        with self._lock:
            if entity_id is not None:
                exists = self._conn.execute(
                    "SELECT 1 FROM entities WHERE entity_id = ?", (entity_id,)
                ).fetchone()
                if exists:
                    raise KeyError(f"entity_id {entity_id} already exists")

            with self._conn:
                if entity_id is None:
                    cur = self._conn.execute(
                        "INSERT INTO entities (metadata, vector) VALUES (?, ?)",
                        (metadata_json, vec.tobytes()),
                    )
                    entity_id = int(cur.lastrowid)
                else:
                    self._conn.execute(
                        "INSERT INTO entities (entity_id, metadata, vector) VALUES (?, ?, ?)",
                        (entity_id, metadata_json, vec.tobytes()),
                    )

            if self._index is None:
                self._index = self._create_index(self._dim)

            self._index.add_with_ids(vec.reshape(1, -1), np.asarray([entity_id], dtype=np.int64))
            self._persist_index()
            return entity_id

    def update(self, entity_id: int, vector: np.ndarray, metadata: Dict[str, str], overwrite: bool) -> None:
        vec = self._prepare_vector(vector)

        with self._lock:
            row = self._conn.execute(
                "SELECT metadata, vector FROM entities WHERE entity_id = ?", (entity_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"entity_id {entity_id} not found")

            existing_meta = json.loads(row["metadata"]) if row["metadata"] else {}
            if metadata:
                existing_meta.update(metadata)
            metadata_json = json.dumps(existing_meta, ensure_ascii=True)

            if not overwrite:
                old_vec = np.frombuffer(row["vector"], dtype=np.float32)
                vec = _normalize_vector(old_vec + vec)

            with self._conn:
                self._conn.execute(
                    "UPDATE entities SET metadata = ?, vector = ? WHERE entity_id = ?",
                    (metadata_json, vec.tobytes(), entity_id),
                )

            if self._index is None:
                self._index = self._create_index(self._dim)
            self._index.remove_ids(np.asarray([entity_id], dtype=np.int64))
            self._index.add_with_ids(vec.reshape(1, -1), np.asarray([entity_id], dtype=np.int64))
            self._persist_index()

    def delete(self, entity_id: int) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM entities WHERE entity_id = ?", (entity_id,)
            ).fetchone()
            if not row:
                return False

            with self._conn:
                self._conn.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))

            if self._index is not None:
                self._index.remove_ids(np.asarray([entity_id], dtype=np.int64))
                self._persist_index()
            return True

    def get(self, entity_id: int) -> Optional[Tuple[int, Dict[str, str]]]:
        row = self._conn.execute(
            "SELECT metadata FROM entities WHERE entity_id = ?", (entity_id,)
        ).fetchone()
        if not row:
            return None
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        return entity_id, meta

    def search(self, vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if top_k <= 0:
            return []
        vec = self._prepare_vector(vector)
        if self._index is None or self._index.ntotal == 0:
            return []
        scores, ids = self._index.search(vec.reshape(1, -1), top_k)
        result: List[Tuple[int, float]] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            result.append((int(idx), float(score)))
        return result

