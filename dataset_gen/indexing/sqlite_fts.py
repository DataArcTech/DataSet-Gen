import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    snippet: str


def _fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS __fts5_test USING fts5(content)")
        conn.execute("DROP TABLE __fts5_test")
        return True
    except sqlite3.OperationalError:
        return False


class SqliteFtsIndex:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.has_fts5 = _fts5_available(self.conn)
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, section_id TEXT, section_title TEXT, content TEXT)"
        )
        # Back-compat: older DBs may miss section_id.
        try:
            cols = [r[1] for r in self.conn.execute("PRAGMA table_info(chunks)").fetchall()]
            if "section_id" not in cols:
                self.conn.execute("ALTER TABLE chunks ADD COLUMN section_id TEXT")
        except sqlite3.OperationalError:
            pass

        if self.has_fts5:
            # If schema changed, drop + recreate the virtual table (safe; we rebuild anyway).
            needs_recreate = False
            try:
                sql_row = self.conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
                ).fetchone()
                sql = (sql_row["sql"] if sql_row else "") or ""
                if "section_id" not in sql:
                    needs_recreate = True
            except sqlite3.OperationalError:
                needs_recreate = True
            if needs_recreate:
                try:
                    self.conn.execute("DROP TABLE IF EXISTS chunks_fts")
                except sqlite3.OperationalError:
                    pass
            self.conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id, section_id, section_title, content)"
            )
        self.conn.commit()

    def rebuild(self, chunks: List[Dict[str, Any]]) -> None:
        self.conn.execute("DELETE FROM chunks")
        if self.has_fts5:
            self.conn.execute("DELETE FROM chunks_fts")
        for c in chunks:
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks(chunk_id, section_id, section_title, content) VALUES (?,?,?,?)",
                (c["chunk_id"], c.get("section_id") or "", c.get("section_title") or "", c.get("text") or ""),
            )
            if self.has_fts5:
                self.conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, section_id, section_title, content) VALUES (?,?,?,?)",
                    (c["chunk_id"], c.get("section_id") or "", c.get("section_title") or "", c.get("text") or ""),
                )
        self.conn.commit()

    def search(self, keywords: List[str], limit: int = 20) -> List[SearchHit]:
        terms = [str(k).strip() for k in (keywords or []) if str(k).strip()]
        if not terms:
            return []

        if self.has_fts5:
            # FTS5 query language is not plain text (it has operators like AND/OR/NEAR, column filters, etc.).
            # We quote each term to avoid syntax errors from LLM-generated tokens (e.g., punctuation like ':' or '-').
            quoted = ['"' + t.replace('"', '""') + '"' for t in terms[:12]]
            query = " OR ".join(quoted)
            rows = self.conn.execute(
                "SELECT chunk_id, section_id, bm25(chunks_fts) AS score, snippet(chunks_fts, 3, '[', ']', '…', 12) AS snippet "
                "FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (query, int(limit)),
            ).fetchall()
            return [
                SearchHit(chunk_id=r["chunk_id"], score=float(r["score"]), snippet=str(r["snippet"])) for r in rows
            ]

        # Fallback: naive LIKE scan with a pseudo-score.
        query = " ".join(terms)
        like = "%" + query + "%"
        rows = self.conn.execute(
            "SELECT chunk_id, content FROM chunks WHERE content LIKE ? LIMIT ?",
            (like, int(limit)),
        ).fetchall()
        hits: List[SearchHit] = []
        for r in rows:
            content = str(r["content"])
            pos = content.lower().find(query.lower())
            if pos < 0:
                pos = 0
            snippet = content[max(0, pos - 120) : pos + 240]
            hits.append(SearchHit(chunk_id=str(r["chunk_id"]), score=0.0, snippet=snippet))
        return hits
