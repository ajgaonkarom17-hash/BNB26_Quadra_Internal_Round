"""
SQLite persistence layer for TrustLayer.

Uses the Python standard-library ``sqlite3`` module (no ORM) so beginners can
read every query. Tables follow section 14 of the specification:

    investigations, evidence, analysis_results,
    cross_modal_results, evidence_relationships

The connection is created per-request (``get_connection``) which is safe and
simple for a prototype. ``row_factory`` returns dict-like rows.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from backend.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'created',
    assessment        TEXT,
    confidence        TEXT,
    confidence_score  REAL,
    coverage          REAL,
    risk_score        REAL,
    summary           TEXT
);

CREATE TABLE IF NOT EXISTS evidence (
    id                TEXT PRIMARY KEY,
    investigation_id  TEXT NOT NULL,
    filename          TEXT NOT NULL,
    modality          TEXT NOT NULL,
    content_type      TEXT,
    size_bytes        INTEGER DEFAULT 0,
    stored_path       TEXT,
    uploaded_at       TEXT NOT NULL,
    individual_score  REAL,
    individual_label  TEXT,
    FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analysis_results (
    id                TEXT PRIMARY KEY,
    evidence_id       TEXT NOT NULL,
    investigation_id  TEXT NOT NULL,
    modality          TEXT NOT NULL,
    score             REAL,
    label             TEXT,
    features          TEXT,
    findings          TEXT,
    engine            TEXT,
    detail            TEXT,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (evidence_id) REFERENCES evidence(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cross_modal_results (
    id                TEXT PRIMARY KEY,
    investigation_id  TEXT NOT NULL,
    pair              TEXT NOT NULL,
    consistency       REAL,
    label             TEXT,
    conflicts         TEXT,
    method            TEXT,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_relationships (
    id                TEXT PRIMARY KEY,
    investigation_id  TEXT NOT NULL,
    source            TEXT NOT NULL,
    target            TEXT NOT NULL,
    relationship      TEXT NOT NULL,
    score             REAL,
    detail            TEXT,
    created_at        TEXT NOT NULL,
    FOREIGN KEY (investigation_id) REFERENCES investigations(id) ON DELETE CASCADE
);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    """Context manager: commits on success, rolls back on error, closes."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db() as conn:
        conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def _json(value: Any) -> str:
    return json.dumps(value, default=str)


def _loads(value: Optional[str], default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def row_to_dict(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in row.keys()}


# ---------------------------------------------------------------------------
# Investigations
# ---------------------------------------------------------------------------
def create_investigation(inv_id: str, name: str) -> dict:
    with db() as conn:
        conn.execute(
            "INSERT INTO investigations (id, name, created_at, status) VALUES (?,?,?,?)",
            (inv_id, name, utcnow(), "created"),
        )
    return get_investigation(inv_id)


def get_investigation(inv_id: str) -> Optional[dict]:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM investigations WHERE id = ?", (inv_id,)
        ).fetchone()
    return row_to_dict(row) if row else None


def list_investigations() -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM investigations ORDER BY created_at DESC"
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def update_investigation(inv_id: str, **fields: Any) -> None:
    allowed = {
        "status", "assessment", "confidence", "confidence_score",
        "coverage", "risk_score", "summary",
    }
    sets, values = [], []
    for key, value in fields.items():
        if key in allowed:
            sets.append(f"{key} = ?")
            values.append(value)
    if not sets:
        return
    values.append(inv_id)
    with db() as conn:
        conn.execute(
            f"UPDATE investigations SET {', '.join(sets)} WHERE id = ?", values
        )


def delete_investigation(inv_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM investigations WHERE id = ?", (inv_id,))


def clear_analysis_data(inv_id: str) -> None:
    """Remove derived data before rebuilding an investigation's analysis."""
    with db() as conn:
        conn.execute("DELETE FROM analysis_results WHERE investigation_id = ?", (inv_id,))
        conn.execute("DELETE FROM cross_modal_results WHERE investigation_id = ?", (inv_id,))
        conn.execute("DELETE FROM evidence_relationships WHERE investigation_id = ?", (inv_id,))
        conn.execute(
            "UPDATE evidence SET individual_score = NULL, individual_label = NULL "
            "WHERE investigation_id = ?", (inv_id,),
        )
        conn.execute(
            """UPDATE investigations SET assessment = NULL, confidence = NULL,
               confidence_score = NULL, coverage = NULL, risk_score = NULL,
               summary = NULL WHERE id = ?""",
            (inv_id,),
        )


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------
def add_evidence(item: dict) -> dict:
    with db() as conn:
        conn.execute(
            """INSERT INTO evidence
               (id, investigation_id, filename, modality, content_type,
                size_bytes, stored_path, uploaded_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                item["id"], item["investigation_id"], item["filename"],
                item["modality"], item.get("content_type", ""),
                item.get("size_bytes", 0), item.get("stored_path", ""),
                utcnow(),
            ),
        )
    return item


def update_evidence(evidence_id: str, **fields: Any) -> None:
    allowed = {"individual_score", "individual_label", "stored_path"}
    sets, values = [], []
    for key, value in fields.items():
        if key in allowed:
            sets.append(f"{key} = ?")
            values.append(value)
    if not sets:
        return
    values.append(evidence_id)
    with db() as conn:
        conn.execute(
            f"UPDATE evidence SET {', '.join(sets)} WHERE id = ?", values
        )


def get_evidence(evidence_id: str) -> Optional[dict]:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM evidence WHERE id = ?", (evidence_id,)
        ).fetchone()
    return row_to_dict(row) if row else None


def list_evidence(inv_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM evidence WHERE investigation_id = ? ORDER BY uploaded_at ASC",
            (inv_id,),
        ).fetchall()
    return [row_to_dict(r) for r in rows]


def delete_evidence(evidence_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM evidence WHERE id = ?", (evidence_id,))


# ---------------------------------------------------------------------------
# Analysis results
# ---------------------------------------------------------------------------
def add_analysis(result: dict) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO analysis_results
               (id, evidence_id, investigation_id, modality, score, label,
                features, findings, engine, detail, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                result["id"], result["evidence_id"], result["investigation_id"],
                result["modality"], result.get("score"), result.get("label"),
                _json(result.get("features", {})),
                _json(result.get("findings", [])),
                result.get("engine", "demo-heuristic"),
                _json(result.get("detail", {})),
                utcnow(),
            ),
        )


def get_analysis(evidence_id: str) -> Optional[dict]:
    with db() as conn:
        row = conn.execute(
            "SELECT * FROM analysis_results WHERE evidence_id = ? ORDER BY created_at DESC LIMIT 1",
            (evidence_id,),
        ).fetchone()
    if not row:
        return None
    data = row_to_dict(row)
    data["features"] = _loads(data.get("features"), {})
    data["findings"] = _loads(data.get("findings"), [])
    data["detail"] = _loads(data.get("detail"), {})
    return data


def list_analyses(inv_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_results WHERE investigation_id = ? ORDER BY created_at ASC",
            (inv_id,),
        ).fetchall()
    out = []
    for row in rows:
        data = row_to_dict(row)
        data["features"] = _loads(data.get("features"), {})
        data["findings"] = _loads(data.get("findings"), [])
        data["detail"] = _loads(data.get("detail"), {})
        out.append(data)
    return out


# ---------------------------------------------------------------------------
# Cross-modal results
# ---------------------------------------------------------------------------
def clear_cross_modal(inv_id: str) -> None:
    with db() as conn:
        conn.execute("DELETE FROM cross_modal_results WHERE investigation_id = ?", (inv_id,))
        conn.execute("DELETE FROM evidence_relationships WHERE investigation_id = ?", (inv_id,))


def add_cross_modal(result: dict) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO cross_modal_results
               (id, investigation_id, pair, consistency, label, conflicts, method, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                result["id"], result["investigation_id"], result["pair"],
                result.get("consistency"), result.get("label"),
                _json(result.get("conflicts", [])),
                result.get("method", "heuristic"),
                utcnow(),
            ),
        )


def list_cross_modal(inv_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM cross_modal_results WHERE investigation_id = ? ORDER BY pair ASC",
            (inv_id,),
        ).fetchall()
    out = []
    for row in rows:
        data = row_to_dict(row)
        data["conflicts"] = _loads(data.get("conflicts"), [])
        out.append(data)
    return out


# ---------------------------------------------------------------------------
# Evidence relationships (graph edges)
# ---------------------------------------------------------------------------
def add_relationship(rel: dict) -> None:
    with db() as conn:
        conn.execute(
            """INSERT INTO evidence_relationships
               (id, investigation_id, source, target, relationship, score, detail, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                rel["id"], rel["investigation_id"], rel["source"], rel["target"],
                rel["relationship"], rel.get("score"),
                _json(rel.get("detail", {})), utcnow(),
            ),
        )


def list_relationships(inv_id: str) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM evidence_relationships WHERE investigation_id = ?",
            (inv_id,),
        ).fetchall()
    out = []
    for row in rows:
        data = row_to_dict(row)
        data["detail"] = _loads(data.get("detail"), {})
        out.append(data)
    return out
