"""
EVIDENCE GRAPH.

Nodes:
  * one per uploaded file (kind = "evidence")
  * detected entities (names / dates / locations from documents)
  * extracted claims (from documents)

Edges:
  * similarity / consistency / contradiction / timestamp / identity
    between evidence nodes, derived from cross-modal comparisons
  * "mentions" edges from a file to the entities/claims it produced

The output is a plain JSON graph {nodes, edges} that the frontend renders
itself with a tiny custom SVG force-ish layout (no heavy graph library needed).
"""

from __future__ import annotations

from backend.services.cross_modal import _label
from itertools import combinations


def _edge_id(inv_id: str, i: int) -> str:
    return f"{inv_id}-edge-{i}"


SEVERITY_TO_REL = {
    "timestamp_conflict": "timestamp_conflict",
    "time_of_day_conflict": "timestamp_conflict",
    "duration_mismatch": "contradiction",
    "timestamp_missing": "missing_alignment",
    "identity_uncertain": "identity_uncertain",
    "location_conflict": "contradiction",
    "speaker_unverified": "identity_uncertain",
}


def build_graph(investigation_id: str, entries: dict, comparisons: list) -> dict:
    """
    entries: {modality: {"evidence_id", "result", "embedding", "stored_path"}}
    comparisons: output of cross_modal.compare_all
    """
    nodes = []
    edges = []
    index = {m: e for m, e in entries.items() if e.get("result") is not None}

    # ---- evidence nodes -------------------------------------------------
    for m, entry in index.items():
        res = entry["result"]
        nodes.append({
            "id": entry["evidence_id"],
            "label": res.filename,
            "kind": "evidence",
            "modality": m,
            "label_text": res.label,
            "score": round(float(res.score), 4),
        })

    # ---- entity / claim nodes from documents ----------------------------
    entity_nodes: dict = {}
    for m, entry in index.items():
        if m != "document":
            continue
        res = entry["result"]
        ents = (res.detail or {}).get("entities", {})
        for group, kind in (("names", "entity_name"), ("dates", "entity_date"),
                            ("locations", "entity_location")):
            for value in ents.get(group, [])[:6]:
                key = f"{kind}:{value}"
                if key not in entity_nodes:
                    entity_nodes[key] = {
                        "id": key, "label": str(value)[:40], "kind": kind,
                    }
                    nodes.append(entity_nodes[key])
                edges.append({
                    "id": _edge_id(investigation_id, len(edges)),
                    "source": entry["evidence_id"],
                    "target": key,
                    "relationship": "mentions",
                    "score": 1.0,
                    "detail": {"group": group},
                })
        for claim in ents.get("claims", [])[:4]:
            key = f"claim:{claim[:50]}"
            if key not in entity_nodes:
                entity_nodes[key] = {
                    "id": key, "label": claim[:50], "kind": "claim",
                }
                nodes.append(entity_nodes[key])
            edges.append({
                "id": _edge_id(investigation_id, len(edges)),
                "source": entry["evidence_id"],
                "target": key,
                "relationship": "asserts_claim",
                "score": 1.0,
                "detail": {},
            })

    # ---- structured-claim nodes from every modality ----------------------
    for m, entry in index.items():
        res = entry["result"]
        for c in (res.detail or {}).get("claims", []) or []:
            if c.get("confidence", 0) < 0.5:
                continue
            key = f"claim:{c['type']}:{str(c['value'])[:40]}"
            if key not in entity_nodes:
                entity_nodes[key] = {
                    "id": key, "label": f"{c['type']}: {str(c['value'])[:30]}",
                    "kind": "claim",
                }
                nodes.append(entity_nodes[key])
            edges.append({
                "id": _edge_id(investigation_id, len(edges)),
                "source": entry["evidence_id"],
                "target": key,
                "relationship": "asserts_claim",
                "score": round(float(c.get("confidence", 0.7)), 4),
                "detail": {"type": c["type"], "method": c.get("method")},
            })

    # ---- cross-modal edges ----------------------------------------------
    for comp in comparisons:
        parts = comp["pair"].split("_")
        if len(parts) != 2:
            continue
        a, b = parts
        if a not in index or b not in index:
            continue
        ea, eb = index[a], index[b]
        consistency = comp["consistency"]
        # relationship follows the human-readable label band, not just raw
        # thresholds on the score, so MIXED results are no longer silently
        # labelled "consistency".
        if comp["label"] == "CONSISTENT":
            rel = "similarity"
        elif comp["label"] == "INCONSISTENT":
            rel = "contradiction"
        else:
            rel = "partial_consistency"
        edges.append({
            "id": _edge_id(investigation_id, len(edges)),
            "source": ea["evidence_id"],
            "target": eb["evidence_id"],
            "relationship": rel,
            "score": round(consistency, 4),
            "detail": {
                "pair": comp["pair"],
                "label": comp["label"],
                "breakdown": comp.get("breakdown", {}),
                "method": comp.get("method"),
            },
        })
        # extra edges for each concrete conflict (deduplicated by type)
        seen = set()
        for conflict in sorted(comp["conflicts"],
                               key=lambda c: {"high": 0, "medium": 1, "low": 2}.get(c.get("severity"), 3)):
            ctype = conflict["type"]
            if ctype in seen:
                continue
            seen.add(ctype)
            severity_w = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(conflict.get("severity"), 0.3)
            edges.append({
                "id": _edge_id(investigation_id, len(edges)),
                "source": ea["evidence_id"],
                "target": eb["evidence_id"],
                "relationship": SEVERITY_TO_REL.get(ctype, "conflict"),
                # a conflict is as strong as the worst of (pair inconsistency,
                # severity weight) — never the meaningless 0.0 it had before.
                "score": round(max(1.0 - consistency, severity_w), 4),
                "detail": {"description": conflict["description"],
                           "severity": conflict["severity"]},
            })

    # ---- event-time edges -------------------------------------------------
    # Two files whose embedded event times roughly agree are likely from the
    # same event; files days apart are not. This makes the graph reflect an
    # explicit, checkable claim rather than a generic similarity score.
    for a, b in combinations(sorted(index), 2):
        pa = {"result": index[a]["result"]}
        pb = {"result": index[b]["result"]}
        ta = _safe_event_time(pa)
        tb = _safe_event_time(pb)
        if ta is None or tb is None:
            continue
        days = abs((ta - tb).total_seconds()) / 86400.0
        # within ~2 hours = plausibly the same event; same calendar day but
        # several hours apart is already a notable discrepancy
        rel = "same_event_time" if days <= (2.0 / 24.0) else "different_event_time"
        edges.append({
            "id": _edge_id(investigation_id, len(edges)),
            "source": index[a]["evidence_id"],
            "target": index[b]["evidence_id"],
            "relationship": rel,
            "score": round(1.0 / (1.0 + days), 4),
            "detail": {"days_apart": round(days, 2),
                       "time_a": ta.isoformat(), "time_b": tb.isoformat()},
        })

    # ---- cross-modality entity corroboration -------------------------------
    # An entity (name/place/date) mentioned by more than one file corroborates
    # that file-pair. We add one "corroborates" edge per matching value.
    mention: dict = {}   # normalised value -> list of (modality, evidence_id)
    for m, entry in index.items():
        res = entry["result"]
        detail = res.detail or {}
        ents = detail.get("entities") or detail.get("image_claims") or {}
        values = []
        for group in ("names", "places", "locations", "speakers", "dates"):
            values.extend(ents.get(group, []) or [])
        for d in ents.get("datetimes", []) or []:
            values.append(d)
        # structured claims (OCR / extraction) from any modality
        for c in detail.get("claims") or []:
            if c.get("type") in ("location", "person", "url", "timestamp", "time_of_day"):
                values.append(str(c.get("value", "")))
        for v in values:
            if isinstance(v, str) and len(v.strip()) >= 3:
                mention.setdefault(v.strip().lower(), []).append((m, entry["evidence_id"]))
    for value, srcs in mention.items():
        mods = {m for m, _ in srcs}
        if len(mods) < 2:
            continue
        seen_pairs = set()
        for (m1, e1), (m2, e2) in combinations(srcs, 2):
            if m1 == m2 or e1 == e2 or (e1, e2) in seen_pairs or (e2, e1) in seen_pairs:
                continue
            seen_pairs.add((e1, e2))
            edges.append({
                "id": _edge_id(investigation_id, len(edges)),
                "source": e1,
                "target": e2,
                "relationship": "corroborates",
                "score": round(1.0 - 0.2 * (len(seen_pairs) - 1), 4),
                "detail": {"shared_value": value},
            })

    return {"nodes": nodes, "edges": edges}


def _safe_event_time(entry: dict):
    try:
        from backend.services.cross_modal import _modality_timestamp
        return _modality_timestamp(entry)
    except Exception:
        return None
