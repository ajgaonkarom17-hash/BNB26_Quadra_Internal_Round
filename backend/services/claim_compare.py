"""
Cross-modal claim comparison.

Takes the structured claims extracted per modality (see claims.py) and
compares them across modalities. Missing values are NEVER conflicts — they
produce "unavailable" notes instead. Only two modalities BOTH asserting
incompatible values is a conflict.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from backend.services.claims import (
    normalize_datetime, normalize_location, normalize_name,
    normalize_time_of_day,
)


def _claims_of(claims: List[dict], type: str) -> List[dict]:
    return [c for c in claims if c.get("type") == type]


def _minute_value(norm_dt: str) -> Optional[int]:
    try:
        hh, mm = norm_dt.split("T")[1].split(":")
        return int(hh) * 60 + int(mm)
    except Exception:
        return None


def _times_agree(a: str, b: str) -> bool:
    ma = _minute_value(a) if "T" in str(a) else None
    if ma is None:
        ma = (int(a[:2]) * 60 + int(a[3:])) if len(str(a)) >= 5 else None
    mb = _minute_value(b) if "T" in str(b) else None
    if mb is None:
        mb = (int(b[:2]) * 60 + int(b[3:])) if len(str(b)) >= 5 else None
    if ma is None or mb is None:
        return str(a) == str(b)
    return abs(ma - mb) <= 30  # half-hour tolerance


def compare_claims(per_modality: Dict[str, List[dict]]) -> List[dict]:
    """
    per_modality: {modality: [claim, ...]}
    Returns a list of finding dicts:
      {kind: agree|conflict|unavailable, claim_type, value_a, value_b,
       modality_a, modality_b, description, severity}
    """
    findings: List[dict] = []
    mods = [m for m, c in per_modality.items() if c]

    def first(mod, typ):
        cs = _claims_of(per_modality[mod], typ)
        return cs[0] if cs else None

    # timestamps (full datetimes)
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            a, b = mods[i], mods[j]
            ca, cb = first(a, "timestamp"), first(b, "timestamp")
            if ca and cb:
                if ca["value"][:10] == cb["value"][:10] and _times_agree(ca["value"], cb["value"]):
                    findings.append({"kind": "agree", "claim_type": "timestamp",
                                     "value_a": ca["value"], "value_b": cb["value"],
                                     "modality_a": a, "modality_b": b, "severity": "info",
                                     "description": f"{a} and {b} timestamps agree ({ca['value']})."})
                else:
                    findings.append({"kind": "conflict", "claim_type": "timestamp",
                                     "value_a": ca["value"], "value_b": cb["value"],
                                     "modality_a": a, "modality_b": b, "severity": "high",
                                     "description": (f"{a} claims {ca['value']} but {b} claims "
                                                     f"{cb['value']}.")})
            elif ca or cb:
                owner = a if ca else b
                other = b if ca else a
                findings.append({"kind": "unavailable", "claim_type": "timestamp",
                                 "value_a": (ca or cb)["value"], "value_b": None,
                                 "modality_a": owner, "modality_b": other, "severity": "low",
                                 "description": f"Timestamp available for {owner} but unavailable for {other}; cannot verify alignment."})

    # time-of-day claims (only when no full timestamp comparison fired)
    if not any(f["claim_type"] == "timestamp" for f in findings):
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                a, b = mods[i], mods[j]
                ca, cb = first(a, "time_of_day"), first(b, "time_of_day")
                if ca and cb:
                    if _times_agree(ca["value"], cb["value"]):
                        findings.append({"kind": "agree", "claim_type": "time_of_day",
                                         "value_a": ca["value"], "value_b": cb["value"],
                                         "modality_a": a, "modality_b": b, "severity": "info",
                                         "description": f"{a} and {b} times of day agree (~{ca['value']})."})
                    else:
                        findings.append({"kind": "conflict", "claim_type": "time_of_day",
                                         "value_a": ca["value"], "value_b": cb["value"],
                                         "modality_a": a, "modality_b": b, "severity": "high",
                                         "description": f"{a} claims ~{ca['value']} but {b} claims ~{cb['value']}."})

    # locations
    seen_loc = set()
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            a, b = mods[i], mods[j]
            la = {c["value"] for c in _claims_of(per_modality[a], "location")}
            lb = {c["value"] for c in _claims_of(per_modality[b], "location")}
            if la and lb and (a, b) not in seen_loc:
                seen_loc.add((a, b))
                if la & lb:
                    findings.append({"kind": "agree", "claim_type": "location",
                                     "value_a": sorted(la & lb)[0], "value_b": sorted(la & lb)[0],
                                     "modality_a": a, "modality_b": b, "severity": "info",
                                     "description": f"{a} and {b} both reference location '{sorted(la & lb)[0]}'."})
                else:
                    findings.append({"kind": "conflict", "claim_type": "location",
                                     "value_a": sorted(la)[0], "value_b": sorted(lb)[0],
                                     "modality_a": a, "modality_b": b, "severity": "medium",
                                     "description": f"{a} references '{sorted(la)[0]}' but {b} references '{sorted(lb)[0]}'."})

    # people
    for i in range(len(mods)):
        for j in range(i + 1, len(mods)):
            a, b = mods[i], mods[j]
            pa = {c["value"] for c in _claims_of(per_modality[a], "person")}
            pb = {c["value"] for c in _claims_of(per_modality[b], "person")}
            if pa and pb:
                if pa & pb:
                    v = sorted(pa & pb)[0]
                    findings.append({"kind": "agree", "claim_type": "person",
                                     "value_a": v, "value_b": v,
                                     "modality_a": a, "modality_b": b, "severity": "info",
                                     "description": f"{a} and {b} both mention '{v}'."})
    return findings
