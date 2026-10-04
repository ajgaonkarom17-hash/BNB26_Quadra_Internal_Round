"""Verify cross-modal rules: conflict vs consistency, without false positives."""
import numpy as np
from types import SimpleNamespace
from backend.services import cross_modal


def make(modality, features=None, detail=None, score=0.2):
    return {
        "evidence_id": modality,
        "embedding": np.ones(8, dtype=np.float32) * 0.5,
        "stored_path": "",
        "result": SimpleNamespace(
            modality=modality,
            evidence_id=modality,
            filename=f"{modality}.x",
            score=score,
            label="AUTHENTIC",
            features=features or {},
            detail=detail or {},
            findings=[],
            quality="ok",
        ),
    }


print("=== CASE 1: consistent time-of-day (should be consistent) ===")
entries = {
    "image": make("image", {"metadata_time_of_day": "17:00"},
                  {"exif": {"DateTime": "2021:03:05 17:00:00"}, "image_claims": {}}),
    "video": make("video", {"duration_seconds": 6.0, "metadata_time_of_day": "17:00"},
                  {"media_metadata": {"creation_time": "2021-03-05T17:00:00+00:00"}}),
}
comps = cross_modal.compare_all(entries)
for c in comps:
    print(" ", c["pair"], round(c["consistency"], 2), c["label"],
          "conflicts:", [x["type"] for x in c["conflicts"]])
assert not any(cf["type"] == "time_of_day_conflict" for c in comps for cf in c["conflicts"]), \
    "false time-of-day conflict!"

print("\n=== CASE 2: conflicting time-of-day (should flag) ===")
entries = {
    "image": make("image", {"metadata_time_of_day": "17:00"},
                  {"exif": {}, "image_claims": {"times_of_day": ["17:00"]}}),
    "video": make("video", {"duration_seconds": 6.0, "metadata_time_of_day": "09:12"},
                  {"media_metadata": {"creation_time": "2021-03-05T09:12:00+00:00"}}),
}
comps = cross_modal.compare_all(entries)
flag = any(cf["type"] == "time_of_day_conflict" for c in comps for cf in c["conflicts"])
for c in comps:
    print(" ", c["pair"], round(c["consistency"], 2), c["label"],
          "conflicts:", [x["type"] for x in c["conflicts"]])
assert flag, "expected a time-of-day conflict!"

print("\n=== CASE 3: near-matching times (30 min, should NOT flag) ===")
entries = {
    "image": make("image", {"metadata_time_of_day": "17:00"}, {"exif": {}, "image_claims": {}}),
    "video": make("video", {"metadata_time_of_day": "17:20"},
                  {"media_metadata": {}}),
}
comps = cross_modal.compare_all(entries)
flag = any(cf["type"] == "time_of_day_conflict" for c in comps for cf in c["conflicts"])
print("  image_video conflicts:", [x["type"] for c in comps for x in c["conflicts"]])
assert not flag, "should tolerate a 20-minute difference"

print("\nCROSS-MODAL RULE TESTS PASSED")
