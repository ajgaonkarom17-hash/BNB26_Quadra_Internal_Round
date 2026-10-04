"""Verify the exact API calls the frontend makes, and the JSON shape it reads."""
import json
import urllib.request

import os
BASE = "http://127.0.0.1:" + os.environ.get("TL_PORT", "8000")


def get(path):
    return json.loads(urllib.request.urlopen(BASE + path).read())


def post(path, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req).read())


# 1. Frontend boot calls
caps = get("/api/system/capabilities")
assert "mode" in caps and "libraries" in caps, caps
print("capabilities.mode =", caps["mode"])

# 2. Sample investigation (Home button)
sample = post("/api/sample")
inv_id = sample["investigation"]["investigation_id"]
print("sample inv:", inv_id, "files:", sample["investigation"]["files"])

# 3. History calls
lst = get("/api/investigations")
assert isinstance(lst["investigations"], list)
print("history count:", len(lst["investigations"]))
row = lst["investigations"][0]
for key in ("id", "name", "assessment", "confidence", "coverage", "status", "created_at"):
    assert key in row, f"missing {key} in history row"

# 4. Results page contract
r = get(f"/api/investigations/{inv_id}/results")
for key in ("investigation", "coverage", "fusion", "cross_modal", "uncertainty",
            "explanation", "individual", "video_timeline"):
    assert key in r, f"results missing {key}"
assert set(["assessment", "confidence_label", "consistency", "engine",
            "available_modalities", "breakdown"]).issubset(r["fusion"].keys())
assert set(["final_assessment", "downgraded", "uncertainty", "coverage",
            "missing", "conflicts", "low_quality"]).issubset(r["uncertainty"].keys())
assert set(["why_flagged", "supporting", "conflicting", "missing",
            "limitations", "what_changed", "top_reasons"]).issubset(r["explanation"].keys())
assert "comparisons" in r["cross_modal"]
assert r["video_timeline"] is not None and "suspicious" in r["video_timeline"]
print("results assessment:", r["uncertainty"]["final_assessment"],
      "| confidence:", r["fusion"]["confidence_label"],
      "| coverage:", r["coverage"])
print("cross pairs:", [c["pair"] for c in r["cross_modal"]["comparisons"]])
print("what_changed aspects:", [w["aspect"] for w in r["explanation"]["what_changed"]])
print("video suspicious markers:", len(r["video_timeline"]["suspicious"]))

# 5. Graph page contract
g = get(f"/api/investigations/{inv_id}/graph")
for key in ("graph", "stored_relationships"):
    assert key in g
assert "nodes" in g["graph"] and "edges" in g["graph"]
node_ids = {n["id"] for n in g["graph"]["nodes"]}
for e in g["graph"]["edges"]:
    assert e["source"] in node_ids and e["target"] in node_ids, f"dangling edge {e}"
print("graph nodes:", len(g["graph"]["nodes"]), "edges:", len(g["graph"]["edges"]))
print("node kinds:", sorted({n["kind"] for n in g["graph"]["nodes"]}))
print("edge rels:", sorted({e["relationship"] for e in g["graph"]["edges"]}))

# 6. evidence endpoint
ev = get(f"/api/investigations/{inv_id}/evidence")
assert len(ev["evidence"]) == 4
print("evidence modalities:", [e["modality"] for e in ev["evidence"]])

print("\nFRONTEND CONTRACT OK")
