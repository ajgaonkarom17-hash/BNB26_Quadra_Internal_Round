"""Quick end-to-end smoke test for the TrustLayer API (run: python scripts/smoke_test.py)."""
from fastapi.testclient import TestClient
from backend.main import app

c = TestClient(app)

print("health:", c.get("/api/health").json())
print("capabilities libraries:", c.get("/api/system/capabilities").json()["libraries"])

inv = c.post("/api/investigations", json={"name": "API Smoke Test"}).json()
print("created:", inv["id"])

up = c.post(
    f"/api/investigations/{inv['id']}/upload",
    files={"file": ("note.txt", b"Report confirmed 2020-01-01 at Riverside Park.", "text/plain")},
).json()
print("upload:", up)

r = c.post(f"/api/investigations/{inv['id']}/analyze").json()
print("assessment:", r["uncertainty"]["final_assessment"], "| coverage:", r["coverage"])
print("steps:", len(r["steps"]), "| cross pairs:", len(r["cross_modal"]["comparisons"]))

graph = c.get(f"/api/investigations/{inv['id']}/graph").json()["graph"]
print("graph nodes/edges:", len(graph["nodes"]), len(graph["edges"]))

results = c.get(f"/api/investigations/{inv['id']}/results").json()
print("results assessment:", results["uncertainty"]["final_assessment"])

print("list:", len(c.get("/api/investigations").json()["investigations"]), "investigations")
print("evaluation:", c.get("/api/evaluation").json()["metrics"])
print("limits ai:", len(c.get("/api/limits").json()["genuinely_ai"]), "items")
print("OK")
