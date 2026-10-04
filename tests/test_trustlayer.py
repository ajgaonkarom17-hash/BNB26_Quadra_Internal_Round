"""TrustLayer prototype tests (behavior, not hard-coded scores)."""

import os

import numpy as np
import pytest

MEDIA = r"C:\Users\reece\Downloads\trustlayer_test_media"
IMG = os.path.join(MEDIA, "01_event_image_5PM.png")
VID = os.path.join(MEDIA, "02_event_video_9AM_conflict.mp4")
AUD = os.path.join(MEDIA, "03_event_audio_5PM.wav")

META = {"evidence_id": "test", "filename": "file"}


# ---- 1-4. analyzers run --------------------------------------------------
def test_image_analysis():
    from backend.services.image_analyzer import analyze
    if os.path.exists(IMG):
        r = analyze(IMG, META)
        assert r.modality == "image" and r.error is None
        assert 0.0 <= r.score <= 1.0


def test_video_analysis():
    from backend.services.video_analyzer import analyze
    if os.path.exists(VID):
        r = analyze(VID, META)
        assert r.modality == "video" and r.features["duration_seconds"] > 0


def test_audio_analysis():
    from backend.services.audio_analyzer import analyze
    if os.path.exists(AUD):
        r = analyze(AUD, META)
        assert r.modality == "audio" and 0.0 <= r.score <= 1.0


def test_document_analysis(tmp_path):
    from backend.services.document_analyzer import analyze
    p = tmp_path / "note.txt"
    p.write_text("Meeting on 2026-10-04 at Riverside Park. Alice said it was confirmed.")
    r = analyze(str(p), META)
    assert r.modality == "document" and r.error is None


# ---- 5-7. missing / unavailable -------------------------------------------
def test_missing_metadata_is_provenance_not_proof():
    from backend.services.image_analyzer import analyze
    if os.path.exists(IMG):
        r = analyze(IMG, META)
        meta = [f for f in r.findings if f.key == "no_exif"]
        assert not meta or meta[0].kind == "missing"
        assert r.label != "SUSPICIOUS" or r.score > 0.5


def test_ocr_unavailable_graceful():
    from backend.services import ocr
    assert isinstance(ocr.ocr_available(), bool)
    assert ocr.image_to_text(object()) == ""  # garbage input never crashes


def test_missing_modality_mask():
    from backend.services.fusion import availability_mask
    m = availability_mask(["image", "audio"])
    assert list(m) == [1, 0, 1, 0]


# ---- 8-13. claim normalization & comparison --------------------------------
def test_timestamp_normalization():
    from backend.services.claims import normalize_time_of_day, normalize_datetime
    assert normalize_time_of_day("5 PM") == "17:00"
    assert normalize_time_of_day("5:00 PM") == "17:00"
    assert normalize_time_of_day("17:00") == "17:00"
    assert normalize_datetime("2026-10-04 17:00") == "2026-10-04T17:00"


def test_location_normalization():
    from backend.services.claims import normalize_location
    assert normalize_location("Riverside Park.") == normalize_location("riverside park")


def test_timestamp_agreement():
    from backend.services.claim_compare import compare_claims
    per = {"image": [{"type": "timestamp", "value": "2026-10-04T17:00"}],
           "document": [{"type": "timestamp", "value": "2026-10-04T17:00"}]}
    f = compare_claims(per)
    assert any(x["kind"] == "agree" for x in f)


def test_timestamp_conflict():
    from backend.services.claim_compare import compare_claims
    per = {"image": [{"type": "timestamp", "value": "2026-10-04T17:00"}],
           "video": [{"type": "timestamp", "value": "2026-10-04T09:00"}]}
    f = compare_claims(per)
    assert any(x["kind"] == "conflict" for x in f)


def test_location_agreement_and_conflict():
    from backend.services.claim_compare import compare_claims
    per = {"image": [{"type": "location", "value": "riverside park"}],
           "document": [{"type": "location", "value": "riverside park"}]}
    assert any(x["kind"] == "agree" for x in compare_claims(per))
    per2 = {"image": [{"type": "location", "value": "riverside park"}],
            "video": [{"type": "location", "value": "downtown station"}]}
    assert any(x["kind"] == "conflict" for x in compare_claims(per2))


def test_missing_value_is_not_conflict():
    from backend.services.claim_compare import compare_claims
    per = {"image": [{"type": "timestamp", "value": "2026-10-04T17:00"}], "video": []}
    assert not any(x["kind"] == "conflict" for x in compare_claims(per))


# ---- 14. evidence coverage & fusion & inconclusive -------------------------
def test_evidence_coverage():
    from backend.services import uncertainty
    assert uncertainty.evidence_coverage(["image", "video"]) == 0.5


def test_fusion_demo():
    from backend.services.fusion import fuse, availability_mask
    emb = {m: np.random.default_rng(0).standard_normal(128).astype(np.float32)
           for m in ("image", "video")}
    out = fuse({"image": 0.2, "video": 0.3}, emb, availability_mask(["image", "video"]),
               {"cross_modal_risk": 0.3, "mean_consistency": 0.7, "min_consistency": 0.6,
                "conflict_count": 0, "pairs_compared": 1}, 0.5)
    assert "risk" in out and 0.0 <= out["risk"] <= 1.0


# ---- 16. evidence graph ------------------------------------------------------
def test_graph_has_claim_edges():
    from backend.services.evidence_graph import build_graph
    class R:
        def __init__(s):
            s.modality="image"; s.filename="i.png"; s.label="INCONCLUSIVE"; s.score=0.2
            s.features={}; s.detail={"claims":[{"type":"timestamp","value":"2026-10-04T17:00","confidence":0.9}]}
    entries={"image":{"evidence_id":"e1","result":R(),"embedding":None,"stored_path":"x"}}
    g = build_graph("inv", entries, [])
    assert any(e["relationship"] == "asserts_claim" for e in g["edges"])


# ---- 17. API endpoints -------------------------------------------------------
def test_api_endpoints():
    from fastapi.testclient import TestClient
    from backend.main import app
    c = TestClient(app)
    inv = c.post("/api/investigations", json={"name": "t"}).json()
    assert c.get("/api/investigations").status_code == 200
    assert c.get(f"/api/investigations/{inv['id']}").status_code == 200
    assert c.get(f"/api/investigations/{inv['id']}/graph").status_code == 200
    assert c.get("/api/health").status_code == 200


# ---- the deliberate media conflict test -------------------------------------
@pytest.mark.skipif(not os.path.exists(IMG), reason="test media not available")
def test_media_timestamp_conflict():
    from backend.services.image_analyzer import analyze as ia
    from backend.services.video_analyzer import analyze as va
    from backend.services.claims import claims_from_text
    import backend.services.ocr as ocr
    i = ia(IMG, {"evidence_id": "i", "filename": "i"})
    v = va(VID, {"evidence_id": "v", "filename": "v"})
    ic = i.detail.get("claims", [])
    vc = v.detail.get("claims", [])
    it = next((c["value"] for c in ic if c["type"] == "timestamp"), None)
    vt = next((c["value"] for c in vc if c["type"] == "timestamp"), None)
    assert it is not None and vt is not None
    assert it[:10] == vt[:10] and it[-5:] != vt[-5:]
    from backend.services.claim_compare import compare_claims
    per = {"image": ic, "video": vc}
    assert any(x["kind"] == "conflict" for x in compare_claims(per))
