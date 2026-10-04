"""
System / evaluation router (mounted at /api).

Provides the data behind the Evaluation & Model page, the Limitations page,
health checks, and the one-click sample investigation.
"""

from __future__ import annotations

import importlib
import platform
import sys

from fastapi import APIRouter, HTTPException

from backend.config import MODE, DISCLAIMER, EMBEDDING_DIM
from backend.models import registry
from backend.database import db

router = APIRouter()


def _module_available(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


@router.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "python": sys.version.split()[0],
            "platform": platform.platform()}


@router.get("/system/capabilities")
def capabilities():
    return {
        "mode": MODE,
        "embedding_dim": EMBEDDING_DIM,
        "disclaimer": DISCLAIMER,
        "libraries": {
            "numpy": _module_available("numpy"),
            "Pillow": _module_available("PIL"),
            "OpenCV": _module_available("cv2"),
            "soundfile": _module_available("soundfile"),
            "librosa": _module_available("librosa"),
            "PyMuPDF": _module_available("fitz"),
            "scikit-learn": _module_available("sklearn"),
            "torch": _module_available("torch"),
            "transformers": _module_available("transformers"),
        },
        "models_present": registry.available_models(),
        "video_decoder": "OpenCV (bundled ffmpeg)" if _module_available("cv2") else "unavailable",
    }


# ---------------------------------------------------------------------------
# Evaluation page data.
# NOTE: we report "not_evaluated" for anything we have not actually measured.
# ---------------------------------------------------------------------------
@router.get("/evaluation")
def evaluation():
    models_present = registry.available_models()
    any_model = any(models_present.values())

    # Load a REAL evaluation report if one has been produced. Never fabricate.
    import json as _json
    from backend.config import MODELS_DIR
    report_path = MODELS_DIR / "evaluation_report.json"
    report = None
    if report_path.exists():
        try:
            report = _json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:
            report = None

    if report and report.get("status") == "ok":
        tm = report.get("test_metrics", {})
        tr = report.get("train_metrics", {}) or {}
        fmt = (lambda m: f"acc {m.get('accuracy')} · prec {m.get('precision')} · "
                         f"rec {m.get('recall')} · f1 {m.get('f1')} · auc {m.get('roc_auc')}")
        metrics = {
            "individual_only": fmt(tr) if tr.get("accuracy") is not None else "Not evaluated yet",
            "multimodal": "Not evaluated yet",
            "multimodal_plus_cross_modal": "Not evaluated yet",
        }
        generalization = {
            "known_manipulations": fmt(tr) if tr.get("accuracy") is not None else "Not evaluated yet",
            "unseen_manipulations": fmt(tm),
            "different_compression": "Not evaluated yet",
            "different_conditions": "Not evaluated yet",
        }
        note = (f"Image detector evaluated on {report.get('dataset')}: "
                f"{report.get('n_test')} held-out images. Trained in MODEL mode "
                f"({report.get('mode_used')}). Multimodal / cross-modal ablations "
                "have not been run yet.")
    else:
        metrics = {
            "individual_only": "Not evaluated yet",
            "multimodal": "Not evaluated yet",
            "multimodal_plus_cross_modal": "Not evaluated yet",
        }
        generalization = {
            "known_manipulations": "Not evaluated yet",
            "unseen_manipulations": "Not evaluated yet",
            "different_compression": "Not evaluated yet",
            "different_conditions": "Not evaluated yet",
        }
        note = ("No evaluation has been run in this workspace. Run "
                "`python -m training.evaluate_image --cifake` after training to "
                "populate real numbers. Results are never fabricated.")

    return {
        "mode": MODE,
        "models_present": models_present,
        "metrics": metrics,
        "generalization": generalization,
        "report": report,
        "note": note,
        "fusion_input": "image_emb(128)+video_emb(128)+audio_emb(128)+"
                        "text_emb(128)+availability_mask(4) = 516-d",
        "modality_dropout": "Supported (mask zeros a modality; see training/train_fusion.py)",
        "any_model_trained": any_model,
    }


@router.get("/limits")
def limits():
    return {
        "disclaimer": DISCLAIMER,
        "genuinely_ai": [
            "STFT / mel-style spectral analysis (numpy FFT) for audio.",
            "Error Level Analysis + Laplacian/high-frequency statistics for images.",
            "Frame-difference temporal analysis for video (OpenCV).",
            "Deterministic random projection into a shared 128-d embedding space.",
            "Small trained heads + MLP fusion (when MODEL mode artifacts exist).",
        ],
        "heuristic": [
            "Manipulation risk weights and thresholds.",
            "Cross-modal consistency scoring (embedding similarity + domain rules).",
            "Face count via OpenCV Haar cascade (classic, not deep).",
            "Metadata anomaly detection.",
            "Text surface style signals (NOT AI-text detection).",
        ],
        "not_implemented": [
            "Deepfake face-swap detectors (FaceForensics++/Celeb-DF models).",
            "Audio anti-spoofing (ASVspoof models).",
            "OCR for scanned documents.",
            "Reverse image search / provenance / C2PA.",
            "Cryptographic signing.",
            "True manipulation localization (only suspicious indicators).",
            "Transformer / cross-attention fusion (interface left open).",
        ],
        "privacy": (
            "All files stay local. TrustLayer does not send evidence to any "
            "external API by default."
        ),
    }


# ---------------------------------------------------------------------------
# Sample investigation builder
# ---------------------------------------------------------------------------
@router.post("/sample")
def create_sample():
    try:
        from scripts.build_sample import build_sample_investigation
    except Exception as exc:
        raise HTTPException(status_code=500,
                            detail=f"Sample builder unavailable: {exc}")
    try:
        result = build_sample_investigation()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Sample build failed: {exc}")
    # immediately analyse it so the demo lands on a populated dashboard
    from backend.services.pipeline import run_analysis
    analysis = run_analysis(result["investigation_id"])
    return {"ok": True, "investigation": result, "analysis": analysis}
