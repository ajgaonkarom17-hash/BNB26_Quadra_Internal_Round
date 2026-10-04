"""
TrustLayer central configuration.

Everything that a student might reasonably want to tweak lives here so that
the rest of the code stays clean. Values can be overridden with environment
variables (see .env.example). No external dependency is required to read them.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# backend/config.py -> backend/ -> project root
BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "trustlayer.db"

for _p in (DATA_DIR, SAMPLE_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, UPLOAD_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


# ---------------------------------------------------------------------------
# Runtime mode
# ---------------------------------------------------------------------------
# "demo"  -> deterministic heuristics + fixed fusion (works with zero training)
# "model" -> tries to load files from models/ ; falls back per-modality to demo
MODE = _env("MODE", "demo").lower()

# Embedding dimension for the *common representation*.
EMBEDDING_DIM = int(_env("EMBEDDING_DIM", "128"))

# Reproducibility: all pseudo-random projections/salts are seeded from this.
RANDOM_SEED = int(_env("RANDOM_SEED", "1337"))

# Upload limits (bytes). 200 MB default keeps a laptop demo comfortable.
MAX_UPLOAD_BYTES = int(_env("MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

# Video frame sampling
VIDEO_MIN_FRAMES = int(_env("VIDEO_MIN_FRAMES", "10"))
VIDEO_MAX_FRAMES = int(_env("VIDEO_MAX_FRAMES", "20"))

# Score thresholds used to turn a 0..1 risk score into a label.
# NOTE: these are *heuristic* operating points, not calibrated probabilities.
SUSPICIOUS_THRESHOLD = float(_env("SUSPICIOUS_THRESHOLD", "0.60"))
AUTHENTIC_THRESHOLD = float(_env("AUTHENTIC_THRESHOLD", "0.35"))

# Combination weights for the *demo* fusion (must sum to 1.0). These are the
# hand-set analogue of a trained fusion head. MODEL mode overrides them.
FUSION_WEIGHTS = {
    "individual_risk": 0.45,   # aggregated per-modality anomaly
    "cross_modal_risk": 0.40,  # aggregated pairwise inconsistency
    "coverage": 0.15,          # more missing evidence -> more uncertainty
}

# Cross-modal conflict override: maximum suspicion added when cross-modal
# disagreement is extreme (all pairs inconsistent). This encodes the core idea
# that individually-plausible-but-mutually-inconsistent evidence is suspicious
# as a *coordinated set*. Set to 0.0 to disable.
CONFLICT_OVERRIDE = float(_env("CONFLICT_OVERRIDE", "0.22"))

DISCLAIMER = (
    "TrustLayer is a prototype decision-support system and is not a forensic "
    "or legal certification tool. Results are indicative, not proof."
)
