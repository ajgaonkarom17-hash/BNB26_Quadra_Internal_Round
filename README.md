# TrustLayer — AI-Powered Digital Authenticity and Trust

> **Do these multiple pieces of digital evidence form a consistent and trustworthy story?**

TrustLayer is a small, runnable, end-to-end **college / hackathon prototype** that
accepts several digital artifacts (image, video, audio, document), analyses each one
individually, then reasons about the **relationships between them**. Individual files
can each look plausible — but inconsistencies *across* files can reveal manipulation.

> ⚠️ **TrustLayer is a prototype decision-support system and is not a forensic or
> legal certification tool.** Results are indicative, not proof. All files stay local
> and are never sent to external APIs by default.

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Features](#3-features)
4. [Installation](#4-installation)
5. [Backend setup](#5-backend-setup)
6. [Frontend setup](#6-frontend-setup)
7. [Demo mode](#7-demo-mode)
8. [Model mode](#8-model-mode)
9. [Dataset structure](#9-dataset-structure)
10. [OCR (optional, local)](#10-ocr-optional-local)
11. [Cross-modal claim comparison](#11-cross-modal-claim-comparison)
12. [Training](#12-training)
13. [Evaluation](#13-evaluation)
14. [API](#14-api)
15. [Database](#15-database)
16. [Limitations](#16-limitations)
17. [Future improvements](#17-future-improvements)
18. [What is genuinely AI vs heuristic](#18-what-is-genuinely-ai-vs-heuristic)

---

## 1. Project overview

The central research idea:

> **Individual files may each appear authentic, but relationships and
> inconsistencies between multiple files can reveal manipulation.**

TrustLayer answers the question above by producing an **overall assessment**
(`AUTHENTIC` / `SUSPICIOUS` / `INCONCLUSIVE`), plus separate **confidence**,
**evidence coverage**, **cross-modal conflicts**, an **evidence graph**, and a
plain-language **explanation** of *why* the result came out the way it did.

It deliberately never outputs a bare `FAKE — 91%`. Model/heuristic scores are never
presented as guaranteed probabilities or absolute proof.

---

## 2. Architecture

```
USER
 ↓  (vanilla-JS dashboard served by FastAPI)
CREATE INVESTIGATION
 ↓
UPLOAD IMAGE / VIDEO / AUDIO / DOCUMENT
 ↓
INDIVIDUAL MODALITY ANALYSIS          backend/services/*_analyzer.py
 ↓
FEATURE / EMBEDDING EXTRACTION        (128-d common representation)
 ↓
COMMON REPRESENTATION                 backend/services/base.py
 ↓
MULTIMODAL FUSION                     backend/services/fusion.py
 ↓
CROSS-MODAL CONSISTENCY               backend/services/cross_modal.py
 ↓
EVIDENCE GRAPH                        backend/services/evidence_graph.py
 ↓
UNCERTAINTY + MISSING/CONFLICTING     backend/services/uncertainty.py
 ↓
FINAL ASSESSMENT                      backend/services/pipeline.py
 ↓
EXPLANATION + DASHBOARD               backend/services/explanation.py
```

Each service is independent, so a stronger model can be dropped in later without
touching the rest of the pipeline.

### Folder structure

```
trustlayer/
├── backend/
│   ├── main.py                 # FastAPI app + static frontend serving
│   ├── config.py               # paths, MODE, thresholds, weights
│   ├── api/
│   │   ├── investigations.py   # upload / analyze / results / graph
│   │   └── system.py           # health, capabilities, evaluation, limits, sample
│   ├── models/
│   │   ├── schemas.py          # Pydantic request models
│   │   └── registry.py         # MODEL-mode artifact loader
│   ├── services/
│   │   ├── base.py             # ModalityResult contract + common representation
│   │   ├── image_analyzer.py
│   │   ├── video_analyzer.py
│   │   ├── audio_analyzer.py
│   │   ├── document_analyzer.py
│   │   ├── cross_modal.py
│   │   ├── fusion.py
│   │   ├── uncertainty.py
│   │   ├── evidence_graph.py
│   │   ├── explanation.py
│   │   └── pipeline.py         # orchestrates the whole pipeline
│   ├── database/
│   │   └── db.py               # SQLite schema + queries
│   └── utils/
│       ├── helpers.py
│       └── upload.py
├── frontend/
│   ├── index.html              # single-page dashboard (no build required)
│   ├── assets/styles.css
│   ├── src/{api,components,graph,views,app}.js
│   ├── package.json            # OPTIONAL Vite dev setup
│   └── vite.config.js
├── data/
│   ├── sample/                 # generated demo files
│   ├── raw/                    # your datasets
│   ├── processed/
│   ├── manifests/              # training manifests (.csv)
│   └── uploads/                # uploaded evidence (per investigation)
├── training/
│   ├── common.py               # numpy logistic / MLP + metrics
│   ├── individual.py           # shared per-modality trainer
│   ├── train_image.py  train_video.py  train_audio.py  train_document.py
│   ├── train_fusion.py         # multimodal + modality dropout + multi-task loss
│   ├── evaluate.py             # known vs unseen + model ablations
│   └── prepare_dataset.py      # build manifests from a folder layout
├── scripts/
│   ├── build_sample.py         # create the demo investigation
│   ├── http_test.py            # HTTP smoke test (pure stdlib)
│   └── frontend_contract_test.py
├── models/                     # trained artifacts live here (git-ignored)
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 3. Features

**Per investigation**

- Upload image, video, audio and document evidence (any combination).
- **Real embedded metadata**: reads MP4/MOV `mvhd` creation time (pure-Python
  atom parsing — no ffmpeg), image EXIF, and WAV/MP3 date tags.
- **Claim extraction**: pulls times-of-day, dates, named speakers and places out
  of text, OCR'd image captions and metadata.
- Individual analysis per modality with real signal processing.
- 128-d common-representation embedding per modality.
- Pairwise **Heuristic Consistency Score** for every available pair
  (`image_video`, `video_audio`, `image_document`, …).
- **Cross-modal conflict checks**, e.g. *"image says 5 PM but video metadata says
  09:12"*, *"report names a speaker the audio cannot verify"*, and
  *"video is 6s but audio is 2.5s"*.
- **Cross-modal conflict override** in fusion: individually-plausible files that
  are mutually inconsistent are escalated to SUSPICIOUS (the core research idea).
- Availability mask `[1,1,0,1]` for missing modalities.
- Multimodal fusion → overall assessment + confidence + consistency.
- Evidence coverage tracked **separately** from confidence.
- Evidence graph (files + entities + claims, with similarity / contradiction /
  timestamp / identity / mentions edges).
- Video timeline with **Suspicious Frame / Timestamp Indicators**.
- "What Changed?" summary (per-aspect status cards).
- Explanation: why flagged / supporting / conflicting / missing / limitations.
- Investigation history stored in SQLite.
- Demo mode works with **no trained models**; Model mode auto-loads any that exist.

**Pages:** Home, New Investigation, Analysis Processing, Investigation Results,
Evidence Graph, Investigation History, Evaluation / Model.

---

## 4. Installation

You need **Python 3.10+**. Node.js is optional (only for the Vite dev setup).

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS / Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

- `fastapi` / `uvicorn` — the web server and REST API.
- `python-multipart` — needed to receive file uploads.
- `numpy`, `pandas` — math and small data tables.
- `Pillow` — image loading + EXIF.
- `opencv-python` — video decoding, frame analysis, face detection.

**Optional packages** (the app degrades gracefully without them, see
`requirements.txt`): `PyMuPDF` (better PDF text), `soundfile` / `librosa`
(MP3/FLAC audio), `scikit-learn` (extra metrics), `torch` / `transformers`
(future deep encoders).

> Without `soundfile`/`librosa`, audio analysis still works for uncompressed
> **WAV** via a built-in NumPy decoder. Without `PyMuPDF`, PDF text uses a small
> built-in fallback parser.

---

## 5. Backend setup

From the **project root** (`trustlayer/`), with the venv activated:

```bash
python -m uvicorn backend.main:app --reload
```

…or use the one-command launcher (recommended on Windows, opens the browser for you):

```bash
python run.py
```

- `uvicorn` is the server. Using `python -m uvicorn` avoids the "uvicorn is not
  recognized" PATH issue on Windows.
- `backend.main:app` means “the `app` object inside `backend/main.py`”.
- `--reload` restarts automatically when you edit code (dev only).

Then open:

- Dashboard: **http://127.0.0.1:8000/**
- Interactive API docs: **http://127.0.0.1:8000/docs**

Generate the demo data the first time:

```bash
python -m scripts.build_sample
```

…or just click **Run Sample Investigation** on the Home page (it creates *and*
analyzes the sample in one step).

---

## 6. Frontend setup

The frontend is plain HTML/CSS/JS and is **served by FastAPI automatically** — no
build step is required for the demo. Step 5 is all you need.

If you want hot-reload while editing the UI, an optional Vite setup is included:

```bash
cd frontend
npm install
npm run dev
```

- `npm install` downloads dev tools (Vite).
- `npm run dev` starts a dev server at `http://localhost:5173` and proxies `/api`
  to the FastAPI server on port 8000 (see `frontend/vite.config.js`).

Keep the FastAPI server running in another terminal when using Vite.

---

## 7. Demo mode

Default. Set in `.env` (copy from `.env.example`) or the environment:

```text
MODE=demo
```

Demo mode uses deterministic, transparent heuristics and a fixed fusion formula.
A **“DEMO / PROTOTYPE ANALYSIS”** badge is shown in the UI. Every analyzer result
is labelled with its engine (`demo-heuristic`), and the UI states clearly that
consistency scores are **heuristic, not probabilities**.

Demo mode needs **no training and no GPU**.

---

## 8. Model mode

```text
MODE=model
```

TrustLayer looks in `models/` for trained artifacts and loads them if present;
if a file is missing it **falls back to the demo heuristic for that modality** and
records the fallback honestly.

Expected artifacts (all optional):

| File | Contents | Used by |
|---|---|---|
| `models/image_head.npz` | `features`, `w`, `b` | image analyzer |
| `models/video_head.npz` | `features`, `w`, `b` | video analyzer |
| `models/audio_head.npz` | `features`, `w`, `b` | audio analyzer |
| `models/document_head.npz` | `features`, `w`, `b` | document analyzer |
| `models/fusion.npz` | `w1,b1,w2,b2` (small MLP) | fusion |

Produce them with the training scripts below. The **Evaluation / Model** page
shows exactly which artifacts are present at runtime.

---

## 9. Dataset structure

### Per-modality manifest (`path,label`)

```csv
path,label
data/raw/image/authentic_001.jpg,0
data/raw/image/manipulated_001.jpg,1
```

`label`: `1` = manipulated/suspicious, `0` = authentic.

### Multimodal manifest (one row per investigation)

```csv
investigation_id,image_path,video_path,audio_path,document_path,image_label,video_label,audio_label,document_label,cross_modal_label,overall_label,split,manipulation
demo_001,data/..,data/..,data/..,data/..,0,0,0,0,1,0,known,none
demo_002,data/..,data/..,,data/..,0,1,0,0,0,1,unseen,face_swap
```

- `overall_label`: `0` authentic, `1` suspicious, `2` inconclusive.
- `cross_modal_label`: `1` consistent, `0` inconsistent.
- `split`: `known` vs `unseen` (used by `training/evaluate.py`).
- `manipulation`: family name (e.g. `face_swap`, `audio_spoof`, `none`).

`training/prepare_dataset.py` builds a `path,label` manifest automatically from a
folder layout (e.g. `authentic/` vs `manipulated/`).

**Possible future datasets** (not downloaded automatically):
CIFAKE / GenImage (images), FaceForensics++ / Celeb-DF / DFDC (video),
ASVspoof (audio). TrustLayer never downloads large datasets for you.

---

## 10. OCR (optional, local)

If `pytesseract` and the Tesseract binary are installed, TrustLayer runs OCR on
images and on a small set of sampled video frames (every other frame, max 8,
never every frame). Video claims must be stable across >= half of the OCR'd
frames to be trusted (`ocr-aggregate` with a confidence equal to frame support).

```bash
pip install pytesseract
# Windows:  winget install --id tesseract-ocr.tesseract -e
# macOS:    brew install tesseract
# Linux:    sudo apt install tesseract-ocr
```

Without Tesseract the pipeline continues normally and reports OCR as
unavailable — nothing crashes and no results are fabricated.

OCR produces structured **claims** (`backend/services/claims.py`):
`{type, value, original, source, evidence_id, confidence, method}` for
timestamps, times of day, dates, locations, people, URLs, emails, phones.

## 11. Cross-modal claim comparison

`backend/services/claim_compare.py` compares claims across modalities with
explicit normalisation — `"5 PM"`, `"5:00 PM"` and `"17:00"` are the same time;
`"Riverside Park"` and `"riverside park."` are the same place. Missing values
are reported as *unavailable*, never as conflicts.

---

## 12. Training

Pattern: `Dataset → Preprocessing → Feature Extraction → Individual Model →
Embedding Generation → Multimodal Dataset → Fusion Model → Evaluation`.

The analyzers already turn each file into a small numeric feature dict, so an
“individual model” is a tiny logistic head over those features. This is honest
(no fake deep learning) and leaves a clean seam to swap in a CNN/transformer later.

Train an individual head:

```bash
python -m training.train_image    --manifest data/manifests/image.csv
python -m training.train_video    --manifest data/manifests/video.csv
python -m training.train_audio    --manifest data/manifests/audio.csv
python -m training.train_document --manifest data/manifests/document.csv
```

Train the fusion model:

```bash
python -m training.train_fusion --manifest data/manifests/multimodal.csv
```

Fusion input (exactly as specified):

```
image_emb(128) + video_emb(128) + audio_emb(128) + text_emb(128) + availability_mask(4) = 516-d
```

- **Modality dropout** is implemented: random present modalities are zeroed and
  the mask updated during training, so the model learns to work with missing
  evidence (e.g. `Mask = [1,1,0,1]`).
- A small **multi-task loss** trains overall classification + consistency
  classification + modality-presence prediction together.

---

## 13. Evaluation

### Image detector on a real dataset (CIFAKE) — one command

```bash
python -m training.setup_cifake --zip "C:\path\to\archive.zip"
```

This extracts a balanced subset, trains the image detector, and evaluates it on a
**held-out split** that was never seen during training. The measured result is
written to `models/evaluation_report.json` and shown on the **Evaluation** page.

Real measured numbers from the included CIFAKE run (6,000 train / 2,000 test):

```
Held-out test (CIFAKE test/ split, n=2000)
  accuracy = 0.9055
  precision= 0.9108
  recall   = 0.899
  f1       = 0.9049
  roc_auc  = 0.9673
```

These are honest, non-inflated hold-out numbers — not in-sample training scores.

### Multimodal ablations

```bash
python -m training.evaluate --manifest data/manifests/eval.csv
```

Compares three configurations on a held-out manifest:

```
Individual-only model
vs  Multimodal model
vs  Multimodal + Cross-modal reasoning
```

and reports **Accuracy / Precision / Recall / F1 / ROC-AUC**, split by `known`
vs `unseen` manipulations (and you can add columns for different compression /
conditions).

> If an experiment has not actually been performed, the Evaluation page displays
> **"Not evaluated yet"**. TrustLayer never invents results.

### Automated tests

```bash
python -m pytest tests -q
```

The suite covers all four analyzers, missing metadata/modality handling, OCR
degradation, timestamp/location normalisation, agreement and conflict
detection, evidence coverage, fusion, the evidence graph, API endpoints, and
the controlled timestamp-conflict media fixture (image says 17:00, video says
09:00 → cross-modal conflict). Tests assert *behavior*, never hard-coded scores.

---

## 14. API

Add `Content-Type: application/json` for JSON bodies; uploads use
`multipart/form-data`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/investigations` | Create (`{"name": "..."}`) |
| `POST` | `/api/investigations/{id}/upload` | Upload one file (`file` field) |
| `POST` | `/api/investigations/{id}/analyze` | Run the full pipeline |
| `GET`  | `/api/investigations` | List investigations |
| `GET`  | `/api/investigations/{id}` | Investigation detail |
| `GET`  | `/api/investigations/{id}/evidence` | Uploaded evidence + per-file analysis |
| `GET`  | `/api/investigations/{id}/graph` | Evidence graph `{nodes, edges}` |
| `GET`  | `/api/investigations/{id}/results` | Full dashboard payload |
| `GET`  | `/api/investigations/{id}/cross-modal` | Pairwise consistency rows |
| `DELETE` | `/api/investigations/{id}` | Delete investigation + files |
| `DELETE` | `/api/evidence/{evidence_id}` | Delete one evidence file |
| `POST` | `/api/sample` | Build + analyze the sample investigation |
| `GET`  | `/api/health` | Health check |
| `GET`  | `/api/system/capabilities` | Mode + detected libraries + models |
| `GET`  | `/api/evaluation` | Evaluation page data (honest “not evaluated”) |
| `GET`  | `/api/limits` | Limitations page data |

**Example**

```bash
curl -X POST http://127.0.0.1:8000/api/investigations \
     -H "Content-Type: application/json" \
     -d "{\"name\":\"Event Authenticity Test\"}"

curl -F "file=@event_image.jpg" \
     http://127.0.0.1:8000/api/investigations/<ID>/upload

curl -X POST http://127.0.0.1:8000/api/investigations/<ID>/analyze
curl http://127.0.0.1:8000/api/investigations/<ID>/results
```

The frontend and backend share exactly these JSON structures; run
`python scripts/frontend_contract_test.py` (with the server up) to verify.

---

## 15. Database

SQLite file at `data/trustlayer.db`, created automatically on startup. Tables:

| Table | Key columns |
|---|---|
| `investigations` | `id, name, created_at, status, assessment, confidence, confidence_score, coverage, risk_score, summary` |
| `evidence` | `id, investigation_id, filename, modality, content_type, size_bytes, stored_path, uploaded_at, individual_score, individual_label` |
| `analysis_results` | `id, evidence_id, investigation_id, modality, score, label, features, findings, engine, detail, created_at` |
| `cross_modal_results` | `id, investigation_id, pair, consistency, label, conflicts, method, created_at` |
| `evidence_relationships` | `id, investigation_id, source, target, relationship, score, detail, created_at` |

JSON columns (`features`, `findings`, `conflicts`, …) are stored as text and
decoded by `backend/database/db.py`.

---

## 16. Limitations

- Individual detectors are **lightweight heuristics** (or small trained heads),
  not forensic-grade. Deepfake / audio-spoofing detectors are **not** included.
- **The image detector is trained on CIFAKE.** On CIFAKE's held-out test split it
  reaches ~0.91 accuracy / 0.97 ROC-AUC, but that is a *specific* dataset of
  32×32 images. On modern high-resolution AI art it is much less reliable — treat
  a single image score as a weak signal, not proof. Features are computed at a
  fixed canonical resolution to reduce (but not eliminate) this domain gap.
- **This is NOT an AI-text detector.** Text signals are surface-style heuristics.
- Video "suspicious timestamps" are **indicators**, not confirmed manipulation
  localization.
- Cross-modal consistency scores are **heuristic**, not calibrated probabilities.
- Face handling uses the classic OpenCV Haar cascade (not deep face recognition).
- OCR for scanned PDFs is not implemented (text PDFs only).
- No reverse image search, provenance (C2PA), or cryptographic signing.
- Cross-modal reasoning needs more than one modality to be meaningful.

See section 16 below for the full "genuinely AI vs heuristic vs not implemented"
breakdown (the same data is available from `GET /api/limits`).

**Privacy:** uploaded files are stored under `data/uploads/<investigation_id>/`
and never leave your machine.

---

## 17. Future improvements

Clear seams are left for:

- Stronger deepfake image/video detectors (CNN / ViT).
- Audio anti-spoofing models (ASVspoof-style).
- Multimodal Transformers, cross-attention fusion, CLIP-style alignment
  (replace `backend/services/fusion.py` internals; the `fuse(...)` interface is
  stable).
- Real OCR, reverse image search, provenance (C2PA), cryptographic hashing.
- Real-time streams, PostgreSQL, and cloud deployment.

---

## 18. What is genuinely AI vs heuristic

TrustLayer is explicit about this (also available from `GET /api/limits`):

**Genuinely AI / signal processing**

- STFT / mel-style spectral analysis (NumPy FFT) for audio.
- Error Level Analysis + Laplacian / high-frequency statistics for images.
- Frame-difference temporal analysis for video (OpenCV).
- Deterministic random projection into a shared 128-d embedding space.
- Small trained heads + MLP fusion (when MODEL-mode artifacts exist).

**Heuristic (rule-based)**

- Manipulation risk weights and thresholds.
- Cross-modal consistency scoring (embedding similarity + domain rules).
- Face counting via OpenCV Haar cascade.
- Metadata anomaly detection.
- Text surface-style signals (**not** AI-text detection).

**Not implemented yet**

- Deepfake face-swap detectors, audio anti-spoofing, OCR, reverse image search,
  provenance, cryptographic signing, true manipulation localization,
  Transformer / cross-attention fusion.

---

## Quick start (TL;DR)

```bash
python -m venv venv
venv\Scripts\activate            # Windows  (source venv/bin/activate on mac/linux)
pip install -r requirements.txt
python run.py                    # or: python -m uvicorn backend.main:app --reload
# open http://127.0.0.1:8000/ and click "Run Sample Investigation"
```

> TrustLayer is a prototype decision-support system and is not a forensic or
> legal certification tool.
