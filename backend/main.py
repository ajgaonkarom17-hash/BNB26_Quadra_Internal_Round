"""
TrustLayer FastAPI entry point.

Run with:
    uvicorn backend.main:app --reload

It:
  * initialises the SQLite database,
  * mounts the REST API under /api,
  * serves the static frontend (frontend/index.html + assets) at /,
  * handles a few common error cases with clean JSON.

The frontend is plain HTML/CSS/JS served by this same server, so there is no
build step required to demo the prototype. (A Vite setup is documented in the
README for those who want it.)
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.config import FRONTEND_DIR, MODE
from backend.database.db import init_db
from backend.api import investigations, system

app = FastAPI(
    title="TrustLayer",
    description=("AI-Powered Digital Authenticity and Trust - a prototype "
                 "multimodal evidence consistency analyzer. Decision-support "
                 "only; not a forensic certification tool."),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # prototype: local dev convenience
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- lifecycle ------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    init_db()
    print(f"[TrustLayer] started in MODE={MODE}")


# ---- API ------------------------------------------------------------------
app.include_router(investigations.router, prefix="/api", tags=["investigations"])
app.include_router(system.router, prefix="/api", tags=["system"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # pragma: no cover
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal error. The prototype handled it gracefully.",
                 "error": str(exc)},
    )


# ---- Frontend (static) ----------------------------------------------------
if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")
if (FRONTEND_DIR / "src").exists():
    app.mount("/src", StaticFiles(directory=str(FRONTEND_DIR / "src")), name="src")


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "TrustLayer API is running. Frontend not found.",
                         "docs": "/docs"})


@app.get("/favicon.ico")
def favicon():
    return JSONResponse({}, status_code=204)
