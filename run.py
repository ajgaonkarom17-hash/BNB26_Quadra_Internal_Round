"""
One-command launcher for TrustLayer.

    python run.py

Starts the FastAPI server (which also serves the dashboard) and opens the
browser. Use this if `uvicorn` is not on your PATH.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser

# Make sure we run from the project root so `backend` is importable.
ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
URL = f"http://{HOST}:{PORT}/"


def _open_browser() -> None:
    time.sleep(2.0)
    try:
        webbrowser.open(URL)
    except Exception:
        pass


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed. Run:  pip install -r requirements.txt")
        sys.exit(1)

    threading.Thread(target=_open_browser, daemon=True).start()
    print(f"TrustLayer starting at {URL}  (Ctrl+C to stop)")
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
