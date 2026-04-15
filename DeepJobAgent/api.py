"""
DeepJobAgent FastAPI backend.

Endpoints:
  POST /api/analyze/stream  — SSE stream; one event per node completion
  GET  /api/health          — health check

Run with:
  uvicorn DeepJobAgent.api:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import json
import sys
import os
import tempfile
import uuid

# Make sure the project root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

_UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "deepjobagent_uploads")
os.makedirs(_UPLOAD_DIR, exist_ok=True)

from DeepJobAgent.graph import graph
from DeepJobAgent.state import DeepJobState

app = FastAPI(title="DeepJobAgent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    job_description: str
    github_username: str
    leetcode_username: str
    linkedin_url: str
    pdf_path: str = ""
    google_docs_ids: list[str] = []


# Maps internal node names to human-readable labels for the UI
NODE_LABELS = {
    "github_scanner":      "GitHub",
    "leetcode_scanner":    "LeetCode",
    "linkedin_scanner":    "LinkedIn",
    "google_docs_scanner": "Documents",
    "aggregate":           "Aggregating",
    "gap_analysis":        "Gap Analysis",
    "plan_generator":      "Learning Plan",
    "resume_tuner":        "Resume Tuner",
}


def _sse(payload: dict) -> str:
    """Format a dict as a Server-Sent Event line."""
    return f"data: {json.dumps(payload, default=str)}\n\n"


@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    """
    Stream analysis progress as Server-Sent Events.
    Each event has shape: { node, label, data, type }
    Final event has type='done'.
    """
    initial_state: DeepJobState = {
        "job_description":   req.job_description,
        "github_username":   req.github_username,
        "leetcode_username": req.leetcode_username,
        "linkedin_url":      req.linkedin_url,
        "pdf_path":          req.pdf_path,
        "google_docs_ids":   req.google_docs_ids,
        "github_data":       None,
        "leetcode_data":     None,
        "linkedin_data":     None,
        "google_docs_data":  None,
        "scanners_complete": 0,
        "skill_gap":         None,
        "learning_plan":     None,
        "tuned_resume":      None,
        "errors":            [],
    }

    async def generate():
        yield _sse({"type": "start", "message": "Pipeline started"})
        try:
            async for chunk in graph.astream(initial_state):
                for node_name, data in chunk.items():
                    if node_name.startswith("__"):
                        continue   # skip LangGraph internal nodes
                    yield _sse({
                        "type":  "node_done",
                        "node":  node_name,
                        "label": NODE_LABELS.get(node_name, node_name),
                        "data":  data,
                    })
            yield _sse({"type": "done"})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.post("/api/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF resume. Returns the server-side path to pass in the analyze request.
    The file is saved to a temp directory and reused within the session.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    filename = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(_UPLOAD_DIR, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return JSONResponse({"pdf_path": file_path, "original_name": file.filename})


@app.get("/api/health")
async def health():
    return {"status": "ok", "agent": "DeepJobAgent"}
