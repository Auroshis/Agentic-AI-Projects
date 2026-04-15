"""
Documents scanner — reads resume from a PDF and/or multiple Google Docs.

Supports:
  - PDF upload (via pdf_path in state)
  - Multiple Google Docs (via google_docs_ids in state — accepts full URLs or bare doc IDs)

All sources are read directly, combined, then parsed once by the LLM.
"""

from __future__ import annotations

import json
import re
from langchain_core.messages import HumanMessage

from DeepJobAgent.state import DeepJobState


def _extract_doc_id(ref: str) -> str:
    """Extract Google Doc ID from a full URL, or return the string as-is if it looks like a bare ID."""
    match = re.search(r'/document/d/([a-zA-Z0-9_-]+)', ref)
    if match:
        return match.group(1)
    return ref.strip()


_PARSE_PROMPT = """\
You are a resume and document parser.

Below is content from one or more documents (PDF resume and/or Google Docs).
Each section is labeled with its source.

Parse everything and return a unified JSON profile of the candidate.

DOCUMENTS:
{combined_text}

Return ONLY valid JSON (no markdown fences) with this exact structure:
{{
  "candidate_name": "",
  "contact_info": {{"email": "", "phone": "", "location": ""}},
  "summary": "",
  "skills": ["Python", "Docker"],
  "technologies": ["FastAPI", "PostgreSQL"],
  "experience_years": 0,
  "experience": [
    {{
      "title": "Software Engineer",
      "company": "ACME Corp",
      "duration": "Jan 2022 - Present",
      "description": "..."
    }}
  ],
  "education": [
    {{"degree": "B.Tech Computer Science", "institution": "IIT", "year": "2020"}}
  ],
  "projects": ["Project A - built X using Y"],
  "certifications": [],
  "raw_text": "<first 500 chars of combined text>",
  "sections": {{}},
  "error": null
}}

Rules:
- Infer skills from technologies mentioned in experience, not just the skills section.
- Estimate experience_years from the oldest start date to today (approximate).
- If the same experience appears in multiple docs, merge rather than duplicate.
"""


# Kept for backward-compat with agents/__init__.py which exports build_google_docs_agent
def build_google_docs_agent(llm):
    """No-op shim — the node no longer uses a ReAct agent internally."""
    return None


async def google_docs_node(state: DeepJobState) -> dict:
    """LangGraph node: reads resume PDF and/or Google Docs, returns google_docs_data."""
    from langchain_openai import ChatOpenAI
    from DeepJobAgent.config import LLM_MODEL, LLM_TEMPERATURE
    from DeepJobAgent.tools.pdf_tools import read_pdf_file
    from DeepJobAgent.tools.google_docs_tools import read_google_doc_raw

    pdf_path = state.get("pdf_path", "") or ""
    google_docs_ids = state.get("google_docs_ids", []) or []

    if not pdf_path and not google_docs_ids:
        return {
            "google_docs_data": {
                "sources": [], "raw_text": "", "sections": {},
                "skills": [], "experience": [], "education": [],
                "error": "No PDF or Google Docs provided",
            },
            "scanners_complete": 1,
            "errors": ["documents_node: no documents provided"],
        }

    combined_parts: list[str] = []
    sources: list[dict] = []
    all_errors: list[str] = []

    # ── Read PDF ──────────────────────────────────────────────────────────────
    if pdf_path:
        result = read_pdf_file(pdf_path)
        if result.get("error"):
            all_errors.append(f"PDF read error: {result['error']}")
        else:
            combined_parts.append(f"=== RESUME (PDF) ===\n{result['raw_text']}")
            sources.append({
                "type": "pdf",
                "ref": pdf_path,
                "title": "Resume PDF",
                "pages": result.get("page_count", 0),
            })

    # ── Read Google Docs ──────────────────────────────────────────────────────
    for doc_ref in google_docs_ids:
        doc_id = _extract_doc_id(doc_ref)
        if not doc_id:
            continue
        result = read_google_doc_raw(doc_id)
        if result.get("error"):
            all_errors.append(f"Google Doc '{doc_id}': {result['error']}")
        else:
            title = result.get("title") or doc_id
            combined_parts.append(f"=== DOCUMENT: {title} ===\n{result['raw_text']}")
            sources.append({"type": "gdoc", "ref": doc_id, "title": title})

    if not combined_parts:
        return {
            "google_docs_data": {
                "sources": sources, "raw_text": "", "sections": {},
                "skills": [], "experience": [], "education": [],
                "error": "; ".join(all_errors) if all_errors else "No content could be read",
            },
            "scanners_complete": 1,
            "errors": all_errors,
        }

    combined_text = "\n\n".join(combined_parts)

    # ── Parse with LLM ────────────────────────────────────────────────────────
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    try:
        resp = await llm.ainvoke([
            HumanMessage(content=_PARSE_PROMPT.format(combined_text=combined_text))
        ])
        raw = (resp.content.strip()
               .removeprefix("```json").removeprefix("```")
               .removesuffix("```").strip())
        docs_data = json.loads(raw)
        docs_data["sources"] = sources
    except json.JSONDecodeError:
        docs_data = {
            "sources": sources, "raw_text": combined_text[:500], "sections": {},
            "skills": [], "experience": [], "education": [],
            "error": "json_parse_failed",
        }
    except Exception as e:
        all_errors.append(f"documents_node LLM: {e}")
        docs_data = {
            "sources": sources, "raw_text": combined_text[:500], "sections": {},
            "skills": [], "experience": [], "education": [],
            "error": str(e),
        }

    return {
        "google_docs_data": docs_data,
        "scanners_complete": 1,
        "errors": all_errors,
    }
