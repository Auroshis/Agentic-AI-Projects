"""
Google Docs tools — reads a resume stored in Google Docs.

Auth options (configure via env vars):
  GOOGLE_CREDS_PATH  — path to a service account JSON key file (recommended)
  GOOGLE_TOKEN_PATH  — path to an OAuth2 token JSON file (for user-credential flow)

The Google Doc must be shared (View) with the service account email if using
the service account path.  The document ID is the string between /d/ and /edit
in the Doc URL: https://docs.google.com/document/d/<DOC_ID>/edit
"""

from __future__ import annotations

import re
from typing import Optional

from langchain_core.tools import tool

from DeepJobAgent.config import GOOGLE_CREDS_PATH

# ── Resume section header patterns ───────────────────────────────────────────
_SECTION_PATTERNS = {
    "summary":    re.compile(r"(summary|objective|about me|profile)", re.I),
    "experience": re.compile(r"(experience|employment|work history|career)", re.I),
    "education":  re.compile(r"(education|academics|degree|university)", re.I),
    "skills":     re.compile(r"(skills|technologies|tech stack|tools|competencies)", re.I),
    "projects":   re.compile(r"(projects|portfolio|side projects)", re.I),
    "certifications": re.compile(r"(certif|license|credential)", re.I),
    "achievements":   re.compile(r"(achievement|award|honor|recognition)", re.I),
}


def _get_docs_service():
    """Initialize Google Docs API client from service account credentials."""
    import os
    try:
        from googleapiclient.discovery import build
        from google.oauth2 import service_account

        if not os.path.exists(GOOGLE_CREDS_PATH):
            raise FileNotFoundError(
                f"Google credentials not found at '{GOOGLE_CREDS_PATH}'. "
                "Set GOOGLE_CREDS_PATH env var to your service account JSON key."
            )

        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_CREDS_PATH,
            scopes=["https://www.googleapis.com/auth/documents.readonly"],
        )
        return build("docs", "v1", credentials=creds, cache_discovery=False)

    except ImportError:
        raise ImportError(
            "Google API packages not installed. Run:\n"
            "  pip install google-api-python-client google-auth google-auth-httplib2"
        )


def _extract_text(doc_body: dict) -> str:
    """
    Recursively walk the Google Docs body JSON structure and extract plain text.
    Handles paragraphs, tables, and lists.
    """
    lines = []

    for element in doc_body.get("content", []):
        # Paragraph
        para = element.get("paragraph")
        if para:
            parts = []
            for pe in para.get("elements", []):
                text_run = pe.get("textRun")
                if text_run:
                    parts.append(text_run.get("content", ""))
            line = "".join(parts)
            lines.append(line)
            continue

        # Table — extract each cell
        table = element.get("table")
        if table:
            for row in table.get("tableRows", []):
                for cell in row.get("tableCells", []):
                    cell_text = _extract_text(cell.get("content", []))
                    lines.append(cell_text)

    return "".join(lines)


def _extract_text_flat(content_list: list) -> str:
    """Accept content list directly (for table cells recursion)."""
    doc_like = {"content": content_list}
    return _extract_text(doc_like)


def _segment_sections(raw_text: str) -> dict:
    """
    Split the raw resume text into logical sections by detecting
    common section headers (case-insensitive).
    Returns a dict of {section_name: section_content}.
    """
    sections: dict[str, str] = {}
    lines = raw_text.splitlines()

    current_section = "header"
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue

        matched_section: Optional[str] = None
        for sec_name, pattern in _SECTION_PATTERNS.items():
            if pattern.search(stripped) and len(stripped) < 60:
                matched_section = sec_name
                break

        if matched_section:
            # Save current buffer to current section
            sections[current_section] = "\n".join(buffer).strip()
            current_section = matched_section
            buffer = []
        else:
            buffer.append(line)

    # Flush last section
    if buffer:
        sections[current_section] = "\n".join(buffer).strip()

    # Remove empty sections
    return {k: v for k, v in sections.items() if v}


def read_google_doc_raw(doc_id: str) -> dict:
    """
    Direct (non-tool) function to read a Google Doc by ID.
    Call this from nodes; use read_google_doc (tool) inside ReAct agents.
    """
    try:
        service = _get_docs_service()
        doc = service.documents().get(documentId=doc_id).execute()
        raw_text = _extract_text(doc.get("body", {}))
        sections = _segment_sections(raw_text)
        return {
            "document_id": doc_id,
            "title": doc.get("title", ""),
            "raw_text": raw_text,
            "sections": sections,
            "error": None,
        }
    except Exception as e:
        return {
            "document_id": doc_id,
            "raw_text": "",
            "sections": {},
            "error": str(e),
        }


@tool
def read_google_doc(doc_id: str) -> dict:
    """
    Read a Google Document by its ID and return the full text plus
    resume sections parsed into a structured dict.

    The document must be shared with the service account email (view access).
    doc_id is the string from the Google Docs URL:
      https://docs.google.com/document/d/<doc_id>/edit
    """
    return read_google_doc_raw(doc_id)
