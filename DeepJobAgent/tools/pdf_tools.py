"""
PDF tools — extracts text from a PDF file.

Requires: pypdf>=4.0.0  (pip install pypdf)
"""

from __future__ import annotations

import re
from typing import Optional


_SECTION_PATTERNS = {
    "summary":        re.compile(r"(summary|objective|about me|profile)", re.I),
    "experience":     re.compile(r"(experience|employment|work history|career)", re.I),
    "education":      re.compile(r"(education|academics|degree|university)", re.I),
    "skills":         re.compile(r"(skills|technologies|tech stack|tools|competencies)", re.I),
    "projects":       re.compile(r"(projects|portfolio|side projects)", re.I),
    "certifications": re.compile(r"(certif|license|credential)", re.I),
    "achievements":   re.compile(r"(achievement|award|honor|recognition)", re.I),
}


def _segment_sections(raw_text: str) -> dict:
    sections: dict[str, str] = {}
    lines = raw_text.splitlines()
    current_section = "header"
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append("")
            continue

        matched: Optional[str] = None
        for sec_name, pattern in _SECTION_PATTERNS.items():
            if pattern.search(stripped) and len(stripped) < 60:
                matched = sec_name
                break

        if matched:
            sections[current_section] = "\n".join(buffer).strip()
            current_section = matched
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections[current_section] = "\n".join(buffer).strip()

    return {k: v for k, v in sections.items() if v}


def read_pdf_file(file_path: str) -> dict:
    """
    Extract text from a PDF at file_path.
    Returns a dict with raw_text, sections, page_count, and error.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        return {
            "file_path": file_path,
            "raw_text": "",
            "sections": {},
            "page_count": 0,
            "error": "pypdf not installed — run: pip install pypdf",
        }

    try:
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        raw_text = "\n".join(pages_text)
        return {
            "file_path": file_path,
            "raw_text": raw_text,
            "sections": _segment_sections(raw_text),
            "page_count": len(reader.pages),
            "error": None,
        }
    except Exception as e:
        return {
            "file_path": file_path,
            "raw_text": "",
            "sections": {},
            "page_count": 0,
            "error": str(e),
        }
