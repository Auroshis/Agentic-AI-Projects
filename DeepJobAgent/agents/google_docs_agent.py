"""Google Docs scanner sub-agent — reads the user's resume."""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from DeepJobAgent.tools.google_docs_tools import read_google_doc
from DeepJobAgent.state import DeepJobState

_SYSTEM_PROMPT = """You are a resume parser that reads resumes from Google Docs.

Your job:
1. Call read_google_doc with the provided document ID.
2. From the raw_text and sections returned, understand the resume structure.
3. Return a clean JSON summary of the resume content.

Return ONLY valid JSON (no markdown fences) with this exact structure:
{
  "document_id": "<id>",
  "title": "",
  "candidate_name": "",
  "contact_info": {"email": "", "phone": "", "location": ""},
  "summary": "",
  "skills": ["Python", "Docker", ...],
  "technologies": ["FastAPI", "PostgreSQL", ...],
  "experience_years": 0,
  "experience": [
    {
      "title": "Software Engineer",
      "company": "ACME Corp",
      "duration": "Jan 2022 - Present",
      "description": "..."
    }
  ],
  "education": [
    {"degree": "B.Tech Computer Science", "institution": "IIT", "year": "2020"}
  ],
  "projects": ["Project A - built X using Y", ...],
  "certifications": [],
  "raw_text": "<first 500 chars>",
  "sections": {},
  "error": null
}

Extract skills from the skills section AND infer from technologies mentioned in experience.
Calculate experience_years from the oldest start date to today (approximate).
"""


def build_google_docs_agent(llm):
    return create_react_agent(
        model=llm,
        tools=[read_google_doc],
        prompt=_SYSTEM_PROMPT,
    )


async def google_docs_node(state: DeepJobState) -> dict:
    """LangGraph node: reads the resume from Google Docs and writes google_docs_data."""
    from langchain_openai import ChatOpenAI
    from DeepJobAgent.config import LLM_MODEL, LLM_TEMPERATURE

    doc_id = state.get("google_docs_id", "")
    if not doc_id:
        return {
            "google_docs_data": {
                "document_id": "", "raw_text": "", "sections": {},
                "skills": [], "experience": [], "education": [],
                "error": "No Google Docs ID provided"
            },
            "scanners_complete": 1,
            "errors": ["google_docs_node: no doc ID provided"],
        }

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = build_google_docs_agent(llm)

    try:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Read the resume from Google Doc ID: {doc_id}")]
        })
        raw = result["messages"][-1].content
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        gdocs_data = json.loads(clean)
    except json.JSONDecodeError:
        gdocs_data = {
            "document_id": doc_id, "raw_text": "", "sections": {},
            "skills": [], "experience": [], "education": [],
            "error": "json_parse_failed"
        }
    except Exception as e:
        gdocs_data = {
            "document_id": doc_id, "raw_text": "", "sections": {},
            "skills": [], "experience": [], "education": [],
            "error": str(e)
        }
        return {
            "google_docs_data": gdocs_data,
            "scanners_complete": 1,
            "errors": [f"google_docs_node: {e}"],
        }

    return {"google_docs_data": gdocs_data, "scanners_complete": 1}
