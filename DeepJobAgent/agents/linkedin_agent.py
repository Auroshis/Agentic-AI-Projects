"""LinkedIn scanner sub-agent."""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from DeepJobAgent.tools.linkedin_tools import scrape_linkedin_profile, parse_linkedin_manual
from DeepJobAgent.state import DeepJobState

_SYSTEM_PROMPT = """You are a LinkedIn profile analyst who extracts professional experience and skills.

Strategy:
1. Call scrape_linkedin_profile with the URL first.
2. If the result has scrape_method='manual' (scraping blocked), note this in the output —
   the user will be asked to provide their profile text separately.
   Do NOT call parse_linkedin_manual unless a profile_text was explicitly given.
3. From whatever data is available, extract structured professional information.

Return ONLY valid JSON (no markdown fences) with this exact structure:
{
  "name": "",
  "headline": "",
  "current_role": "",
  "skills": ["Python", "Machine Learning", ...],
  "experience": [
    {"title": "...", "company": "...", "duration": "...", "description": "..."}
  ],
  "education": [
    {"degree": "...", "institution": "...", "year": "..."}
  ],
  "scrape_method": "curl_cffi|manual",
  "error": null,
  "manual_input_required": false
}

If scraping failed and no manual text is available, set:
  manual_input_required: true
  error: "explanation for the user"
  skills, experience, education: []
"""


def build_linkedin_agent(llm):
    return create_react_agent(
        model=llm,
        tools=[scrape_linkedin_profile, parse_linkedin_manual],
        prompt=_SYSTEM_PROMPT,
    )


async def linkedin_node(state: DeepJobState) -> dict:
    """LangGraph node: runs the LinkedIn scanner and writes linkedin_data."""
    from langchain_openai import ChatOpenAI
    from DeepJobAgent.config import LLM_MODEL, LLM_TEMPERATURE

    url = state.get("linkedin_url", "")
    if not url:
        return {
            "linkedin_data": {
                "name": "", "headline": "", "current_role": "",
                "skills": [], "experience": [], "education": [],
                "scrape_method": "manual", "error": "No LinkedIn URL provided",
                "manual_input_required": True
            },
            "scanners_complete": 1,
            "errors": ["linkedin_node: no URL provided"],
        }

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = build_linkedin_agent(llm)

    try:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Scan LinkedIn profile: {url}")]
        })
        raw = result["messages"][-1].content
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        linkedin_data = json.loads(clean)
    except json.JSONDecodeError:
        linkedin_data = {
            "name": "", "headline": "", "current_role": "",
            "skills": [], "experience": [], "education": [],
            "scrape_method": "manual", "error": "json_parse_failed",
            "manual_input_required": True
        }
    except Exception as e:
        linkedin_data = {
            "name": "", "headline": "", "current_role": "",
            "skills": [], "experience": [], "education": [],
            "scrape_method": "manual", "error": str(e),
            "manual_input_required": True
        }
        return {
            "linkedin_data": linkedin_data,
            "scanners_complete": 1,
            "errors": [f"linkedin_node: {e}"],
        }

    return {"linkedin_data": linkedin_data, "scanners_complete": 1}
