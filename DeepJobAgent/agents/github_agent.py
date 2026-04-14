"""GitHub scanner sub-agent."""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from DeepJobAgent.tools.github_tools import (
    get_github_profile,
    get_github_repos,
    get_github_languages,
)
from DeepJobAgent.state import DeepJobState

_SYSTEM_PROMPT = """You are a GitHub profile analyst specializing in technical skill extraction.

Your job:
1. Call get_github_profile to get the user's bio and stats
2. Call get_github_repos to get their repositories
3. Call get_github_languages to get their language breakdown
4. Synthesize all data into a JSON summary

Return ONLY valid JSON (no markdown fences) with this exact structure:
{
  "username": "<username>",
  "top_languages": ["Python", "JavaScript", ...],
  "all_topics": ["machine-learning", "react", ...],
  "repos": [
    {"name": "...", "description": "...", "language": "...", "topics": [...], "stars": 0}
  ],
  "highlights": "2-3 sentence summary of their strongest technical areas",
  "error": null
}

Extract topics from repo topics field AND infer from repo names/descriptions.
Sort top_languages by frequency. Include up to 15 repos sorted by stars.
"""


def build_github_agent(llm):
    return create_react_agent(
        model=llm,
        tools=[get_github_profile, get_github_repos, get_github_languages],
        prompt=_SYSTEM_PROMPT,
    )


async def github_node(state: DeepJobState) -> dict:
    """LangGraph node: runs the GitHub scanner and writes github_data."""
    from langchain_openai import ChatOpenAI
    from DeepJobAgent.config import LLM_MODEL, LLM_TEMPERATURE

    username = state.get("github_username", "")
    if not username:
        return {
            "github_data": {
                "username": "", "top_languages": [], "all_topics": [],
                "repos": [], "highlights": "", "error": "No GitHub username provided"
            },
            "scanners_complete": 1,
            "errors": ["github_node: no username provided"],
        }

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = build_github_agent(llm)

    try:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Analyze GitHub profile for username: {username}")]
        })
        raw = result["messages"][-1].content
        # Strip markdown fences if the model added them
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        github_data = json.loads(clean)
    except json.JSONDecodeError:
        github_data = {
            "username": username, "top_languages": [], "all_topics": [],
            "repos": [], "highlights": raw, "error": "json_parse_failed"
        }
    except Exception as e:
        github_data = {
            "username": username, "top_languages": [], "all_topics": [],
            "repos": [], "highlights": "", "error": str(e)
        }
        return {
            "github_data": github_data,
            "scanners_complete": 1,
            "errors": [f"github_node: {e}"],
        }

    return {"github_data": github_data, "scanners_complete": 1}
