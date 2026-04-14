"""LeetCode scanner sub-agent."""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from DeepJobAgent.tools.leetcode_tools import get_leetcode_stats, get_leetcode_topics
from DeepJobAgent.state import DeepJobState

_SYSTEM_PROMPT = """You are a LeetCode profile analyst who extracts algorithmic and DSA skill signals.

Your job:
1. Call get_leetcode_stats to get problem counts by difficulty
2. Call get_leetcode_topics to get topic/tag breakdown
3. Synthesize into a JSON skill summary

Return ONLY valid JSON (no markdown fences) with this exact structure:
{
  "username": "<username>",
  "total_solved": 0,
  "easy_solved": 0,
  "medium_solved": 0,
  "hard_solved": 0,
  "ranking": null,
  "topic_tags": ["Array", "Dynamic Programming", "Graph", ...],
  "skill_level": "beginner|intermediate|advanced",
  "highlights": "2-3 sentence summary of their problem-solving strengths",
  "error": null
}

Derive skill_level from:
  beginner:     total_solved < 100 or mostly Easy
  intermediate: 100-300 solved with good Medium coverage
  advanced:     300+ solved with significant Hard problems

topic_tags should be ALL topics from advanced + intermediate categories (by solved count descending).
"""


def build_leetcode_agent(llm):
    return create_react_agent(
        model=llm,
        tools=[get_leetcode_stats, get_leetcode_topics],
        prompt=_SYSTEM_PROMPT,
    )


async def leetcode_node(state: DeepJobState) -> dict:
    """LangGraph node: runs the LeetCode scanner and writes leetcode_data."""
    from langchain_openai import ChatOpenAI
    from DeepJobAgent.config import LLM_MODEL, LLM_TEMPERATURE

    username = state.get("leetcode_username", "")
    if not username:
        return {
            "leetcode_data": {
                "username": "", "total_solved": 0, "easy_solved": 0,
                "medium_solved": 0, "hard_solved": 0, "ranking": None,
                "topic_tags": [], "skill_level": "unknown",
                "highlights": "", "error": "No LeetCode username provided"
            },
            "scanners_complete": 1,
            "errors": ["leetcode_node: no username provided"],
        }

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = build_leetcode_agent(llm)

    try:
        result = await agent.ainvoke({
            "messages": [HumanMessage(content=f"Analyze LeetCode profile for username: {username}")]
        })
        raw = result["messages"][-1].content
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        leetcode_data = json.loads(clean)
    except json.JSONDecodeError:
        leetcode_data = {
            "username": username, "total_solved": 0, "easy_solved": 0,
            "medium_solved": 0, "hard_solved": 0, "ranking": None,
            "topic_tags": [], "skill_level": "unknown",
            "highlights": raw, "error": "json_parse_failed"
        }
    except Exception as e:
        leetcode_data = {
            "username": username, "total_solved": 0, "easy_solved": 0,
            "medium_solved": 0, "hard_solved": 0, "ranking": None,
            "topic_tags": [], "skill_level": "unknown",
            "highlights": "", "error": str(e)
        }
        return {
            "leetcode_data": leetcode_data,
            "scanners_complete": 1,
            "errors": [f"leetcode_node: {e}"],
        }

    return {"leetcode_data": leetcode_data, "scanners_complete": 1}
