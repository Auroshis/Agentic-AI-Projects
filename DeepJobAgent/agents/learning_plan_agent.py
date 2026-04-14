"""Learning plan agent — generates a structured week-by-week plan for skill gaps."""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from DeepJobAgent.state import DeepJobState

_SYSTEM_PROMPT = """You are a senior technical mentor creating personalised learning plans.

Given a list of skill gaps and the candidate's current profile, create a realistic,
actionable week-by-week learning plan.

Use the available tools to structure the plan properly.

Return ONLY valid JSON (no markdown fences):
{
  "total_weeks": 12,
  "priority_order": ["skill1", "skill2", ...],
  "weekly_plan": [
    {
      "week": 1,
      "focus_skills": ["Kubernetes basics"],
      "daily_commitment_hours": 2,
      "resources": [
        {"type": "course", "title": "...", "platform": "...", "url": "...", "hours": 10}
      ],
      "milestones": ["Deploy a pod locally", "Understand deployments vs statefulsets"],
      "project": "Deploy a simple Flask app on a local k8s cluster"
    }
  ],
  "recommended_resources": [
    {"skill": "MLOps", "type": "book|course|docs|practice", "title": "...", "platform": "...", "estimated_hours": 20}
  ],
  "plan_summary": "Overview paragraph of the learning journey"
}

Prioritise missing_skills that appear in the JD's required (not nice-to-have) list.
Group related skills (e.g., Docker + Kubernetes) in the same weeks.
Be realistic: assume 2 hours/day on weekdays, 4 hours on weekends (~18h/week).
"""


@tool
def prioritize_skills(missing_skills: list, partial_skills: list, jd_keywords: str) -> dict:
    """
    Prioritize which skills to learn first based on JD importance and learning dependencies.
    Returns an ordered list with rationale.
    """
    return {
        "missing_skills": missing_skills,
        "partial_skills": partial_skills,
        "jd_context": jd_keywords,
        "instruction": (
            "Rank skills by: 1) How explicitly required vs nice-to-have in the JD "
            "2) Learning dependencies (learn Docker before Kubernetes) "
            "3) Time to learn (quick wins first to build momentum)"
        )
    }


@tool
def estimate_learning_time(skill: str, current_level: str = "beginner") -> dict:
    """
    Estimate how many hours a candidate needs to reach job-ready proficiency in a skill.
    current_level: 'none', 'beginner', 'intermediate'
    """
    time_estimates = {
        # (none, beginner, intermediate) hours to job-ready
        "docker":             (20, 10, 5),
        "kubernetes":         (40, 25, 15),
        "mlops":              (80, 50, 30),
        "system design":      (60, 40, 20),
        "aws":                (60, 35, 20),
        "terraform":          (30, 18, 10),
        "kafka":              (35, 20, 12),
        "fastapi":            (15, 8,  3),
        "pytorch":            (40, 25, 15),
        "tensorflow":         (40, 25, 15),
        "react":              (50, 30, 15),
        "typescript":         (25, 12, 5),
        "postgresql":         (30, 15, 8),
        "redis":              (20, 10, 5),
        "graphql":            (20, 12, 6),
    }
    key = skill.lower()
    level_idx = {"none": 0, "beginner": 1, "intermediate": 2}.get(current_level, 0)
    estimates = time_estimates.get(key, (30, 20, 10))
    hours = estimates[level_idx]
    weeks = max(1, hours // 18)   # assuming 18 hours/week available
    return {
        "skill": skill,
        "estimated_hours": hours,
        "estimated_weeks": weeks,
        "current_level": current_level,
    }


def build_learning_plan_agent(llm):
    return create_react_agent(
        model=llm,
        tools=[prioritize_skills, estimate_learning_time],
        prompt=_SYSTEM_PROMPT,
    )


def _build_plan_prompt(state: DeepJobState) -> str:
    gap = state.get("skill_gap") or {}
    return f"""Create a learning plan for this candidate:

JOB TARGET:
{state['job_description'][:500]}...

GAP ANALYSIS RESULTS:
  - Gap score: {gap.get('gap_score', 0):.0%}
  - Missing skills: {gap.get('missing_skills', [])}
  - Partial skills: {gap.get('partial_skills', [])}
  - Strong matches: {gap.get('strong_matches', [])}
  - Experience gap: {gap.get('experience_gap', 'none')}
  - Summary: {gap.get('analysis_summary', '')}

Create a week-by-week learning plan. Use prioritize_skills and estimate_learning_time tools
for the top missing skills, then build the full plan JSON.
"""


async def learning_plan_node(state: DeepJobState) -> dict:
    """LangGraph node: creates a learning plan and writes learning_plan."""
    from langchain_openai import ChatOpenAI
    from DeepJobAgent.config import LLM_MODEL, LLM_TEMPERATURE

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = build_learning_plan_agent(llm)

    prompt = _build_plan_prompt(state)

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        raw = result["messages"][-1].content
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        learning_plan = json.loads(clean)
    except json.JSONDecodeError:
        learning_plan = {
            "total_weeks": 0, "priority_order": [], "weekly_plan": [],
            "recommended_resources": [], "plan_summary": raw
        }
    except Exception as e:
        return {"errors": [f"learning_plan_node: {e}"]}

    return {"learning_plan": learning_plan}
