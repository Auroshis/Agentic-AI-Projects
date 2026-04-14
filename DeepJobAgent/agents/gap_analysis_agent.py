"""Gap analysis agent — compares JD requirements against all scanner outputs."""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from DeepJobAgent.state import DeepJobState

_SYSTEM_PROMPT = """You are an expert technical recruiter and career coach performing a gap analysis.

You will receive a job description (JD) and aggregated profile data from GitHub, LeetCode,
LinkedIn, and the candidate's resume (Google Docs).

Your job:
1. Use extract_jd_requirements to parse required skills, experience, and qualifications from the JD.
2. Use aggregate_candidate_skills to merge all skill signals into one candidate profile.
3. Perform a detailed gap analysis comparing requirements vs candidate.
4. Return structured JSON output.

Return ONLY valid JSON (no markdown fences):
{
  "required_skills": ["Python", "MLOps", "System Design", ...],
  "candidate_skills": ["Python", "FastAPI", ...],
  "missing_skills": ["MLOps", "Kubernetes", ...],
  "partial_skills": ["System Design (some exposure)", ...],
  "strong_matches": ["Python", "REST APIs", ...],
  "gap_score": 0.65,
  "experience_gap": "JD requires 5 years, candidate has 3 years",
  "education_match": true,
  "analysis_summary": "Detailed paragraph explaining the overall fit, key gaps, and strengths"
}

gap_score = (strong_matches) / (required_skills) — a float between 0 and 1.
partial_skills entries should include context e.g. "Docker (basic, no orchestration)".
"""


@tool
def extract_jd_requirements(job_description: str) -> dict:
    """
    Parse a job description and extract structured requirements.
    Returns required skills, experience level, education, and keywords.
    """
    # This tool is intentionally simple — the LLM does the heavy lifting
    return {
        "raw_jd": job_description,
        "instruction": (
            "Parse this JD and identify: "
            "1) Must-have technical skills "
            "2) Nice-to-have skills "
            "3) Experience years required "
            "4) Education requirements "
            "5) Domain keywords (e.g., fintech, ML, distributed systems)"
        )
    }


@tool
def aggregate_candidate_skills(
    github_languages: list,
    github_topics: list,
    leetcode_topics: list,
    linkedin_skills: list,
    resume_skills: list,
    resume_technologies: list,
) -> dict:
    """
    Merge all skill signals from different sources into a deduplicated candidate profile.
    """
    all_skills = set()
    for source in [github_languages, github_topics, leetcode_topics,
                   linkedin_skills, resume_skills, resume_technologies]:
        for skill in source:
            if isinstance(skill, str) and skill.strip():
                all_skills.add(skill.strip())
            elif isinstance(skill, dict) and skill.get("tag"):
                all_skills.add(skill["tag"])

    return {
        "unified_skills": sorted(all_skills),
        "skill_count": len(all_skills),
        "sources": {
            "github_languages": github_languages,
            "github_topics": github_topics,
            "leetcode_topics": leetcode_topics,
            "linkedin_skills": linkedin_skills,
            "resume_skills": resume_skills,
        }
    }


def build_gap_analysis_agent(llm):
    return create_react_agent(
        model=llm,
        tools=[extract_jd_requirements, aggregate_candidate_skills],
        prompt=_SYSTEM_PROMPT,
    )


def _build_gap_prompt(state: DeepJobState) -> str:
    gh = state.get("github_data") or {}
    lc = state.get("leetcode_data") or {}
    li = state.get("linkedin_data") or {}
    gd = state.get("google_docs_data") or {}

    return f"""Perform a gap analysis with this information:

JOB DESCRIPTION:
{state['job_description']}

GITHUB PROFILE:
  - Languages: {gh.get('top_languages', [])}
  - Project topics: {gh.get('all_topics', [])}
  - Summary: {gh.get('highlights', 'N/A')}

LEETCODE PROFILE:
  - Total solved: {lc.get('total_solved', 0)} (E:{lc.get('easy_solved',0)} M:{lc.get('medium_solved',0)} H:{lc.get('hard_solved',0)})
  - Topics: {lc.get('topic_tags', [])}
  - Skill level: {lc.get('skill_level', 'unknown')}

LINKEDIN PROFILE:
  - Current role: {li.get('current_role', 'N/A')}
  - Skills: {li.get('skills', [])}
  - Experience: {[e.get('title') + ' at ' + e.get('company','') for e in li.get('experience', [])]}

RESUME (from Google Docs):
  - Skills: {gd.get('skills', [])}
  - Technologies: {gd.get('technologies', [])}
  - Experience years: {gd.get('experience_years', 'unknown')}
  - Roles: {[e.get('title') for e in gd.get('experience', [])]}

Use aggregate_candidate_skills and extract_jd_requirements tools, then produce the gap analysis JSON.
"""


async def gap_analysis_node(state: DeepJobState) -> dict:
    """LangGraph node: performs gap analysis and writes skill_gap."""
    from langchain_openai import ChatOpenAI
    from DeepJobAgent.config import LLM_MODEL, LLM_TEMPERATURE

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = build_gap_analysis_agent(llm)

    prompt = _build_gap_prompt(state)

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        raw = result["messages"][-1].content
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        skill_gap = json.loads(clean)
    except json.JSONDecodeError:
        skill_gap = {
            "required_skills": [], "candidate_skills": [],
            "missing_skills": [], "partial_skills": [], "strong_matches": [],
            "gap_score": 0.0, "experience_gap": "", "education_match": False,
            "analysis_summary": raw
        }
    except Exception as e:
        return {"errors": [f"gap_analysis_node: {e}"]}

    return {"skill_gap": skill_gap}
