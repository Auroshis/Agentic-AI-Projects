"""Resume tuner agent — rewrites resume sections to match the JD."""

from __future__ import annotations

import json
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from DeepJobAgent.state import DeepJobState

_SYSTEM_PROMPT = """You are an expert resume coach and technical recruiter who optimises resumes
for ATS (Applicant Tracking Systems) and human reviewers.

Your job:
1. Use analyse_jd_keywords to extract high-value ATS keywords from the JD
2. Use rewrite_experience_bullet to improve individual bullet points
3. Use craft_summary to write a targeted professional summary
4. Return the fully tuned resume as structured JSON

Guidelines:
- Use strong action verbs: "Architected", "Engineered", "Reduced", "Increased", "Led"
- Quantify achievements where possible: "Reduced latency by 40%"
- Mirror JD language exactly for ATS matching
- Keep bullets concise: max 2 lines each
- Highlight the strong_matches skills prominently

Return ONLY valid JSON (no markdown fences):
{
  "candidate_name": "",
  "tuned_sections": {
    "summary": "Results-driven Software Engineer with 5 years...",
    "skills": "Python | FastAPI | Docker | Kubernetes | PostgreSQL | Redis",
    "experience": [
      {
        "title": "...", "company": "...", "duration": "...",
        "bullets": ["• Engineered X resulting in Y", "• Led team of Z to deliver..."]
      }
    ],
    "education": "...",
    "projects": "..."
  },
  "ats_keywords_added": ["MLOps", "CI/CD", "distributed systems"],
  "cover_letter_snippet": "Opening paragraph for a cover letter tailored to this JD",
  "tuning_notes": "What was changed and why — useful for the candidate to understand the edits"
}
"""


@tool
def analyse_jd_keywords(job_description: str) -> dict:
    """
    Extract high-value ATS keywords and phrases from a job description.
    Returns must-use keywords, action verbs used in the JD, and domain terms.
    """
    return {
        "jd": job_description,
        "instruction": (
            "Extract: "
            "1) Technical skills explicitly named "
            "2) Soft skills and qualifications mentioned "
            "3) Company/domain-specific terminology "
            "4) Verbs used in 'you will' / 'responsibilities' sections "
            "These become the keywords to weave into the resume."
        )
    }


@tool
def rewrite_experience_bullet(
    original_bullet: str,
    target_keywords: list,
    role_context: str,
) -> str:
    """
    Rewrite a single resume bullet point to:
    - Start with a strong action verb
    - Include measurable impact if inferrable
    - Naturally incorporate target keywords
    - Stay under 2 lines

    Returns the rewritten bullet as a string starting with '•'.
    """
    return (
        f"Rewrite this bullet: '{original_bullet}' "
        f"for a {role_context} role, incorporating these keywords: {target_keywords}. "
        "Use a strong action verb and include impact if inferrable."
    )


@tool
def craft_summary(
    candidate_name: str,
    years_experience: int,
    strong_skills: list,
    target_role: str,
    company_name: str = "",
) -> str:
    """
    Craft a 3-4 sentence professional summary targeting a specific role.
    """
    company_part = f" at {company_name}" if company_name else ""
    return (
        f"Write a 3-4 sentence professional summary for {candidate_name}, "
        f"a {years_experience}-year professional with expertise in {', '.join(strong_skills[:5])}, "
        f"targeting the {target_role} role{company_part}. "
        "Be specific, quantified where possible, and ATS-optimised."
    )


def build_resume_tuner_agent(llm):
    return create_react_agent(
        model=llm,
        tools=[analyse_jd_keywords, rewrite_experience_bullet, craft_summary],
        prompt=_SYSTEM_PROMPT,
    )


def _build_tuner_prompt(state: DeepJobState) -> str:
    gap = state.get("skill_gap") or {}
    gd  = state.get("google_docs_data") or {}
    li  = state.get("linkedin_data") or {}

    # Build original resume sections from Google Docs data
    original_sections = gd.get("sections") or {}
    original_experience = gd.get("experience") or li.get("experience") or []

    return f"""Tune this resume for the following job description:

JOB DESCRIPTION:
{state['job_description']}

CANDIDATE STRONG MATCHES:
{gap.get('strong_matches', [])}

ATS KEYWORDS TO ADD:
{gap.get('missing_skills', [])[:8]}  (weave these in naturally where truthful)

ORIGINAL RESUME SECTIONS:
{json.dumps(original_sections, indent=2)}

ORIGINAL EXPERIENCE:
{json.dumps(original_experience, indent=2)}

CANDIDATE SKILLS:
{gd.get('skills', []) or li.get('skills', [])}

Use analyse_jd_keywords on the JD first, then tune each section.
Return the fully rewritten resume JSON.
"""


async def resume_tuner_node(state: DeepJobState) -> dict:
    """LangGraph node: rewrites resume to match JD and writes tuned_resume."""
    from langchain_openai import ChatOpenAI
    from DeepJobAgent.config import LLM_MODEL, LLM_TEMPERATURE

    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    agent = build_resume_tuner_agent(llm)

    prompt = _build_tuner_prompt(state)

    try:
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        raw = result["messages"][-1].content
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        tuned_resume = json.loads(clean)
    except json.JSONDecodeError:
        tuned_resume = {
            "candidate_name": "", "tuned_sections": {},
            "ats_keywords_added": [], "cover_letter_snippet": "",
            "tuning_notes": raw
        }
    except Exception as e:
        return {"errors": [f"resume_tuner_node: {e}"]}

    return {"tuned_resume": tuned_resume}
