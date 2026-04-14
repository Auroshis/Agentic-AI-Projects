"""
DeepJobAgent — Entry point.

Usage:
  python -m DeepJobAgent.main

Or with custom inputs:
  python DeepJobAgent/main.py

Set all credentials in your .env file first (see README section below).
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from typing import Optional

from DeepJobAgent.graph import graph
from DeepJobAgent.state import DeepJobState


# ── Sample job description (replace with your target JD) ─────────────────────
SAMPLE_JD = """
Senior Machine Learning Engineer — FinTech AI Platform

We are looking for a Senior ML Engineer to join our platform team.

Requirements:
- 5+ years of software engineering experience
- 3+ years of production ML experience
- Proficiency in Python (NumPy, Pandas, Scikit-learn, PyTorch or TensorFlow)
- Experience with MLOps: ML pipelines, model serving, monitoring (MLflow, Kubeflow, or similar)
- Familiarity with containerisation (Docker) and orchestration (Kubernetes)
- Experience with distributed data processing (Spark or Dask)
- Strong understanding of system design and distributed systems
- SQL and NoSQL database experience (PostgreSQL, Redis)
- CI/CD pipelines (GitHub Actions, Jenkins)

Nice to have:
- Experience with LLMs and LangChain/LangGraph
- Knowledge of financial domain (fraud detection, risk modelling)
- Experience with Kafka or similar streaming systems
- AWS/GCP cloud platform experience

You will:
- Design and deploy ML models to production serving 10M+ daily predictions
- Build and maintain ML pipelines and feature stores
- Collaborate with data scientists and platform engineers
- Drive MLOps best practices across the team
"""


async def run_agent(
    job_description: str,
    github_username: str,
    leetcode_username: str,
    linkedin_url: str,
    google_docs_id: str,
) -> DeepJobState:
    """Run the full DeepJobAgent pipeline and return the final state."""

    initial_state: DeepJobState = {
        "job_description":  job_description,
        "github_username":  github_username,
        "leetcode_username": leetcode_username,
        "linkedin_url":     linkedin_url,
        "google_docs_id":   google_docs_id,
        # Scanner outputs — start as None
        "github_data":      None,
        "leetcode_data":    None,
        "linkedin_data":    None,
        "google_docs_data": None,
        # Fan-in counter
        "scanners_complete": 0,
        # Analysis outputs
        "skill_gap":        None,
        "learning_plan":    None,
        "tuned_resume":     None,
        # Error log
        "errors":           [],
    }

    print("\n" + "═" * 60)
    print("  DeepJobAgent — Career Gap Analyser")
    print("═" * 60)
    print(f"  GitHub:    {github_username}")
    print(f"  LeetCode:  {leetcode_username}")
    print(f"  LinkedIn:  {linkedin_url}")
    print(f"  Resume:    Google Doc {google_docs_id[:20]}...")
    print("═" * 60)
    print("\n[1/4] Scanning all profiles in parallel...")

    final_state = await graph.ainvoke(initial_state)

    return final_state


def print_report(state: DeepJobState) -> None:
    """Pretty-print the final analysis report to stdout."""
    print("\n" + "═" * 60)
    print("  DEEP JOB AGENT — FINAL REPORT")
    print("═" * 60)

    # ── Gap Analysis ──────────────────────────────────────────────────────────
    gap = state.get("skill_gap") or {}
    print("\n📊  GAP ANALYSIS")
    print(f"  Match score:  {gap.get('gap_score', 0):.0%}")
    print(f"  Strong matches ({len(gap.get('strong_matches', []))}):")
    for s in gap.get("strong_matches", []):
        print(f"    ✓ {s}")
    print(f"\n  Missing skills ({len(gap.get('missing_skills', []))}):")
    for s in gap.get("missing_skills", []):
        print(f"    ✗ {s}")
    print(f"\n  Partial skills ({len(gap.get('partial_skills', []))}):")
    for s in gap.get("partial_skills", []):
        print(f"    ~ {s}")
    print(f"\n  Summary:")
    summary = gap.get("analysis_summary", "")
    for line in textwrap.wrap(summary, width=58):
        print(f"    {line}")

    # ── Learning Plan ─────────────────────────────────────────────────────────
    plan = state.get("learning_plan") or {}
    print(f"\n📚  LEARNING PLAN  ({plan.get('total_weeks', '?')} weeks)")
    print(f"  Priority order: {' → '.join(plan.get('priority_order', [])[:5])}")
    print()
    for week in (plan.get("weekly_plan") or [])[:4]:   # show first 4 weeks
        print(f"  Week {week.get('week', '?')}: {', '.join(week.get('focus_skills', []))}")
        for milestone in week.get("milestones", [])[:2]:
            print(f"    • {milestone}")
    if len(plan.get("weekly_plan", [])) > 4:
        print(f"  ... ({len(plan['weekly_plan']) - 4} more weeks)")

    # ── Tuned Resume ──────────────────────────────────────────────────────────
    resume = state.get("tuned_resume") or {}
    print(f"\n📝  TUNED RESUME")
    print(f"  ATS keywords added: {', '.join(resume.get('ats_keywords_added', []))}")
    print(f"\n  NEW PROFESSIONAL SUMMARY:")
    summary_text = resume.get("tuned_sections", {}).get("summary", "")
    for line in textwrap.wrap(summary_text, width=58):
        print(f"    {line}")

    cover = resume.get("cover_letter_snippet", "")
    if cover:
        print(f"\n  COVER LETTER OPENING:")
        for line in textwrap.wrap(cover, width=58):
            print(f"    {line}")

    print(f"\n  Tuning notes:")
    for line in textwrap.wrap(resume.get("tuning_notes", ""), width=58):
        print(f"    {line}")

    # ── Errors ────────────────────────────────────────────────────────────────
    errors = state.get("errors", [])
    if errors:
        print(f"\n⚠️   WARNINGS / ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    • {e}")

    print("\n" + "═" * 60)

    # Dump full JSON to a file for inspection
    with open("deepjob_report.json", "w") as f:
        # Convert state to serialisable form
        json.dump(
            {k: v for k, v in state.items() if v is not None},
            f, indent=2, default=str
        )
    print("  Full report saved to: deepjob_report.json")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    # ── Configure your inputs here ────────────────────────────────────────────
    GITHUB_USERNAME  = "torvalds"          # replace with your GitHub username
    LEETCODE_USERNAME = "neal_wu"          # replace with your LeetCode username
    LINKEDIN_URL     = "https://www.linkedin.com/in/williamhgates/"  # replace with yours
    GOOGLE_DOC_ID    = ""                  # paste the ID from your Google Docs resume URL

    final_state = asyncio.run(
        run_agent(
            job_description=SAMPLE_JD,
            github_username=GITHUB_USERNAME,
            leetcode_username=LEETCODE_USERNAME,
            linkedin_url=LINKEDIN_URL,
            google_docs_id=GOOGLE_DOC_ID,
        )
    )

    print_report(final_state)
