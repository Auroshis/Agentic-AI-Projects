from __future__ import annotations

import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict


class GithubProfile(TypedDict):
    username: str
    repos: list
    top_languages: list        # sorted by frequency
    all_topics: list           # deduplicated project topics
    error: Optional[str]


class LeetcodeProfile(TypedDict):
    username: str
    total_solved: int
    easy_solved: int
    medium_solved: int
    hard_solved: int
    topic_tags: list           # tags from solved problems
    ranking: Optional[int]
    error: Optional[str]


class LinkedinProfile(TypedDict):
    name: str
    headline: str
    current_role: str
    skills: list
    experience: list           # [{title, company, duration, description}]
    education: list            # [{degree, institution, year}]
    scrape_method: str         # "curl_cffi" | "bs4" | "manual"
    error: Optional[str]


class DocumentsData(TypedDict):
    sources: list              # [{type: 'pdf'|'gdoc', ref: str, title: str}]
    raw_text: str              # combined text from all sources
    sections: dict             # {"experience": "...", "skills": "...", ...}
    skills: list
    experience: list
    education: list
    error: Optional[str]


# Keep alias for any external code that references the old name
GoogleDocsResume = DocumentsData


class SkillGap(TypedDict):
    required_skills: list
    candidate_skills: list
    missing_skills: list
    partial_skills: list       # have some exposure but not proficiency
    strong_matches: list
    gap_score: float           # 0.0 (no match) – 1.0 (perfect)
    analysis_summary: str


class LearningPlan(TypedDict):
    total_weeks: int
    weekly_plan: list          # [{week, focus_skills, resources, milestones}]
    priority_order: list       # skills sorted by JD importance
    recommended_resources: list
    plan_summary: str


class TunedResume(TypedDict):
    original_sections: dict
    tuned_sections: dict       # rewritten sections
    ats_keywords_added: list
    cover_letter_snippet: str
    tuning_notes: str


class DeepJobState(TypedDict):
    # ── Inputs ───────────────────────────────────────────────────────────────
    job_description: str
    github_username: str
    leetcode_username: str
    linkedin_url: str
    pdf_path: str              # server-side path to uploaded PDF (empty if none)
    google_docs_ids: list      # list of Google Doc IDs or URLs

    # ── Scanner outputs (one writer per key, no reducer needed) ─────────────
    github_data: Optional[GithubProfile]
    leetcode_data: Optional[LeetcodeProfile]
    linkedin_data: Optional[LinkedinProfile]
    google_docs_data: Optional[DocumentsData]

    # ── Fan-in counter: each scanner writes +1 ───────────────────────────────
    scanners_complete: Annotated[int, operator.add]

    # ── Sequential pipeline outputs ──────────────────────────────────────────
    skill_gap: Optional[SkillGap]
    learning_plan: Optional[LearningPlan]
    tuned_resume: Optional[TunedResume]

    # ── Shared error log ─────────────────────────────────────────────────────
    errors: Annotated[list, operator.add]
