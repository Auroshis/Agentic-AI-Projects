"""
DeepJobAgent — Main LangGraph workflow.

Architecture:
                          ┌─ github_scanner ──────┐
  START ──────────────────┼─ leetcode_scanner ─────┼─► aggregate ─► gap_analysis ─► learning_plan ─► resume_tuner ─► END
                          ├─ linkedin_scanner ─────┤
                          └─ google_docs_scanner ──┘

The 4 scanner nodes fan out from START in parallel (LangGraph Pregel step 1).
All 4 feed into `aggregate`, which acts as a barrier (LangGraph waits for all
nodes in step 1 to finish before running step 2).
The analysis pipeline (gap → plan → tuner) runs sequentially after aggregation.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from DeepJobAgent.state import DeepJobState
from DeepJobAgent.agents.github_agent import github_node
from DeepJobAgent.agents.leetcode_agent import leetcode_node
from DeepJobAgent.agents.linkedin_agent import linkedin_node
from DeepJobAgent.agents.google_docs_agent import google_docs_node
from DeepJobAgent.agents.gap_analysis_agent import gap_analysis_node
from DeepJobAgent.agents.learning_plan_agent import learning_plan_node
from DeepJobAgent.agents.resume_tuner_agent import resume_tuner_node


def aggregate_node(state: DeepJobState) -> dict:
    """
    Barrier node — waits for all 4 scanners to complete.
    Validates that at least 2 sources returned data; logs warnings otherwise.
    """
    sources = {
        "github":      state.get("github_data"),
        "leetcode":    state.get("leetcode_data"),
        "linkedin":    state.get("linkedin_data"),
        "google_docs": state.get("google_docs_data"),
    }
    available = [name for name, data in sources.items() if data and not data.get("error")]
    failed    = [name for name, data in sources.items() if not data or data.get("error")]

    warnings = []
    if failed:
        warnings.append(f"Data missing or errored from: {failed}. Gap analysis will proceed with available sources.")

    # Return warnings as errors (additive reducer will append them)
    return {"errors": warnings} if warnings else {}


def build_graph() -> StateGraph:
    """Assemble and compile the DeepJobAgent LangGraph."""
    builder = StateGraph(DeepJobState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("github_scanner",      github_node)
    builder.add_node("leetcode_scanner",    leetcode_node)
    builder.add_node("linkedin_scanner",    linkedin_node)
    builder.add_node("google_docs_scanner", google_docs_node)
    builder.add_node("aggregate",           aggregate_node)
    builder.add_node("gap_analysis",        gap_analysis_node)
    builder.add_node("plan_generator",      learning_plan_node)
    builder.add_node("resume_tuner",        resume_tuner_node)

    # ── Fan-out: START → all 4 scanners (parallel) ────────────────────────────
    builder.add_edge(START, "github_scanner")
    builder.add_edge(START, "leetcode_scanner")
    builder.add_edge(START, "linkedin_scanner")
    builder.add_edge(START, "google_docs_scanner")

    # ── Fan-in: all 4 scanners → aggregate (barrier) ──────────────────────────
    builder.add_edge("github_scanner",      "aggregate")
    builder.add_edge("leetcode_scanner",    "aggregate")
    builder.add_edge("linkedin_scanner",    "aggregate")
    builder.add_edge("google_docs_scanner", "aggregate")

    # ── Sequential analysis pipeline ──────────────────────────────────────────
    builder.add_edge("aggregate",       "gap_analysis")
    builder.add_edge("gap_analysis",    "plan_generator")
    builder.add_edge("plan_generator",  "resume_tuner")
    builder.add_edge("resume_tuner",    END)

    return builder.compile()


# Compile once at import time
graph = build_graph()
