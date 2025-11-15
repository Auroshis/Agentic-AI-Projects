import os
import traceback
import logging
from typing import TypedDict, List, Any

from langgraph.graph import StateGraph, START, END
from gemini_client import GeminiClient
from github_mcp_client import GitHubMCPClient


# ---------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEBUG = True  # Turn off in production


# ---------------------------------------------------------
# State Schema
# ---------------------------------------------------------
class InterviewAgentState(TypedDict):
    job_description: str
    resume_text: str
    github_user: str

    worked_topics: List[str]
    resume_skills: List[str]
    required_topics: List[str]
    missing_topics: List[str]
    overlap_topics: List[str]

    learning_plan: str
    tuned_resume: str


gemini = GeminiClient()


# ---------------------------------------------------------
# Helper: error wrapper
# ---------------------------------------------------------
async def safe_run(node_name: str, fn, fallback: Any):
    """Executes any async node function `fn` with error handling.
    `fn` should be a callable (no args) that returns a coroutine when called.
    """
    try:
        logger.info(f"Running node: {node_name}")
        # fn is a function; call it to get a coroutine and await it
        return await fn()
    except Exception as e:
        logger.error(f"[{node_name}] FAILED: {str(e)}")
        if DEBUG:
            traceback.print_exc()
        return fallback


def safe_run_sync(node_name: str, fn, fallback: Any):
    """Executes sync nodes with error handling."""
    try:
        logger.info(f"Running node: {node_name}")
        return fn()
    except Exception as e:
        logger.error(f"[{node_name}] FAILED: {str(e)}")
        if DEBUG:
            traceback.print_exc()
        return fallback


# ---------------------------------------------------------
# Node Implementations
# ---------------------------------------------------------

# ------------------ GITHUB ANALYSIS ----------------------
# ------------------ GITHUB ANALYSIS ----------------------
async def github_analysis_node(state: InterviewAgentState):

    async def run():
        user = state["github_user"]

        gh = GitHubMCPClient(token=os.getenv("GITHUB_TOKEN"))

        await gh.connect()

        try:
            # GitHub MCP tool: repos.list_repos_for_user
            repos = await gh.call(
                "repos.list_repos_for_user",
                username=user
            )

            repo_topics = []
            languages = []

            for repo in repos.get("repositories", []):
                name = repo["name"]

                # MCP: repos.list_languages
                langs = await gh.call(
                    "repos.list_languages",
                    owner=user,
                    repo=name
                )
                languages.extend(list(langs.get("languages", {}).keys()))

                # MCP: repos.get_topics
                topics = await gh.call(
                    "repos.get_topics",
                    owner=user,
                    repo=name
                )
                repo_topics.extend(topics.get("names", []))

            return {
                "worked_topics": sorted(
                    list(set(repo_topics + languages))
                )
            }

        finally:
            await gh.close()

    return await safe_run(
        "github_analysis",
        run,
        fallback={"worked_topics": []}
    )


    async def run():
        user = state["github_user"]
        gh = GitHubMCP()
        await gh.connect()

        try:
            repos = await gh.call("listRepos", username=user)

            repo_topics = []
            languages = []

            for repo in repos.get("repos", []):
                name = repo["name"]

                langs = await gh.call("getRepoLanguages",
                                      owner=user, repo=name)
                languages.extend(list(langs.get("languages", {}).keys()))

                topics = await gh.call("getRepoTopics",
                                       owner=user, repo=name)
                repo_topics.extend(topics.get("topics", []))

            return {
                "worked_topics": sorted(list(set(repo_topics + languages)))
            }

        finally:
            await gh.close()

    return await safe_run(
        "github_analysis",
        run,
        fallback={"worked_topics": []}
    )



# ------------------ RESUME SKILL EXTRACTION ----------------------
async def resume_expert_node(state: InterviewAgentState):

    async def run():
        resume = state["resume_text"]
        prompt = f"""
Extract concise skills (max 40).
One skill per line, max 5 words.

Resume:
{resume}
"""
        result = await gemini.generate(prompt)
        skills = [s.strip() for s in result.split("\n") if s.strip()]
        return {"resume_skills": skills}

    # pass function
    return await safe_run(
        "resume_expert",
        run,
        fallback={"resume_skills": []}
    )


# ------------------ JD EXTRACTION ----------------------
async def jd_extraction_node(state):

    async def run():
        jd = state["job_description"]

        prompt = f"""
Extract key required topics from the JD.
One short phrase per line.
No explanations.

{jd}
"""
        result = await gemini.generate(prompt)
        topics = [t.strip() for t in result.split("\n") if t.strip()]
        return {"required_topics": topics}

    # FIX: pass function, not coroutine object
    return await safe_run(
        "jd_extraction",
        run,  # <-- FIX
        fallback={"required_topics": []}
    )


# ------------------ GAP ANALYSIS ----------------------
def gap_analysis_node(state: InterviewAgentState):

    def run():
        worked = set(state.get("worked_topics", [])) | set(state.get("resume_skills", []))
        required = set(state.get("required_topics", []))

        missing = list(required - worked)
        overlap = list(required & worked)

        return {
            "missing_topics": missing,
            "overlap_topics": overlap
        }

    return safe_run_sync(
        "gap_analysis",
        run,
        fallback={"missing_topics": [], "overlap_topics": []}
    )


# ------------------ LEARNING PLAN ----------------------
async def learning_plan_node(state: InterviewAgentState):

    async def run():
        prompt = f"""
Create a 6-step learning plan.
Each step ≤ 10 words.

Topics: {state.get("missing_topics", [])}
"""
        plan = await gemini.generate(prompt)
        return {"learning_plan": plan}

    return await safe_run(
        "learning_plan",
        run,
        fallback={"learning_plan": "Unable to generate learning plan due to an error."}
    )


# ------------------ RESUME TUNING ----------------------
async def resume_tuning_node(state: InterviewAgentState):

    async def run():
        prompt = f"""
Rewrite resume focusing on strengths: {state.get("overlap_topics", [])}.
Short bullets (≤ 15 words).
Only output revised resume.

Job Description:
{state.get('job_description', '')}

Resume:
{state.get('resume_text', '')}
"""
        tuned = await gemini.generate(prompt)
        return {"tuned_resume": tuned}

    return await safe_run(
        "resume_tuning",
        run,
        fallback={"tuned_resume": state.get("resume_text", "")}
    )


# ---------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------
def build_graph():
    logger.info("Building graph...")

    graph = StateGraph(InterviewAgentState)

    graph.add_node("github_analysis", github_analysis_node)
    graph.add_node("resume_expert", resume_expert_node)
    graph.add_node("jd_extraction", jd_extraction_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("learning_plan", learning_plan_node)
    graph.add_node("resume_tuning", resume_tuning_node)

    graph.add_edge(START, "github_analysis")
    graph.add_edge(START, "resume_expert")
    graph.add_edge(START, "jd_extraction")

    graph.add_edge("github_analysis", "gap_analysis")
    graph.add_edge("resume_expert", "gap_analysis")
    graph.add_edge("jd_extraction", "gap_analysis")

    graph.add_edge("gap_analysis", "learning_plan")
    graph.add_edge("gap_analysis", "resume_tuning")

    graph.add_edge("learning_plan", END)
    graph.add_edge("resume_tuning", END)

    compiled = graph.compile()
    logger.info("Graph compiled successfully.")
    return compiled
