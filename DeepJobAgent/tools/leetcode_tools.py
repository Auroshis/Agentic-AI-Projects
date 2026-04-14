"""LeetCode GraphQL tools — uses the public API, no auth required."""

import httpx
from langchain_core.tools import tool

from DeepJobAgent.config import LEETCODE_GQL_URL

_HEADERS = {"Content-Type": "application/json", "Referer": "https://leetcode.com"}


@tool
def get_leetcode_stats(username: str) -> dict:
    """
    Fetch problem-solving statistics for a LeetCode user:
    total solved, breakdown by difficulty (Easy/Medium/Hard), and global ranking.
    """
    query = """
    query userPublicProfile($username: String!) {
        matchedUser(username: $username) {
            username
            profile { ranking }
            submitStats: submitStatsGlobal {
                acSubmissionNum {
                    difficulty
                    count
                    submissions
                }
            }
        }
    }
    """
    try:
        r = httpx.post(
            LEETCODE_GQL_URL,
            json={"query": query, "variables": {"username": username}},
            headers=_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    user = r.json().get("data", {}).get("matchedUser")
    if not user:
        return {"error": f"User '{username}' not found on LeetCode"}

    breakdown = {}
    for item in user.get("submitStats", {}).get("acSubmissionNum", []):
        breakdown[item["difficulty"].lower()] = item["count"]

    return {
        "username": username,
        "ranking": user.get("profile", {}).get("ranking"),
        "total_solved": breakdown.get("all", 0),
        "easy_solved": breakdown.get("easy", 0),
        "medium_solved": breakdown.get("medium", 0),
        "hard_solved": breakdown.get("hard", 0),
    }


@tool
def get_leetcode_topics(username: str) -> dict:
    """
    Fetch the topic/skill tags a LeetCode user has solved problems in.
    Returns advanced, intermediate, and fundamental topics with problem counts.
    """
    query = """
    query skillStats($username: String!) {
        matchedUser(username: $username) {
            tagProblemCounts {
                advanced      { tagName problemsSolved }
                intermediate  { tagName problemsSolved }
                fundamental   { tagName problemsSolved }
            }
        }
    }
    """
    try:
        r = httpx.post(
            LEETCODE_GQL_URL,
            json={"query": query, "variables": {"username": username}},
            headers=_HEADERS,
            timeout=15,
        )
        r.raise_for_status()
    except Exception as e:
        return {"error": str(e)}

    user = r.json().get("data", {}).get("matchedUser") or {}
    tags = user.get("tagProblemCounts") or {}

    def _extract(lst):
        return [{"tag": t["tagName"], "solved": t["problemsSolved"]} for t in (lst or [])]

    return {
        "advanced": _extract(tags.get("advanced")),
        "intermediate": _extract(tags.get("intermediate")),
        "fundamental": _extract(tags.get("fundamental")),
    }
