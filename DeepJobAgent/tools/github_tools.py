"""GitHub REST API tools for the GitHub scanner sub-agent."""

import requests
from langchain_core.tools import tool

from DeepJobAgent.config import GITHUB_API_BASE, GITHUB_TOKEN


def _headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


@tool
def get_github_profile(username: str) -> dict:
    """Fetch a GitHub user's public profile: bio, repo count, followers, location."""
    r = requests.get(f"{GITHUB_API_BASE}/users/{username}", headers=_headers(), timeout=10)
    if r.status_code != 200:
        return {"error": f"GitHub API {r.status_code}: {r.text[:200]}"}
    d = r.json()
    return {
        "name": d.get("name"),
        "bio": d.get("bio"),
        "public_repos": d.get("public_repos"),
        "followers": d.get("followers"),
        "following": d.get("following"),
        "location": d.get("location"),
        "blog": d.get("blog"),
        "company": d.get("company"),
        "hireable": d.get("hireable"),
    }


@tool
def get_github_repos(username: str, limit: int = 50) -> list:
    """
    Fetch a user's public repositories sorted by most recently updated.
    Returns name, description, language, topics, star count.
    """
    params = {"sort": "updated", "per_page": min(limit, 100), "type": "owner"}
    r = requests.get(
        f"{GITHUB_API_BASE}/users/{username}/repos",
        headers=_headers(),
        params=params,
        timeout=10,
    )
    if r.status_code != 200:
        return [{"error": f"GitHub API {r.status_code}: {r.text[:200]}"}]

    return [
        {
            "name": repo["name"],
            "description": repo.get("description") or "",
            "language": repo.get("language"),
            "topics": repo.get("topics", []),
            "stars": repo["stargazers_count"],
            "forks": repo["forks_count"],
            "url": repo["html_url"],
            "updated_at": repo["updated_at"][:10],
        }
        for repo in r.json()
    ]


@tool
def get_github_languages(username: str) -> dict:
    """
    Aggregate programming languages used across all public repos.
    Returns a dict mapping language → number of repos using it, sorted descending.
    """
    params = {"per_page": 100, "type": "owner"}
    r = requests.get(
        f"{GITHUB_API_BASE}/users/{username}/repos",
        headers=_headers(),
        params=params,
        timeout=10,
    )
    if r.status_code != 200:
        return {"error": f"GitHub API {r.status_code}"}

    counts: dict[str, int] = {}
    for repo in r.json():
        lang = repo.get("language")
        if lang:
            counts[lang] = counts.get(lang, 0) + 1

    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
