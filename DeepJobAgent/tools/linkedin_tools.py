"""
LinkedIn profile tools.

LinkedIn actively blocks scraping. This module implements a three-tier
fallback strategy:
  1. curl_cffi Chrome impersonation (best chance for public data)
  2. Plain requests + BeautifulSoup (catches simple cases)
  3. Manual input mode — user pastes their profile summary

The agent will automatically choose the right tier based on what succeeds.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

try:
    from curl_cffi import requests as cffi_requests
    HAS_CURL_CFFI = True
except ImportError:
    HAS_CURL_CFFI = False

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _extract_with_bs4(html: str) -> dict:
    """Parse LinkedIn public profile HTML with BeautifulSoup."""
    if not HAS_BS4:
        return {}

    soup = BeautifulSoup(html, "html.parser")

    def _text(sel, attr=None):
        el = soup.select_one(sel)
        if not el:
            return ""
        return el.get(attr, "") if attr else el.get_text(strip=True)

    name = _text("h1")
    headline = _text(".top-card-layout__headline") or _text("[data-test-id='hero-headline']")

    # Skills — LinkedIn embeds a JSON-LD block on public profiles
    skills = []
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "{}")
            if isinstance(data, dict):
                skills_raw = data.get("knowsAbout", [])
                if skills_raw:
                    skills = [s if isinstance(s, str) else s.get("name", "") for s in skills_raw]
        except json.JSONDecodeError:
            pass

    return {
        "name": name,
        "headline": headline,
        "skills": skills,
        "current_role": "",
        "experience": [],
        "education": [],
    }


@tool
def scrape_linkedin_profile(url: str) -> dict:
    """
    Attempt to scrape a public LinkedIn profile URL.
    Returns structured profile data or an error with instructions for manual input.

    LinkedIn restricts scraping heavily — if automated scraping fails,
    the tool returns scrape_method='manual' so the agent can prompt the
    user to provide their profile data via parse_linkedin_manual().
    """
    html: Optional[str] = None

    # Tier 1: curl_cffi with Chrome TLS impersonation
    if HAS_CURL_CFFI:
        try:
            resp = cffi_requests.get(url, headers=_HEADERS, impersonate="chrome124", timeout=15)
            if resp.status_code == 200 and len(resp.text) > 2000:
                html = resp.text
        except Exception:
            pass

    # Tier 2: plain requests fallback
    if html is None:
        try:
            import requests
            resp = requests.get(url, headers=_HEADERS, timeout=10)
            if resp.status_code == 200 and len(resp.text) > 2000:
                html = resp.text
        except Exception:
            pass

    if html is None:
        return {
            "name": "", "headline": "", "current_role": "",
            "skills": [], "experience": [], "education": [],
            "scrape_method": "manual",
            "error": (
                "LinkedIn scraping is blocked. Please use parse_linkedin_manual() "
                "and paste your LinkedIn profile text (copy from LinkedIn > More > Save to PDF, "
                "or copy the About section + Experience section text)."
            ),
        }

    extracted = _extract_with_bs4(html)

    # If we got almost nothing useful, fall back to manual
    if not extracted.get("name") and not extracted.get("skills"):
        return {
            **extracted,
            "scrape_method": "manual",
            "error": (
                "Scraped HTML returned but could not parse structured data "
                "(LinkedIn may have served a login wall). "
                "Please use parse_linkedin_manual() with your profile text."
            ),
        }

    return {
        "name": extracted.get("name", ""),
        "headline": extracted.get("headline", ""),
        "current_role": extracted.get("current_role", ""),
        "skills": extracted.get("skills", []),
        "experience": extracted.get("experience", []),
        "education": extracted.get("education", []),
        "scrape_method": "curl_cffi" if HAS_CURL_CFFI else "requests",
        "error": None,
    }


@tool
def parse_linkedin_manual(profile_text: str) -> dict:
    """
    Accept a manually pasted LinkedIn profile summary and return structured data.
    Use this when automated scraping is blocked.

    profile_text should contain the user's About section, Experience entries,
    Education, and Skills — paste directly from LinkedIn or a PDF export.
    """
    if not profile_text or len(profile_text) < 50:
        return {
            "name": "", "headline": "", "current_role": "",
            "skills": [], "experience": [], "education": [],
            "scrape_method": "manual",
            "error": "Profile text is too short to extract meaningful data.",
        }

    return {
        "name": "",
        "headline": "",
        "current_role": "",
        "skills": [],
        "experience": [],
        "education": [],
        "scrape_method": "manual",
        "raw_text": profile_text,
        "error": None,
        "note": (
            "Raw text stored. The gap analysis agent will extract skills "
            "and experience directly from this text."
        ),
    }
