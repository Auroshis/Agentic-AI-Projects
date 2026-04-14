import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY", "")
GITHUB_TOKEN        = os.getenv("GITHUB_TOKEN", "")          # optional, raises rate limit
GOOGLE_CREDS_PATH   = os.getenv("GOOGLE_CREDS_PATH", "credentials.json")
LANGSMITH_API_KEY   = os.getenv("LANGSMITH_API_KEY", "")
LANGCHAIN_TRACING   = os.getenv("LANGCHAIN_TRACING_V2", "false")

LLM_MODEL           = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE     = float(os.getenv("LLM_TEMPERATURE", "0"))

GITHUB_API_BASE     = "https://api.github.com"
LEETCODE_GQL_URL    = "https://leetcode.com/graphql"

# Set LangSmith env vars if provided
if LANGSMITH_API_KEY:
    os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "DeepJobAgent"
