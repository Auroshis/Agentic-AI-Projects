# server.py
from fastmcp import FastMCP

mcp = FastMCP("Learning HTTP MCP")

# ---------------- TOOLS ----------------
@mcp.tool()
def add(a: int, b: int) -> int:
    return a + b

@mcp.tool()
def greet(name: str) -> str:
    return f"Hello {name}"

# ---------------- RESOURCES ----------------
@mcp.resource("config://app-info")
def app_info():
    return {
        "name": "Learning MCP Server",
        "version": "1.0.0"
    }

# ---------------- PROMPTS ----------------
@mcp.prompt()
def summarize(text: str) -> str:
    return f"Summarize the following text clearly:\n\n{text}"

# ---------------- RUN HTTP ----------------
if __name__ == "__main__":
    mcp.run(transport="http", port=8000)