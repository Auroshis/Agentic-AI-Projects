# github_mcp_client.py
import httpx
from mcp import ClientSession

class GitHubMCPClient:
    def __init__(self, token: str | None = None):
        self.base_url = "https://api.githubcopilot.com/mcp/"
        self.token = token
        self.session: ClientSession | None = None

    async def connect(self):
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        self.session = await ClientSession.connect_http(
            url=self.base_url,
            headers=headers
        )
        return self.session

    async def call(self, tool: str, **params):
        if not self.session:
            raise RuntimeError("GitHub MCP not connected.")

        return await self.session.call_tool(
            name=tool,
            arguments=params
        )

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None