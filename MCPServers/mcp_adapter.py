import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

async def main():

    client = MultiServerMCPClient(
        {
            "local": {
                "url": "http://localhost:8000/mcp",
                "transport": "http",
            }
        }
    )

    tools = await client.get_tools()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    agent = create_react_agent(llm, tools)

    result = await agent.ainvoke(
        {"messages": [("user", "Add 5 and 7 using the available tool")]}
    )

    print(result)


asyncio.run(main())