import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools.flight_tool import AVIATION_STACK_API_KEY

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_search_tool = None

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },
    }
)

async def get_all_tools():
    tools = await client.get_tools()
    print("Available Tools:")

    for tool in tools:
        print(f"{tool.name}")

async def get_tavily_search_tool():
    global tavily_search_tool

    if tavily_search_tool is not None:
        return tavily_search_tool
    
    tools = await client.get_tools()

    tavily_search_tool = next(tool for tool in tools if tool.name == "tavily_search")

async def tavily_mcp_search(query: str) -> str:
    await get_tavily_search_tool()

    result = await tavily_search_tool.ainvoke({"query": query})
    return result