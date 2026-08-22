import os
import asyncio
import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient

from tools.flight_tool import AVIATION_STACK_API_KEY

load_dotenv()

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set")

llm = ChatGroq(
    model = "openai/gpt-oss-safeguard-20b",
    api_key = GROQ_API_KEY,
)


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPEN_WEATHER_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
tavily_search_tool = None
aviation_stack_tools = {}
weather_tool = None
forecast_tool = None

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"
        },
        "aviation_stack": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "--with",
                "mcp[cli]<2",
                "aviationstack-mcp"
            ],
            "env": {
                "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
            }
        },
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["tools/custom_weather_mcp_server.py"],
            "env": {
                "OPEN_WEATHER_API_KEY": OPEN_WEATHER_API_KEY
            }
        }
    }
)

async def get_all_tools():
    tools = await client.get_tools()
    print("Available Tools:")

    for tool in tools:
        print(f"{tool.name}")

async def initialize_weather_tool():
    global weather_tool, forecast_tool

    tools = await client.get_tools()

    weather_tool = next(tool for tool in tools if tool.name == "get_current_weather")
    forecast_tool = next(tool for tool in tools if tool.name == "get_forecast")


async def initialize_mcp():
    global tavily_search_tool
    global aviation_stack_tools

    if tavily_search_tool is not None:
        return tavily_search_tool
    
    tools = await client.get_tools()

    tavily_search_tool = next(tool for tool in tools if tool.name == "tavily_search")
    
    aviation_stack_tools = {
        tool.name: tool for tool in tools if tool.name != "tavily_search"
    }

async def tavily_mcp_search(query: str) -> str:
    await initialize_mcp()

    result = await tavily_search_tool.ainvoke({"query": query})
    return result

async def aviation_mcp_call(tool_name: str, tool_args:dict = None):
    tools = await client.get_tools()

    tool = next(tool for tool in tools if tool.name == tool_name)

    result = await tool.ainvoke(tool_args or {})
    return result

async def weather_mcp_search(city: str):
    await initialize_weather_tool()

    return await weather_tool.ainvoke({"city": city})

async def forecast_mcp_search(city: str):
    await initialize_weather_tool()

    return await forecast_tool.ainvoke({"city": city})

def extract_destination(query: str):
    prompt = f"""
        You are a travel expert. You are given a user's query.
        Extract the destination city from the user's query.
        The destination city is the city the user wants to visit.
        The destination city is usually mentioned in the query.
        The destination city is usually a city name.    
    """
    response = llm.invoke(prompt)
    return response.content.strip()