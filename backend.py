import os
import certifi
import uuid
import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from pycountry.db import Data
from requests import get
# from tools.tavily_tool import tavily_search
# from tools.flight_tool import search_flights
from state.travel_state import TravelState
from db.connect import get_checkpointer
from mcp_client import extract_destination, forecast_mcp_search, tavily_mcp_search, aviation_mcp_call, weather_mcp_search
from constants import FLIGHT_AGENT_PROMPT

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

# def flight_agent(state: TravelState):
#     query = state["user_query"]
#     flight_data = search_flights(query)

#     return {
#         "flight_results": flight_data,
#         "messages": [
#             AIMessage(content="Flight results fetched"),
#         ],
#         "llm_calls": state.get("llm_calls", 0) + 1
#     }

def flight_agent(state: TravelState):
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        prompt = FLIGHT_AGENT_PROMPT.format(
            query = query,
            airport_data = str(airports)[:1500],
            airline_data = str(airlines)[:1500]
        )

        response = llm.invoke([
            SystemMessage(content="You are a travel flight expert."),
            HumanMessage(content=prompt)
        ])

        return {
            "flight_results": response.content,
        }
    except Exception as e:
        return {
            "flight_results": f"Error fetching flight data: {e}",
            "messages": [
                AIMessage(content=f"Error fetching flight data: {e}"),
            ],
            "llm_calls": state.get("llm_calls", 0) + 1
        }

def hotel_agent(state: TravelState):
    query = f"Suggest beest hotels for {state['user_query']}"
    # hotel_data = tavily_search(query)
    hotel_data = asyncio.run(tavily_mcp_search(query))
    return {
        "hotel_results": hotel_data,
        "messages": [
            AIMessage(content="Hotel results fetched"),
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def weather_agent(state: TravelState):
    city = extract_destination(state["user_query"])
    weather_data = asyncio.run(weather_mcp_search(city))
    forecast_data = asyncio.run(forecast_mcp_search(city))
    return {
        "weather_results": f"""
            Current Weather:
            {weather_data}

            Forecast:
            {forecast_data}
        """,
        "messages": [
            AIMessage(content="Weather results fetched"),
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def itinerary_agent(state: TravelState):
    prompt = f"""
        Create a complete travel itinerary for the user.

        User Query:
        {state['user_query']}

        Flight Results:
        {state['flight_results']}

        Hotel Results:
        {state['hotel_results']}

        Weather:
        {state['weather_results']}

        Make the itinerary practical, budget-aware, and easy to follow.
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

# Flights:
#         {state['flight_results']}
def final_agent(state: TravelState):
    final_prompt = f"""
        Generate the final travel response for the user.

        User Request:
        {state['user_query']}


        Hotels:
        {state['hotel_results']}

        Weather:
        {state['weather_results']}

        Itinerary:
        {state["itinerary"]}

        Format the final answer beautifully using these sections:

        1. Trip Summary
        2. Flight Information
        3. Hotel Suggestions
        4. Weather Information
        5. Day-by-Day Itinerary
        6. Estimated Budget
        7. Final Recommendations

        Important:
        - Be clear and practical.
        - Mention that live flight API may not provide ticket prices if pricing is unavailable.
        - Keep the response useful for real travel planning.
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner. You are given a user request, flight results, hotel results and itinerary. You need to generate a final travel response for the user."),
        HumanMessage(content=final_prompt),
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

graph = StateGraph(TravelState)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)

graph.compile()

travel_graph = graph.compile(checkpointer=get_checkpointer())

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if thread_id is None:
        thread_id = f"user_{uuid.uuid4().hex}"
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke({
            "message": [
                HumanMessage(content=user_input)
            ], 
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itinerary": "",
            "llm_calls": 0
        }, config=config
    )

    final_answer = result["messages"][-1].content
    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0)
    }


