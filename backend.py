import os
from typing import Any
import certifi
import uuid
import asyncio
import json
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
    model = "openai/gpt-oss-120b",
    api_key = GROQ_API_KEY,
)

KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


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

def _llm_text(system_prompt: str, user_prompt: str):
    response = llm.invoke([
        SystemMessage(content = system_prompt),
        HumanMessage(content = user_prompt)
    ])
    return response.content

def _json_from_llm(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON found in the response")
    json_str = text[start:end + 1]
    return json.loads(json_str)

def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }

def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)
    GUARDRAIL_PROMPT = f"""
        Determine whether the following request belongs to travel planning or travel
        information. Valid requests can include destinations, flights, hotels, weather,
        budgets, visas, transportation, sightseeing, food, packing, or itineraries.

        Block clearly unrelated requests and requests asking for harmful or illegal
        instructions. Do not block a valid travel request merely because some details
        are missing.

        Return strict JSON only:
        {{
        "allowed": true,
        "reason": ""
        }}

        User request:
        {query}
    """

    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            GUARDRAIL_PROMPT,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as e:
        allowed = True
        guardrail_reason = "Guardrail validation failed. Proceeding with default settings."
        llm_calls += 1

    if not allowed:
        reason = guardrail_reason or (
            "I can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )

        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
        You are the supervisor of a multi-agent travel-planning system.
        Choose only the specialist agents needed for the request.

        Available agents:
        - flight_agent: flights, airports, airlines, routes, airfare, or booking advice
        - hotel_agent: hotels, accommodation, neighborhoods, or places to stay
        - weather_agent: weather, climate, season, forecast, or packing advice
        - budget_agent: cost, affordability, price limits, or budget feasibility
        - itinerary_agent: creates the integrated travel plan and must always be included

        Return strict JSON only using this schema:
        {{
        "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
        "trip_constraints": {{
            "destination": "",
            "origin": "",
            "duration": "",
            "budget": "",
            "travel_style": "",
            "special_preferences": []
        }},
        "reasoning": ""
        }}

        User request:
        {query}
    """
    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER if name in requested_agents and name in KNOWN_AGENTS
        ]

        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")
        
        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})

        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)
        
        reasoning = parsed.get("reasoning", "").strip()
        llm_calls += 1
    except Exception as e:
        selected_agents = AGENT_ORDER
        constraints = _empty_constraints()
        reasoning = "Supervisor validation failed. Proceeding with default settings."
    
    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }

def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or ("This request was blocked by the guardrail.")
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }

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

def budget_agent(state: TravelState):
    prompt = f"""
        Analyze whether this trip is realistic for the user's budget.

        User Query:
        {state['user_query']}

        Trip Constraints:
        {state.get('trip_constraints', {})}

        Flight Results:
        {state.get('flight_results', '')}

        Hotel Results:
        {state.get('hotel_results', '')}

        Weather Results:
        {state.get('weather_results', '')}

        Return:
        1. Estimated cost categories
        2. Budget risk areas
        3. Money-saving suggestions
        4. Overall feasibility

        If exact live prices are unavailable, clearly label estimates as approximate.
    """

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def itinerary_agent(state: TravelState):
    prompt = f"""
        Create a complete travel itinerary for the user.

        User Query:
        {state['user_query']}

        Flight Results:
        {state['flight_results'][:1500]}

        Hotel Results:
        {state['hotel_results'][:1500]}

        Weather:
        {state['weather_results'][:1500]}

        Budget:
        {state['budget_results'][:1500]}

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
# Budget Analysis:
#         {state['budget_results']}
def final_agent(state: TravelState):
    final_prompt = f"""
        Generate the final travel response for the user.

        User Request:
        {state['user_query']}

        Hotels:
        {state['hotel_results'][:1500]}

        Weather:
        {state['weather_results'][:1500]}

        Itinerary:
        {state["itinerary"][:1500]}

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

ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)
graph.add_conditional_edges("flight_agent", route_after_agent("flight_agent"), ROUTE_MAP)
graph.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP)
graph.add_conditional_edges("weather_agent", route_after_agent("weather_agent"), ROUTE_MAP)
graph.add_conditional_edges("budget_agent", route_after_agent("budget_agent"), ROUTE_MAP)

graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

graph.compile()

travel_graph = graph.compile(checkpointer=get_checkpointer())

def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message


    return {
        "thread_id": thread_id,
        "answer": answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": result.get("itinerary", ""),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }



def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)