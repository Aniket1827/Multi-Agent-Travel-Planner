from typing import Any, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str
    llm_calls: int

    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str