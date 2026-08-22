import os
import requests
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()

mcp =FastMCP("Weather MCP Server")

OPEN_WEATHER_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

@mcp.tool()
def get_current_weather(city: str):
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": city,
            "appid": OPEN_WEATHER_API_KEY,
            "units": "metric"
        }
    )
    response = response.json()
    if response.status_code == 200:
        return {
            "city": response["name"],
            "temperature_c": response["main"]["temp"],
            "feels_like_c": response["main"]["feels_like"],
            "humidity": response["main"]["humidity"],
            "condition": response["weather"][0]["description"],
            "wind_speed": response["wind"]["speed"],
        }
    else:
        return {
            "error": f"Failed to fetch weather data: {response.status_code} {response.text}"
        }

@mcp.tool()
def get_forecast(city: str):
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/forecast",
        params={
            "q": city,
            "appid": OPEN_WEATHER_API_KEY,
            "units": "metric"
        }
    )

    response = response.json()
    if response.status_code == 200:
        forecast = []
        for item in response["list"][:-5]:
            forecast.append({
                "datetime": item["dt_txt"],
                "temperature_c": item["main"]["temp"],
                "weather": item["weather"][0]["description"],
            })

        return {
            "city": city,
            "forecast": forecast,
        }

    else:
        return {
            "error": f"Failed to fetch forecast data: {response.status_code} {response.text}"
        }

if __name__ == "__main__":
    mcp.run()