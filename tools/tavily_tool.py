import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def tavily_search(query: str):
    """
    Search the web for information based on the query.
    """
    response = client.search(
        query=query,
        max_results=5,
    )

    results = []

    for i, r in enumerate(response["results"], 1):
        title = r.get("title", "Unknown Title")
        url = r.get("url", "Unknown URL")
        snippet = r.get("content", "Unknown Snippet").strip()

        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
        
        results.append(f"Result {i}:\nTitle: {title}\nURL: {url}\nSnippet: {snippet}")
    
    return "\n\n".join(results)