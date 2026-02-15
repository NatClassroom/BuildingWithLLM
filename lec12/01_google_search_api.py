# TOPIC: Google Custom Search API
# This demo shows how to use Google's Custom Search JSON API
# and what the response body looks like.
#
# Setup required:
# 1. Create a Custom Search Engine at https://programmablesearchengine.google.com/
# 2. Get API key from https://console.cloud.google.com/apis/credentials
# 3. Enable "Custom Search API" in Google Cloud Console
# 4. Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID in .env

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_SEARCH_ENGINE_ID")


def google_search(query: str, num_results: int = 5) -> dict:
    """
    Perform a Google Custom Search.

    Args:
        query: The search query string
        num_results: Number of results to return (max 10 per request)

    Returns:
        dict: The raw API response
    """
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": min(num_results, 10),  # API limit is 10 per request
    }

    response = requests.get(url, params=params)
    response.raise_for_status()

    return response.json()


def extract_search_results(raw_response: dict) -> list[dict]:
    """
    Extract clean search results from the raw API response.

    Args:
        raw_response: The raw response from Google Search API

    Returns:
        list: Simplified list of search results
    """
    items = raw_response.get("items", [])

    results = []
    for item in items:
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet"),
            "displayLink": item.get("displayLink"),
        })

    return results


if __name__ == "__main__":
    query = "Python playwright tutorial"

    print(f"Searching for: '{query}'\n")
    print("=" * 60)

    # Get raw response
    raw_response = google_search(query)

    # Show the raw response structure (truncated for readability)
    print("\n[RAW RESPONSE STRUCTURE]")
    print("-" * 40)

    # Show key fields in the response
    print(f"kind: {raw_response.get('kind')}")
    print(f"searchInformation:")
    search_info = raw_response.get("searchInformation", {})
    print(f"  totalResults: {search_info.get('totalResults')}")
    print(f"  searchTime: {search_info.get('searchTime')} seconds")

    print(f"\nNumber of items returned: {len(raw_response.get('items', []))}")

    # Show first raw item as example
    if raw_response.get("items"):
        print("\n[FIRST RAW ITEM EXAMPLE]")
        print("-" * 40)
        first_item = raw_response["items"][0]
        print(json.dumps(first_item, indent=2)[:1000] + "...")

    # Show cleaned results
    print("\n[EXTRACTED RESULTS]")
    print("-" * 40)

    results = extract_search_results(raw_response)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['link']}")
        print(f"   Snippet: {result['snippet'][:100]}...")
