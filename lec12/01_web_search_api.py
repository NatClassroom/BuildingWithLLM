# TOPIC: Web Search API (DuckDuckGo)
# This demo shows how to perform web searches programmatically
# using DuckDuckGo's search API (no API key required).
#
# Setup required:
# pip install ddgs

from ddgs import DDGS


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Perform a web search using DuckDuckGo.

    Args:
        query: The search query string
        num_results: Number of results to return

    Returns:
        list: Search results with title, url, and snippet
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=num_results))
    return results


def format_results_for_llm(results: list[dict]) -> str:
    """
    Format search results as a string for LLM consumption.

    Args:
        results: List of search result dicts

    Returns:
        str: Formatted string of search results
    """
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(
            f"{i}. {result['title']}\n"
            f"   URL: {result['href']}\n"
            f"   {result['body']}"
        )
    return "\n\n".join(formatted)


if __name__ == "__main__":
    import json

    query = "Python playwright tutorial"

    print(f"Searching for: '{query}'\n")
    print("=" * 60)

    # Get search results
    results = web_search(query)

    # Show raw response structure
    print("\n[RAW RESPONSE STRUCTURE]")
    print("-" * 40)
    print(f"Number of results: {len(results)}")

    if results:
        print("\n[FIRST RAW RESULT EXAMPLE]")
        print("-" * 40)
        print(json.dumps(results[0], indent=2))

    # Show all results formatted
    print("\n[ALL RESULTS]")
    print("-" * 40)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"   URL: {result['href']}")
        print(f"   Snippet: {result['body'][:150]}...")

    # Show LLM-friendly format
    print("\n[LLM-FRIENDLY FORMAT]")
    print("-" * 40)
    print(format_results_for_llm(results[:3]))
