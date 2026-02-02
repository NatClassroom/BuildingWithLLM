# TOPIC: Chart Creation Tool
# Give the agent ability to create visualizations

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def create_bar_chart(title: str, labels: list[str], values: list[float], filename: str) -> dict:
    """Create a bar chart.

    Args:
        title: Chart title
        labels: Labels for each bar
        values: Values for each bar
        filename: Output filename (without extension)
    """
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="steelblue")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = f"{filename}.png"
    plt.savefig(path)
    plt.close()
    return {"created": path, "type": "bar_chart"}


def create_pie_chart(title: str, labels: list[str], values: list[float], filename: str) -> dict:
    """Create a pie chart.

    Args:
        title: Chart title
        labels: Labels for each slice
        values: Values for each slice
        filename: Output filename (without extension)
    """
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.tight_layout()
    path = f"{filename}.png"
    plt.savefig(path)
    plt.close()
    return {"created": path, "type": "pie_chart"}


def create_line_chart(title: str, x_values: list[float], y_values: list[float], filename: str) -> dict:
    """Create a line chart.

    Args:
        title: Chart title
        x_values: X-axis values
        y_values: Y-axis values
        filename: Output filename (without extension)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker="o", color="steelblue")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{filename}.png"
    plt.savefig(path)
    plt.close()
    return {"created": path, "type": "line_chart"}


# Let the agent create charts from natural language
tools = [create_bar_chart, create_pie_chart, create_line_chart]
config = types.GenerateContentConfig(tools=tools)

queries = [
    "Create a bar chart showing sales by region: North 150, South 200, East 175, West 225. Save as regional_sales",
    "Make a pie chart of market share: Apple 35%, Samsung 25%, Xiaomi 15%, Others 25%. Save as market_share",
    "Plot monthly revenue: Jan 100, Feb 120, Mar 115, Apr 140, May 160, Jun 155. Save as revenue_trend",
]

for query in queries:
    print(f"\nUser: {query}")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query,
        config=config,
    )
    print(f"Agent: {response.text}")
