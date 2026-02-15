# TOPIC: Multimodal Screen Understanding
# This demo shows how to capture screenshots with Playwright
# and use a vision-capable LLM (Gemini) to understand the screen.
#
# This is a key capability for agentic web browsing:
# - The agent takes a screenshot of the current page
# - The LLM analyzes the screenshot to understand what's on screen
# - The LLM decides what action to take next
#
# Setup required:
# 1. pip install playwright google-genai
# 2. playwright install chromium
# 3. Set GEMINI_API_KEY in .env

import asyncio
import base64
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from playwright.async_api import async_playwright

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


async def capture_screenshot(url: str, output_path: str = "screenshot.png") -> bytes:
    """
    Capture a screenshot of a webpage.

    Args:
        url: The URL to capture
        output_path: Where to save the screenshot

    Returns:
        bytes: The screenshot image data
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 720})

        await page.goto(url)
        await page.wait_for_load_state("networkidle")

        # Capture screenshot
        screenshot_bytes = await page.screenshot(path=output_path, full_page=False)

        await browser.close()

    return screenshot_bytes


def analyze_screenshot_with_gemini(
    screenshot_bytes: bytes, prompt: str
) -> str:
    """
    Send a screenshot to Gemini for analysis.

    Args:
        screenshot_bytes: The screenshot image data
        prompt: The question/instruction for the LLM

    Returns:
        str: The LLM's analysis
    """
    # Create the image part
    image_part = types.Part.from_bytes(data=screenshot_bytes, mime_type="image/png")

    # Create content with image and text
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(
                role="user",
                parts=[
                    image_part,
                    types.Part.from_text(text=prompt),
                ],
            )
        ],
    )

    return response.text


async def demo_page_understanding():
    """Demonstrate understanding a webpage through its screenshot."""

    print("[1] Capturing screenshot of Hacker News...")
    screenshot = await capture_screenshot(
        "https://news.ycombinator.com", "hn_screenshot.png"
    )
    print(f"    Screenshot saved to hn_screenshot.png")

    print("\n[2] Asking Gemini to describe the page...")
    description = analyze_screenshot_with_gemini(
        screenshot,
        "Describe what you see on this webpage. What kind of website is this? "
        "List the main elements visible on the screen.",
    )
    print(f"    Response:\n{description}")

    print("\n[3] Asking Gemini to extract specific information...")
    extraction = analyze_screenshot_with_gemini(
        screenshot,
        "Extract the top 3 article titles visible on this page. "
        "Format them as a numbered list.",
    )
    print(f"    Response:\n{extraction}")


async def demo_navigation_decision():
    """Demonstrate using LLM to decide navigation actions."""

    print("\n[4] Capturing screenshot of Wikipedia main page...")
    screenshot = await capture_screenshot(
        "https://en.wikipedia.org/wiki/Main_Page", "wiki_screenshot.png"
    )
    print(f"    Screenshot saved to wiki_screenshot.png")

    print("\n[5] Asking Gemini what actions are possible...")
    actions = analyze_screenshot_with_gemini(
        screenshot,
        """You are a web browsing agent. Analyze this screenshot and list:
1. What clickable elements do you see? (links, buttons, search boxes)
2. If I wanted to search for "Machine Learning", what element should I interact with?
3. Describe the exact location of the search box on the screen.""",
    )
    print(f"    Response:\n{actions}")


async def demo_form_understanding():
    """Demonstrate understanding form elements."""

    print("\n[6] Capturing screenshot of a form...")
    screenshot = await capture_screenshot(
        "https://httpbin.org/forms/post", "form_screenshot.png"
    )
    print(f"    Screenshot saved to form_screenshot.png")

    print("\n[7] Asking Gemini to understand the form...")
    form_analysis = analyze_screenshot_with_gemini(
        screenshot,
        """Analyze this form and answer:
1. What fields does this form contain?
2. What are the input types (text, radio, checkbox)?
3. If I wanted to order a large pizza with bacon topping,
   what values should I enter in each field?""",
    )
    print(f"    Response:\n{form_analysis}")


async def main():
    print("=" * 60)
    print("MULTIMODAL SCREEN UNDERSTANDING DEMO")
    print("=" * 60)

    await demo_page_understanding()
    await demo_navigation_decision()
    await demo_form_understanding()

    # Cleanup screenshots
    for f in ["hn_screenshot.png", "wiki_screenshot.png", "form_screenshot.png"]:
        if Path(f).exists():
            Path(f).unlink()
            print(f"\nCleaned up {f}")

    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
