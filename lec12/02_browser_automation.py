# TOPIC: Browser Automation with Playwright
# This demo shows how to use Playwright for browser automation.
#
# Setup required:
# 1. pip install playwright
# 2. playwright install chromium
#
# Playwright enables programmatic control of web browsers,
# essential for agentic web search where an LLM needs to
# navigate and interact with web pages.

import asyncio
from playwright.async_api import async_playwright


async def basic_navigation_demo():
    """Demonstrate basic page navigation and element interaction."""

    async with async_playwright() as p:
        # Launch browser (headless=False to see the browser)
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("[1] Navigating to example.com...")
        await page.goto("https://example.com")

        # Get page title
        title = await page.title()
        print(f"    Page title: {title}")

        # Get page URL
        print(f"    Current URL: {page.url}")

        # Get text content from an element
        heading = await page.locator("h1").text_content()
        print(f"    H1 heading: {heading}")

        # Get all paragraph text
        paragraphs = await page.locator("p").all_text_contents()
        print(f"    Number of paragraphs: {len(paragraphs)}")

        print("\n[2] Navigating to Wikipedia...")
        await page.goto("https://en.wikipedia.org")

        title = await page.title()
        print(f"    Page title: {title}")

        # Type into search box and search
        print("\n[3] Searching for 'Artificial Intelligence'...")
        search_input = page.locator("#searchInput")
        await search_input.fill("Artificial Intelligence")
        await search_input.press("Enter")

        # Wait for navigation
        await page.wait_for_load_state("networkidle")

        print(f"    Navigated to: {page.url}")

        # Extract first paragraph of the article
        first_para = await page.locator("#mw-content-text p").first.text_content()
        print(f"    First paragraph preview: {first_para[:200]}...")

        # Get all links on the page
        links = await page.locator("a[href]").count()
        print(f"    Total links on page: {links}")

        await browser.close()


async def extract_page_content_demo():
    """Demonstrate extracting structured content from a page."""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("\n[4] Extracting structured content from Hacker News...")
        await page.goto("https://news.ycombinator.com")

        # Extract top stories
        stories = await page.locator(".titleline > a").all()

        print("    Top 5 stories:")
        for i, story in enumerate(stories[:5], 1):
            title = await story.text_content()
            href = await story.get_attribute("href")
            print(f"    {i}. {title}")
            print(f"       URL: {href}")

        await browser.close()


async def form_interaction_demo():
    """Demonstrate form interactions."""

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("\n[5] Form interaction demo with httpbin...")
        await page.goto("https://httpbin.org/forms/post")

        # Fill out the form
        await page.fill('input[name="custname"]', "Test User")
        await page.fill('input[name="custtel"]', "555-1234")
        await page.fill('input[name="custemail"]', "test@example.com")

        # Select a radio button
        await page.check('input[value="medium"]')

        # Check a checkbox
        await page.check('input[name="topping"][value="cheese"]')

        print("    Form filled successfully!")
        print("    - Customer name: Test User")
        print("    - Phone: 555-1234")
        print("    - Size: medium")
        print("    - Topping: cheese")

        await browser.close()


async def main():
    print("=" * 60)
    print("PLAYWRIGHT BROWSER AUTOMATION DEMO")
    print("=" * 60)

    await basic_navigation_demo()
    await extract_page_content_demo()
    await form_interaction_demo()

    print("\n" + "=" * 60)
    print("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
