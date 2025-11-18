from asyncio import Lock
import re
import unicodedata

import httpx
from lxml.etree import tostring
from lxml.html import HtmlElement, fromstring
from lxml_html_clean import Cleaner
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright.async_api._generated import Playwright

# Global browser instance cache
_playwright: Playwright | None = None
_browser: Browser | None = None
_browser_lock = Lock()


async def _get_browser() -> Browser:
    """Get or create a cached browser instance (thread-safe)."""
    global _playwright, _browser

    # Fast path: browser already exists and is connected
    if _browser is not None and _browser.is_connected():
        return _browser

    # Slow path: need to create browser (acquire lock)
    async with _browser_lock:
        # Double-check after acquiring lock (another task might have created it)
        if _browser is None or not _browser.is_connected():
            if _playwright is None:
                _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch()
        return _browser


async def close_browser() -> None:
    """Close the browser and playwright instance. Call during shutdown."""
    global _playwright, _browser
    async with _browser_lock:
        if _browser is not None:
            await _browser.close()
            _browser = None
        if _playwright is not None:
            await _playwright.stop()
            _playwright = None


async def fetch_html_httpx(url: str, timeout: float = 30.0) -> str:
    """Fetch HTML using httpx (fast, but doesn't execute JavaScript)."""
    async with httpx.AsyncClient() as client:
        response: httpx.Response = await client.get(
            url, timeout=timeout, follow_redirects=True
        )
        response.raise_for_status()
        return response.text


async def fetch_html_playwright(url: str, timeout: float = 30.0) -> str:
    """Fetch HTML using Playwright (handles SPAs and JavaScript-rendered content)."""
    browser = await _get_browser()

    # Use browser context for isolation (prevents state leakage between scrapes)
    context: BrowserContext = await browser.new_context()
    try:
        page: Page = await context.new_page()
        await page.goto(url, timeout=int(timeout * 1000), wait_until="networkidle")
        html: str = await page.content()
        return html
    finally:
        await context.close()  # Closes all pages and clears state


def clean_html(html: str) -> str:
    """Clean HTML for LLM consumption."""
    doc: HtmlElement = fromstring(html)

    # Remove boilerplate elements
    for element in doc.xpath("//nav | //header | //footer | //aside"):
        element.getparent().remove(element)

    # Remove images with empty alt attributes
    for img in doc.xpath('//img[@alt=""]'):
        img.getparent().remove(img)

    cleaner = Cleaner(
        scripts=True,
        javascript=True,
        comments=True,
        style=True,
        inline_style=True,
        links=True,
        meta=True,
        page_structure=True,
        processing_instructions=True,
        embedded=True,
        frames=True,
        forms=False,
        annoying_tags=True,
        remove_unknown_tags=False,
        safe_attrs_only=True,
        safe_attrs=frozenset(["href", "alt", "title"]),
    )

    cleaned: HtmlElement = cleaner.clean_html(doc)
    html = tostring(cleaned, encoding="unicode", method="html")

    # Normalize Unicode and whitespace
    html = unicodedata.normalize("NFC", html)
    html = html.replace("\r\n", "\n").replace("\r", "\n")
    html = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", html)
    html = re.sub(r"[\u200B-\u200D\uFEFF]", "", html)
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r" *\n *", "\n", html)
    html = re.sub(r"\n{3,}", "\n\n", html)

    return html.strip()


async def fetch_and_clean_html(
    url: str, timeout: float = 30.0, use_playwright: bool = False
) -> str:
    """
    Fetch HTML content from a URL and clean it for LLM consumption.

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds
        use_playwright: Use Playwright for JavaScript-rendered content (SPAs)

    Returns:
        Cleaned text content suitable for LLM input
    """
    if use_playwright:
        html = await fetch_html_playwright(url, timeout)
    else:
        html = await fetch_html_httpx(url, timeout)

    return clean_html(html)
