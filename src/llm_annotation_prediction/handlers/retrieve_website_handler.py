from asyncio import Task, TaskGroup
from logging import getLogger

from pydantic import BaseModel, TypeAdapter

from llm_annotation_prediction.handlers.handler import Handler, HandlerConfig
from llm_annotation_prediction.helpers.constants import Context
from llm_annotation_prediction.helpers.open_router import RequestDto, ResponseDto
from llm_annotation_prediction.helpers.rate_limiter.base import RateLimiterConfig
from llm_annotation_prediction.helpers.rate_limiter.local import LocalRateLimiter
from llm_annotation_prediction.helpers.scrape import fetch_and_clean_html
from llm_annotation_prediction.helpers.utils import extract_values

_logger = getLogger("RetrieveWebsite")


class RetrieveWebsiteHandlerConfig(HandlerConfig):
    type: str = "RetrieveWebsite"

    # Path(s) to extract URLs from context (e.g., "publication_list.urls")
    # Can be a single path or list of paths to try in order
    context_url_paths: list[str]

    # Where to store the formatted results in context
    key_for_context_storage: str = "retrieved_websites"

    # Template for formatting each scraped page
    format_template: str = "## {url}\n\n{content}\n\n"

    # Timeout for each request in seconds
    timeout: float = 30.0

    # Maximum number of URLs to scrape (prevents token overflow)
    max_urls: int = 10

    # Maximum content length per page (truncate if longer, None for no limit)
    max_content_length: int | None = 50000

    # Maximum requests per second for rate limiting
    max_requests_per_second: float = 1.0

    # Maximum concurrent requests
    max_concurrent_requests: int = 1

    # Maximum retries per request
    max_retries: int = 3

    # Use Playwright for JavaScript-rendered content (SPAs)
    use_playwright: bool = False


class ScrapeResult(BaseModel):
    """
    Result of a single website scraping operation.
    """

    url: str
    """The URL that was scraped"""

    content: str | None
    """The cleaned HTML content, or None if scraping failed"""

    error: str | None = None
    """Error message if scraping failed, None otherwise"""

    truncated: bool = False
    """Whether the content was truncated due to length limits"""


ScrapedResultsAdapter: TypeAdapter[list[ScrapeResult]] = TypeAdapter(list[ScrapeResult])


class RetrieveWebsiteHandler(Handler):
    """
    Retrieves and cleans websites based on URLs found in the LLM's previous response.
    The scraped content is stored in the context for use in subsequent messages.
    """

    def __init__(self, config: RetrieveWebsiteHandlerConfig, context: Context):
        super().__init__(config, context)
        self._config: RetrieveWebsiteHandlerConfig = config

        # Initialize rate limiter
        # Note: task_timeout should be larger than HTTP timeout to account for HTML processing
        # Add 50% buffer for HTML cleaning/processing time
        rate_limiter_config = RateLimiterConfig(
            name="WebsiteScraper",
            initial_rps=config.max_requests_per_second,
            min_rps=0.5,
            max_rps=config.max_requests_per_second,
            max_concurrency=config.max_concurrent_requests,
            max_retries=config.max_retries,
            task_timeout=config.timeout * 1.5,  # 50% buffer for HTML processing
        )
        self._rate_limiter = LocalRateLimiter(rate_limiter_config)

    async def handle_request(
        self, request_dto: RequestDto, is_tool_cycle: bool = False
    ) -> RequestDto:
        raise TypeError(f"{__name__} cannot be used for request handling")

    async def handle_response(
        self, response_dto: ResponseDto, is_tool_cycle: bool = False
    ) -> ResponseDto:
        """
        Extract URLs from context, scrape them, and store results.
        """
        if is_tool_cycle:  # Only process after main LLM response
            return response_dto

        try:
            _logger.debug("Looking for URLs to scrape in context")

            # Extract URLs from context
            urls: list[str] = self._extract_urls_from_context()

            if not urls:
                _logger.info("No URLs found to scrape")
                return response_dto

            # Limit number of URLs
            if len(urls) > self._config.max_urls:
                _logger.warning(
                    f"Found {len(urls)} URLs, limiting to {self._config.max_urls}"
                )
                urls = urls[: self._config.max_urls]

            _logger.info(f"Scraping {len(urls)} URLs")

            # Scrape all URLs concurrently with rate limiting
            results: list[ScrapeResult] = await self._scrape_urls(urls)

            # Store results in context
            self._store_results(results)

        finally:
            # Always close the rate limiter
            await self._rate_limiter.close()

        return response_dto

    def _extract_urls_from_context(self) -> list[str]:
        """
        Extract URLs from context based on configured paths.
        Returns a flat list of unique URLs.
        """
        urls: set[str] = set()

        for path in self._config.context_url_paths:
            extracted = extract_values(self._context, path)
            # Filter to only string values (URLs)
            urls.update(v.strip() for v in extracted if isinstance(v, str))

        return list(urls)

    async def _scrape_urls(self, urls: list[str]) -> list[ScrapeResult]:
        """
        Scrape all URLs concurrently with rate limiting.
        """
        results: list[Task[ScrapeResult]] = []

        async with TaskGroup() as tg:
            for url in urls:
                task: Task[ScrapeResult] = tg.create_task(self._scrape_single_url(url))
                # Store task reference with URL for later retrieval
                task.url = url  # type: ignore[attr-defined]
                results.append(task)

        # Gather results from completed tasks
        scraped_results: list[ScrapeResult] = []
        for task in results:
            try:
                result = task.result()
                scraped_results.append(result)
            except Exception as e:
                # Task failed - log and skip
                url_str: str = getattr(task, "url", "unknown")
                _logger.warning(f"Failed to scrape {url_str}: {e}")
                scraped_results.append(
                    ScrapeResult(url=url_str, content=None, error=str(e))
                )

        return scraped_results

    async def _scrape_single_url(self, url: str) -> ScrapeResult:
        """
        Scrape a single URL with rate limiting and error handling.
        """
        try:
            _logger.debug(f"Scraping {url}")

            # Use rate limiter for automatic retries and backoff
            content = await self._rate_limiter.enqueue(
                fetch_and_clean_html,
                url,
                self._config.timeout,
                self._config.use_playwright,
            )

            # Truncate if needed
            truncated = False
            if (
                self._config.max_content_length
                and len(content) > self._config.max_content_length
            ):
                original_length = len(content)
                content = content[: self._config.max_content_length]
                truncated = True
                _logger.debug(
                    f"Truncated content for {url}: {original_length} -> "
                    f"{self._config.max_content_length} chars"
                )

            return ScrapeResult(url=url, content=content, truncated=truncated)

        except Exception as e:
            _logger.warning(f"Failed to scrape {url}: {e}")
            return ScrapeResult(url=url, content=None, error=str(e))

    def _store_results(self, results: list[ScrapeResult]) -> None:
        """
        Store scraped results in context in both formatted and raw forms.
        """
        # Store raw results for debugging/analysis
        raw_key = f"{self._config.key_for_context_storage}_raw"
        self._context[raw_key] = ScrapedResultsAdapter.dump_json(results).decode(
            "utf-8"
        )

        # Format results using template
        formatted_parts: list[str] = []
        successful_count = 0

        for result in results:
            if result.content is not None:
                # Format successful scrapes
                formatted: str = self._config.format_template.format(
                    url=result.url, content=result.content
                )

                # Add truncation notice if needed
                if result.truncated:
                    formatted += (
                        f"*[Content truncated to {self._config.max_content_length} "
                        f"characters]*\n\n"
                    )

                formatted_parts.append(formatted)
                successful_count += 1
            else:
                # Format errors
                formatted_parts.append(
                    f"## {result.url}\n\n"
                    f"*Error: Could not retrieve page - {result.error}*\n\n"
                )

        # Store formatted string for use in messages
        formatted_content: str = "".join(formatted_parts)
        self._context[self._config.key_for_context_storage] = formatted_content

        _logger.info(
            f"Stored {successful_count}/{len(results)} successfully scraped websites "
            f"in context['{self._config.key_for_context_storage}']"
        )
