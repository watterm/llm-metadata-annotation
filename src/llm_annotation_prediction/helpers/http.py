import functools
import json
from logging import getLogger
from typing import Any

import httpx
from pydantic import BaseModel

from llm_annotation_prediction.helpers.open_router import KeyInfo, KeyInfoWrap

_logger = getLogger("OpenRouter")


class SimplifiedResponse(BaseModel):
    """
    Helper class for saving OpenRouter responses. Strips irrelevant HTTP context.
    """

    status_code: int

    request_headers: dict[str, str]
    response_headers: dict[str, str]

    request_body: dict[str, Any]
    response_body: dict[str, Any]

    url: str
    elapsed_time: float  # Time in seconds from request until close of response

    @classmethod
    def from_httpx_response(cls, response: httpx.Response) -> "SimplifiedResponse":
        # Hide API key in logs
        request_headers = dict(response.request.headers)
        if "authorization" in request_headers:
            request_headers["authorization"] = "**REDACTED**"

        return cls(
            status_code=response.status_code,
            request_headers=request_headers,
            response_headers=dict(response.headers),
            request_body=json.loads(
                response.request.content
            ),  # Assuming the request is JSON
            response_body=response.json(),  # Assuming the response is JSON
            url=str(response.url),
            elapsed_time=response.elapsed.total_seconds(),
        )


@functools.lru_cache(maxsize=1)
def get_key_info(key: str, api_url: httpx.URL) -> KeyInfo:
    """
    Lazy loading for OpenRouter API key information. Using a cache will only make the
    request (thread-safely) once and afterwards return the cached response.
    """
    _logger.info("Retrieving OpenRouter key information")
    headers = {"Authorization": f"Bearer {key}"}
    response = httpx.get(api_url.join("auth/key"), headers=headers)

    if response.status_code != 200:
        _logger.critical(response.text)
        _logger.critical(
            f"Could not receive OpenRouter key information ({response.status_code}). Quitting."
        )
        exit(1)

    key_info = KeyInfoWrap.model_validate(response.json()).data
    key_info.label = "**REDACTED**"  # Key must not show up in logs
    _logger.info(f"Key info: {key_info}")
    return key_info
