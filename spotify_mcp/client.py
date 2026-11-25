"""Async Spotify Web API HTTP client with token injection and retry handling."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from mcp.server.mcpserver import ToolError

from spotify_mcp.auth import AuthError, TokenManager

logger = logging.getLogger(__name__)

SPOTIFY_API_BASE_URL = "https://api.spotify.com/v1"
DEFAULT_TIMEOUT_SECONDS = 15.0
MAX_RATE_LIMIT_RETRIES = 3
DEFAULT_RETRY_AFTER_SECONDS = 2.0


def _parse_retry_after(header_value: str | None) -> float:
    if not header_value:
        return DEFAULT_RETRY_AFTER_SECONDS
    try:
        parsed = float(header_value)
    except ValueError:
        return DEFAULT_RETRY_AFTER_SECONDS
    return max(0.0, parsed)


def _extract_spotify_error(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return text or f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            status = err.get("status", response.status_code)
            message = err.get("message", "Unknown Spotify API error")
            return f"Spotify API error ({status}): {message}"
        if isinstance(err, str):
            return f"Spotify API error ({response.status_code}): {err}"
    return f"Spotify API error ({response.status_code})"


class SpotifyClient:
    """Shared async client for the 2026 Spotify Web API."""

    def __init__(
        self,
        *,
        token_manager: TokenManager | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=SPOTIFY_API_BASE_URL,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )
        self.token_manager = token_manager or TokenManager(http_client=self._client)

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient if owned by this instance."""
        if self._owns_client:
            await self._client.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform an authenticated Spotify API request with 429 and 401 recovery."""
        refreshed_after_401 = False
        rate_limit_attempts = 0

        while True:
            try:
                access_token = (
                    await self.token_manager.force_refresh()
                    if refreshed_after_401
                    else await self.token_manager.get_valid_access_token()
                )
            except AuthError as exc:
                raise ToolError(str(exc)) from exc

            headers = {"Authorization": f"Bearer {access_token}"}

            try:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=json,
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise ToolError(f"Network error calling Spotify API {method} {path}: {exc}") from exc

            if response.status_code == 401 and not refreshed_after_401:
                logger.warning("Received 401 from Spotify API; forcing one token refresh.")
                refreshed_after_401 = True
                continue

            if response.status_code == 401:
                raise ToolError(
                    "Spotify authentication failed (401 Unauthorized) after token refresh. "
                    "Please re-run `uv run spotify-mcp-auth`."
                )

            if response.status_code == 429:
                if rate_limit_attempts >= MAX_RATE_LIMIT_RETRIES:
                    raise ToolError(
                        "Spotify API rate limit exceeded (429) after maximum retries. "
                        "Please try again shortly."
                    )
                delay = _parse_retry_after(response.headers.get("Retry-After"))
                rate_limit_attempts += 1
                logger.warning(
                    "Spotify API rate limited (429) on %s %s; retrying in %.2fs (attempt %d/%d).",
                    method,
                    path,
                    delay,
                    rate_limit_attempts,
                    MAX_RATE_LIMIT_RETRIES,
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 400:
                raise ToolError(_extract_spotify_error(response))

            if response.status_code == 204 or not response.content:
                return {}

            try:
                data = response.json()
            except ValueError as exc:
                raise ToolError(f"Invalid JSON response from Spotify API {method} {path}.") from exc

            if not isinstance(data, dict):
                return {"data": data}
            return data

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Issue an authenticated GET request."""
        return await self.request("GET", path, params=params)

    async def post(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue an authenticated POST request."""
        return await self.request("POST", path, params=params, json=json)

    async def put(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue an authenticated PUT request."""
        return await self.request("PUT", path, params=params, json=json)
