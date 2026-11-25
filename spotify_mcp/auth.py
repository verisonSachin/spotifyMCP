"""OAuth2 + PKCE loopback authentication and XDG token management for Spotify."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse
import webbrowser

import httpx

logger = logging.getLogger(__name__)

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
REDIRECT_HOST = "127.0.0.1"
REDIRECT_PORT = 8888
REDIRECT_PATH = "/callback"
REDIRECT_URI = f"http://{REDIRECT_HOST}:{REDIRECT_PORT}{REDIRECT_PATH}"

REQUIRED_SCOPES: tuple[str, ...] = (
    "playlist-modify-private",
    "playlist-modify-public",
    "playlist-read-private",
    "user-library-read",
    "user-library-modify",
)

REFRESH_LEEWAY_SECONDS = 60.0


class AuthError(RuntimeError):
    """Raised when Spotify OAuth tokens are missing, invalid, or cannot be refreshed."""


@dataclass(slots=True)
class TokenData:
    """Persisted OAuth2 token payload."""

    access_token: str
    refresh_token: str
    expires_at: float
    scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenData:
        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        expires_at = data.get("expires_at")
        if not isinstance(access_token, str) or not access_token:
            raise AuthError("Stored token data is missing 'access_token'.")
        if not isinstance(refresh_token, str) or not refresh_token:
            raise AuthError("Stored token data is missing 'refresh_token'.")
        if not isinstance(expires_at, (int, float)):
            raise AuthError("Stored token data is missing numeric 'expires_at'.")
        scope = data.get("scope", "")
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=float(expires_at),
            scope=scope if isinstance(scope, str) else "",
        )


def get_token_path() -> Path:
    """Return the XDG token file path without storing anything in the repository."""
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    base_dir = Path(xdg_config_home).expanduser() if xdg_config_home else Path.home() / ".config"
    return base_dir / "spotify-mcp" / "tokens.json"


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a cryptographic PKCE (code_verifier, code_challenge) pair."""
    code_verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def build_authorize_url(
    client_id: str,
    code_challenge: str,
    state: str,
    *,
    redirect_uri: str = REDIRECT_URI,
    scopes: tuple[str, ...] = REQUIRED_SCOPES,
) -> str:
    """Build the Spotify OAuth2 authorization URL with PKCE parameters."""
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
        "state": state,
        "scope": " ".join(scopes),
    }
    return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"


def save_tokens_atomic(tokens: TokenData, token_path: Path | None = None) -> Path:
    """Atomically persist tokens with 0o700 directory and 0o600 file permissions."""
    path = token_path or get_token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)

    payload = json.dumps(tokens.to_dict(), indent=2) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".tokens.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def load_tokens(token_path: Path | None = None) -> TokenData:
    """Load persisted OAuth tokens from disk."""
    path = token_path or get_token_path()
    if not path.exists():
        raise AuthError(
            f"Spotify tokens not found at {path}. Run `uv run spotify-mcp-auth` to authenticate."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthError(
            f"Failed to read Spotify tokens from {path}. Re-run `uv run spotify-mcp-auth`."
        ) from exc
    if not isinstance(raw, dict):
        raise AuthError("Invalid token file format. Re-run `uv run spotify-mcp-auth`.")
    return TokenData.from_dict(raw)


class TokenManager:
    """Async-safe token reader and refresher guarded by an asyncio.Lock."""

    def __init__(
        self,
        token_path: Path | None = None,
        *,
        client_id: str | None = None,
        client_secret: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.token_path = token_path or get_token_path()
        self._client_id = client_id
        self._client_secret = client_secret
        self._http_client = http_client
        self._lock = asyncio.Lock()
        self._cached: TokenData | None = None

    def _resolve_client_credentials(self) -> tuple[str, str | None]:
        client_id = self._client_id or os.environ.get("SPOTIFY_CLIENT_ID")
        if not client_id:
            raise AuthError(
                "SPOTIFY_CLIENT_ID environment variable is required to refresh Spotify tokens."
            )
        client_secret = self._client_secret
        if client_secret is None:
            client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET") or None
        return client_id, client_secret

    def _is_expiring(self, tokens: TokenData) -> bool:
        return time.time() >= tokens.expires_at - REFRESH_LEEWAY_SECONDS

    async def get_valid_access_token(self) -> str:
        """Return a valid access token, refreshing proactively if near expiry."""
        async with self._lock:
            if self._cached is None:
                self._cached = load_tokens(self.token_path)
            if not self._is_expiring(self._cached):
                return self._cached.access_token
            refreshed = await self._refresh_locked(self._cached)
            return refreshed.access_token

    async def force_refresh(self) -> str:
        """Force a token refresh regardless of local expiry timestamp."""
        async with self._lock:
            current = self._cached or load_tokens(self.token_path)
            refreshed = await self._refresh_locked(current)
            return refreshed.access_token

    async def _refresh_locked(self, current: TokenData) -> TokenData:
        client_id, client_secret = self._resolve_client_credentials()
        form: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": current.refresh_token,
            "client_id": client_id,
        }
        if client_secret:
            form["client_secret"] = client_secret

        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(timeout=15.0)
        try:
            response = await client.post(SPOTIFY_TOKEN_URL, data=form)
        except httpx.HTTPError as exc:
            raise AuthError(f"Network error refreshing Spotify token: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

        if response.status_code >= 400:
            raise AuthError(
                f"Spotify token refresh failed ({response.status_code}). "
                "Re-run `uv run spotify-mcp-auth`."
            )

        payload = response.json()
        access_token = payload.get("access_token")
        expires_in = payload.get("expires_in", 3600)
        if not isinstance(access_token, str) or not access_token:
            raise AuthError(
                "Spotify token refresh response missing access_token. "
                "Re-run `uv run spotify-mcp-auth`."
            )

        rotated_refresh = payload.get("refresh_token")
        refresh_token = (
            rotated_refresh
            if isinstance(rotated_refresh, str) and rotated_refresh
            else current.refresh_token
        )
        scope = payload.get("scope")
        updated = TokenData(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=time.time() + float(expires_in),
            scope=scope if isinstance(scope, str) else current.scope,
        )
        save_tokens_atomic(updated, self.token_path)
        self._cached = updated
        logger.info("Refreshed Spotify access token.")
        return updated


def exchange_code_for_tokens(
    *,
    code: str,
    code_verifier: str,
    client_id: str,
    client_secret: str | None = None,
    redirect_uri: str = REDIRECT_URI,
) -> TokenData:
    """Exchange an authorization code for access and refresh tokens (sync CLI helper)."""
    form: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    if client_secret:
        form["client_secret"] = client_secret

    response = httpx.post(SPOTIFY_TOKEN_URL, data=form, timeout=15.0)
    if response.status_code >= 400:
        raise AuthError(
            f"Spotify token exchange failed ({response.status_code}): {response.text}"
        )
    payload = response.json()
    access_token = payload.get("access_token")
    refresh_token = payload.get("refresh_token")
    expires_in = payload.get("expires_in", 3600)
    if not isinstance(access_token, str) or not isinstance(refresh_token, str):
        raise AuthError("Spotify token response missing access_token or refresh_token.")
    scope = payload.get("scope", "")
    return TokenData(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=time.time() + float(expires_in),
        scope=scope if isinstance(scope, str) else "",
    )


def _run_loopback_auth(client_id: str, client_secret: str | None) -> TokenData:
    code_verifier, code_challenge = generate_pkce_pair()
    expected_state = secrets.token_urlsafe(32)
    auth_url = build_authorize_url(client_id, code_challenge, expected_state)

    result: dict[str, str] = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != REDIRECT_PATH:
                self.send_response(404)
                self.end_headers()
                return

            params = parse_qs(parsed.query)
            error = params.get("error", [None])[0]
            state = params.get("state", [None])[0]
            code = params.get("code", [None])[0]

            if error:
                result["error"] = f"Spotify authorization denied: {error}"
                self._send_html(400, "<h1>Authorization Failed</h1><p>You may close this tab.</p>")
                return

            if state != expected_state or not code:
                result["error"] = "Invalid OAuth state or missing authorization code."
                self._send_html(400, "<h1>Invalid Callback</h1><p>State mismatch or missing code.</p>")
                return

            result["code"] = code
            self._send_html(
                200,
                "<h1>Spotify MCP Authenticated</h1><p>Tokens saved. You may close this tab.</p>",
            )

        def _send_html(self, status: int, body: str) -> None:
            encoded = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = HTTPServer((REDIRECT_HOST, REDIRECT_PORT), CallbackHandler)
    try:
        sys.stdout.write(
            f"Open this URL in your browser to authorize Spotify MCP:\n\n{auth_url}\n\n"
        )
        sys.stdout.flush()
        webbrowser.open(auth_url)

        while "code" not in result and "error" not in result:
            server.handle_request()
    finally:
        server.server_close()

    if "error" in result:
        raise AuthError(result["error"])

    return exchange_code_for_tokens(
        code=result["code"],
        code_verifier=code_verifier,
        client_id=client_id,
        client_secret=client_secret,
    )


def main() -> None:
    """CLI entrypoint (`spotify-mcp-auth`) to complete OAuth2 + PKCE login."""
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    if not client_id:
        sys.stderr.write("Error: SPOTIFY_CLIENT_ID environment variable is not set.\n")
        raise SystemExit(1)

    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET") or None
    try:
        tokens = _run_loopback_auth(client_id, client_secret)
        saved_path = save_tokens_atomic(tokens)
    except AuthError as exc:
        sys.stderr.write(f"Authentication failed: {exc}\n")
        raise SystemExit(1) from exc

    sys.stdout.write(f"Authentication successful. Tokens saved to {saved_path}\n")
