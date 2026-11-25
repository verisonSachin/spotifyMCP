"""MCP tool implementations and remix-aware track matching for Spotify."""

from __future__ import annotations

import asyncio
import re
import unicodedata
from typing import Any

from mcp.server.mcpserver import Context, MCPServer, ToolError
from mcp.types import ToolAnnotations

from spotify_mcp.client import SpotifyClient
from spotify_mcp.models import (
    AddPlaylistItemsResult,
    BuildPlaylistResult,
    CreatedPlaylist,
    MatchedTrack,
    PlaylistItemResult,
    PlaylistSummary,
    SaveToLibraryResult,
    TrackQuery,
    TrackResult,
)

TRACK_URI_RE = re.compile(r"^spotify:track:[A-Za-z0-9]+$")
LIBRARY_URI_RE = re.compile(r"^spotify:(track|album|artist):[A-Za-z0-9]+$")
PLAYLIST_BATCH_SIZE = 100
SEARCH_CONCURRENCY_LIMIT = 4

VERSION_KEYWORDS: frozenset[str] = frozenset(
    {
        "remix",
        "mix",
        "edit",
        "vip",
        "dub",
        "rework",
        "bootleg",
        "live",
        "acoustic",
        "instrumental",
        "karaoke",
        "extended",
        "club",
        "radio",
    }
)


def normalize_text(text: str) -> str:
    """Fold accents, lowercase, and strip punctuation for robust matching."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_folded = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_folded.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def tokenize(text: str) -> set[str]:
    """Return normalized word tokens from text."""
    normalized = normalize_text(text)
    return set(normalized.split()) if normalized else set()


def _extract_version_tokens(title: str) -> set[str]:
    tokens = tokenize(title)
    if not (tokens & VERSION_KEYWORDS):
        return set()
    # Include parenthetical/bracketed remix descriptors (e.g. "four", "tet", "remix")
    paren_parts = re.findall(r"[\(\[]([^\)\]]+)[\)\]]", title)
    version_tokens = set(tokens & VERSION_KEYWORDS)
    for part in paren_parts:
        part_tokens = tokenize(part)
        if part_tokens & VERSION_KEYWORDS:
            version_tokens.update(part_tokens)
    return version_tokens


def _token_overlap_ratio(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & actual) / len(expected)


def score_track_candidate(
    requested_artist: str,
    requested_title: str,
    candidate: dict[str, Any],
) -> float:
    """Score a Spotify track candidate against requested artist and title."""
    cand_name = str(candidate.get("name", ""))
    cand_artists_raw = candidate.get("artists") or []
    cand_artist_names = [
        str(a.get("name", "")) for a in cand_artists_raw if isinstance(a, dict)
    ]
    cand_artists_joined = " ".join(cand_artist_names)

    req_title_norm = normalize_text(requested_title)
    cand_title_norm = normalize_text(cand_name)

    req_title_tokens = tokenize(requested_title)
    cand_title_tokens = tokenize(cand_name)

    req_artist_tokens = tokenize(requested_artist)
    cand_artist_tokens = tokenize(cand_artists_joined)

    title_recall = _token_overlap_ratio(req_title_tokens, cand_title_tokens)
    title_precision = _token_overlap_ratio(cand_title_tokens, req_title_tokens)
    artist_recall = _token_overlap_ratio(req_artist_tokens, cand_artist_tokens)

    score = (title_recall * 50.0) + (title_precision * 15.0) + (artist_recall * 35.0)

    if req_title_norm and req_title_norm == cand_title_norm:
        score += 15.0

    req_version = _extract_version_tokens(requested_title)
    cand_version = _extract_version_tokens(cand_name)

    if req_version:
        if cand_version:
            overlap = _token_overlap_ratio(req_version, cand_version)
            score += 25.0 * overlap
            if overlap == 0.0:
                score -= 20.0
        else:
            score -= 25.0
    else:
        if cand_version:
            score -= 25.0

    return score


def select_best_track(
    requested_artist: str,
    requested_title: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Select the highest-scoring candidate if it meets minimum match quality."""
    if not candidates:
        return None

    best_candidate: dict[str, Any] | None = None
    best_score = float("-inf")
    for candidate in candidates:
        uri = candidate.get("uri")
        if not isinstance(uri, str) or not TRACK_URI_RE.match(uri):
            continue
        score = score_track_candidate(requested_artist, requested_title, candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None or best_score < 25.0:
        return None
    return best_candidate


def _compact_track(item: dict[str, Any]) -> TrackResult:
    artists = [
        str(a.get("name", ""))
        for a in (item.get("artists") or [])
        if isinstance(a, dict) and a.get("name")
    ]
    album_obj = item.get("album") or {}
    external_urls = item.get("external_urls") or {}
    return {
        "uri": str(item.get("uri", "")),
        "name": str(item.get("name", "")),
        "artists": artists,
        "album": str(album_obj.get("name", "")) if isinstance(album_obj, dict) else "",
        "duration_ms": int(item.get("duration_ms", 0) or 0),
        "url": str(external_urls.get("spotify", "")) if isinstance(external_urls, dict) else "",
    }


def _compact_playlist_summary(item: dict[str, Any]) -> PlaylistSummary:
    external_urls = item.get("external_urls") or {}
    items_meta = item.get("items")
    if not isinstance(items_meta, dict):
        items_meta = item.get("tracks") or {}
    total = items_meta.get("total", 0) if isinstance(items_meta, dict) else 0
    return {
        "id": str(item.get("id", "")),
        "name": str(item.get("name", "")),
        "public": item.get("public"),
        "item_count": int(total or 0),
        "url": str(external_urls.get("spotify", "")) if isinstance(external_urls, dict) else "",
    }


def _compact_playlist_item(entry: dict[str, Any]) -> PlaylistItemResult | None:
    track = entry.get("item")
    if not isinstance(track, dict):
        return None
    uri = track.get("uri")
    if not isinstance(uri, str) or not uri:
        return None
    track_type = track.get("type")
    if track_type is not None and track_type != "track":
        return None
    artists = [
        str(a.get("name", ""))
        for a in (track.get("artists") or [])
        if isinstance(a, dict) and a.get("name")
    ]
    album_obj = track.get("album") or {}
    return {
        "uri": uri,
        "name": str(track.get("name", "")),
        "artists": artists,
        "album": str(album_obj.get("name", "")) if isinstance(album_obj, dict) else "",
    }


def _get_client(ctx: Context) -> SpotifyClient:
    lifespan_ctx = ctx.request_context.lifespan_context
    if isinstance(lifespan_ctx, SpotifyClient):
        return lifespan_ctx
    client = getattr(lifespan_ctx, "spotify_client", None)
    if isinstance(client, SpotifyClient):
        return client
    raise ToolError("SpotifyClient is not initialized in server lifespan context.")


async def _search_tracks_impl(
    client: SpotifyClient,
    query: str,
    limit: int = 5,
) -> list[TrackResult]:
    if not query or not query.strip():
        raise ToolError("query must be a non-empty string.")
    if not (1 <= limit <= 10):
        raise ToolError("limit must be between 1 and 10 (Spotify 2026 search API cap).")

    data = await client.get(
        "/search",
        params={"q": query.strip(), "type": "track", "limit": limit},
    )
    tracks_obj = data.get("tracks") or {}
    items = tracks_obj.get("items") or []
    return [_compact_track(item) for item in items if isinstance(item, dict)]


async def _get_my_playlists_impl(
    client: SpotifyClient,
    limit: int = 20,
    offset: int = 0,
) -> list[PlaylistSummary]:
    if not (1 <= limit <= 50):
        raise ToolError("limit must be between 1 and 50.")
    if offset < 0:
        raise ToolError("offset must be >= 0.")

    data = await client.get("/me/playlists", params={"limit": limit, "offset": offset})
    items = data.get("items") or []
    return [_compact_playlist_summary(item) for item in items if isinstance(item, dict)]


async def _get_playlist_items_impl(
    client: SpotifyClient,
    playlist_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[PlaylistItemResult]:
    if not playlist_id or not playlist_id.strip():
        raise ToolError("playlist_id must be a non-empty string.")
    if not (1 <= limit <= 100):
        raise ToolError("limit must be between 1 and 100.")
    if offset < 0:
        raise ToolError("offset must be >= 0.")

    data = await client.get(
        f"/playlists/{playlist_id.strip()}/items",
        params={"limit": limit, "offset": offset},
    )
    items = data.get("items") or []
    results: list[PlaylistItemResult] = []
    for entry in items:
        if isinstance(entry, dict):
            compact = _compact_playlist_item(entry)
            if compact is not None:
                results.append(compact)
    return results


async def _create_playlist_impl(
    client: SpotifyClient,
    name: str,
    description: str = "",
    public: bool = False,
) -> CreatedPlaylist:
    if not name or not name.strip():
        raise ToolError("name must be a non-empty string.")

    data = await client.post(
        "/me/playlists",
        json={
            "name": name.strip(),
            "description": description,
            "public": public,
        },
    )
    external_urls = data.get("external_urls") or {}
    return {
        "id": str(data.get("id", "")),
        "name": str(data.get("name", name.strip())),
        "url": str(external_urls.get("spotify", "")) if isinstance(external_urls, dict) else "",
        "public": data.get("public"),
    }


async def _add_playlist_items_impl(
    client: SpotifyClient,
    playlist_id: str,
    uris: list[str],
    position: int | None = None,
) -> AddPlaylistItemsResult:
    if not playlist_id or not playlist_id.strip():
        raise ToolError("playlist_id must be a non-empty string.")
    if not uris:
        raise ToolError("uris must contain at least 1 Spotify track URI.")
    if position is not None and position < 0:
        raise ToolError("position must be >= 0 when provided.")

    for uri in uris:
        if not isinstance(uri, str) or not TRACK_URI_RE.match(uri):
            raise ToolError(
                f"Invalid track URI '{uri}'. Expected format 'spotify:track:<id>'."
            )

    clean_id = playlist_id.strip()
    snapshot_id = ""
    added_count = 0
    next_position = position

    for idx in range(0, len(uris), PLAYLIST_BATCH_SIZE):
        batch = uris[idx : idx + PLAYLIST_BATCH_SIZE]
        body: dict[str, Any] = {"uris": batch}
        if next_position is not None:
            body["position"] = next_position
            next_position += len(batch)

        resp = await client.post(f"/playlists/{clean_id}/items", json=body)
        added_count += len(batch)
        if isinstance(resp.get("snapshot_id"), str):
            snapshot_id = resp["snapshot_id"]

    return {
        "playlist_id": clean_id,
        "added_count": added_count,
        "snapshot_id": snapshot_id,
    }


async def _save_to_library_impl(
    client: SpotifyClient,
    uris: list[str],
) -> SaveToLibraryResult:
    if not (1 <= len(uris) <= 40):
        raise ToolError("uris must contain between 1 and 40 Spotify URIs.")

    for uri in uris:
        if not isinstance(uri, str) or not LIBRARY_URI_RE.match(uri):
            raise ToolError(
                f"Invalid library URI '{uri}'. Expected track, album, or artist Spotify URI."
            )

    await client.put("/me/library", json={"uris": uris})
    return {
        "saved_count": len(uris),
        "uris": list(uris),
    }


async def _resolve_single_track(
    client: SpotifyClient,
    semaphore: asyncio.Semaphore,
    query: TrackQuery,
) -> tuple[str, dict[str, Any] | None]:
    artist = str(query.get("artist", "")).strip()
    title = str(query.get("title", "")).strip()
    requested_label = f"{artist} - {title}"

    async with semaphore:
        quoted_q = f'track:"{title}" artist:"{artist}"'
        first_resp = await client.get(
            "/search",
            params={"q": quoted_q, "type": "track", "limit": 5},
        )
        candidates = (first_resp.get("tracks") or {}).get("items") or []

        if not candidates:
            fallback_q = f"{artist} {title}"
            second_resp = await client.get(
                "/search",
                params={"q": fallback_q, "type": "track", "limit": 5},
            )
            candidates = (second_resp.get("tracks") or {}).get("items") or []

    best = select_best_track(artist, title, [c for c in candidates if isinstance(c, dict)])
    return requested_label, best


async def _build_playlist_from_list_impl(
    client: SpotifyClient,
    name: str,
    tracks: list[TrackQuery],
    description: str = "",
    public: bool = False,
) -> BuildPlaylistResult:
    if not name or not name.strip():
        raise ToolError("name must be a non-empty string.")
    if not (1 <= len(tracks) <= 100):
        raise ToolError("tracks must contain between 1 and 100 track queries.")

    for idx, track in enumerate(tracks):
        artist = str(track.get("artist", "")).strip() if isinstance(track, dict) else ""
        title = str(track.get("title", "")).strip() if isinstance(track, dict) else ""
        if not artist or not title:
            raise ToolError(f"Track at index {idx} must include non-empty 'artist' and 'title'.")

    semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY_LIMIT)
    resolved = await asyncio.gather(
        *(_resolve_single_track(client, semaphore, track) for track in tracks)
    )

    matched: list[MatchedTrack] = []
    unmatched: list[str] = []
    seen_uris: set[str] = set()
    ordered_uris: list[str] = []

    for requested_label, best in resolved:
        if best is None:
            unmatched.append(requested_label)
            continue

        uri = str(best["uri"])
        if uri in seen_uris:
            continue
        seen_uris.add(uri)
        ordered_uris.append(uri)

        cand_artists = ", ".join(
            str(a.get("name", ""))
            for a in (best.get("artists") or [])
            if isinstance(a, dict) and a.get("name")
        )
        matched.append(
            {
                "requested": requested_label,
                "matched": f"{cand_artists} - {best.get('name', '')}",
                "uri": uri,
            }
        )

    created = await _create_playlist_impl(
        client,
        name=name,
        description=description,
        public=public,
    )

    added_count = 0
    if ordered_uris:
        add_result = await _add_playlist_items_impl(
            client,
            playlist_id=created["id"],
            uris=ordered_uris,
        )
        added_count = add_result["added_count"]

    return {
        "playlist": {
            "id": created["id"],
            "name": created["name"],
            "url": created["url"],
        },
        "added_count": added_count,
        "matched": matched,
        "unmatched": unmatched,
    }


def register_tools(mcp: MCPServer) -> None:
    """Register all 7 Spotify MCP tools on the provided MCPServer."""

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    async def search_tracks(ctx: Context, query: str, limit: int = 5) -> list[TrackResult]:
        """Search Spotify tracks (limit 1-10) and return compact track metadata."""
        return await _search_tracks_impl(_get_client(ctx), query=query, limit=limit)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    async def get_my_playlists(
        ctx: Context,
        limit: int = 20,
        offset: int = 0,
    ) -> list[PlaylistSummary]:
        """List the authenticated user's playlists (limit 1-50)."""
        return await _get_my_playlists_impl(_get_client(ctx), limit=limit, offset=offset)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
    async def get_playlist_items(
        ctx: Context,
        playlist_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[PlaylistItemResult]:
        """Inspect tracks in a Spotify playlist using the 2026 /items endpoint."""
        return await _get_playlist_items_impl(
            _get_client(ctx),
            playlist_id=playlist_id,
            limit=limit,
            offset=offset,
        )

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False))
    async def create_playlist(
        ctx: Context,
        name: str,
        description: str = "",
        public: bool = False,
    ) -> CreatedPlaylist:
        """Create a new private or public Spotify playlist via /v1/me/playlists."""
        return await _create_playlist_impl(
            _get_client(ctx),
            name=name,
            description=description,
            public=public,
        )

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False))
    async def add_playlist_items(
        ctx: Context,
        playlist_id: str,
        uris: list[str],
        position: int | None = None,
    ) -> AddPlaylistItemsResult:
        """Append Spotify track URIs to a playlist in batches of 100."""
        return await _add_playlist_items_impl(
            _get_client(ctx),
            playlist_id=playlist_id,
            uris=uris,
            position=position,
        )

    @mcp.tool(annotations=ToolAnnotations(idempotent_hint=True))
    async def save_to_library(ctx: Context, uris: list[str]) -> SaveToLibraryResult:
        """Save 1-40 track, album, or artist URIs to the user's Spotify library."""
        return await _save_to_library_impl(_get_client(ctx), uris=uris)

    @mcp.tool(annotations=ToolAnnotations(read_only_hint=False))
    async def build_playlist_from_list(
        ctx: Context,
        name: str,
        tracks: list[TrackQuery],
        description: str = "",
        public: bool = False,
    ) -> BuildPlaylistResult:
        """Resolve artist/title pairs, create a playlist, and add matched tracks."""
        return await _build_playlist_from_list_impl(
            _get_client(ctx),
            name=name,
            tracks=tracks,
            description=description,
            public=public,
        )
