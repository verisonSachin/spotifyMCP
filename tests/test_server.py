"""In-memory MCP protocol tests using mcp.Client and respx HTTP mocks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import httpx
from mcp import Client
import pytest
import respx

from spotify_mcp.auth import TokenData, TokenManager, save_tokens_atomic
from spotify_mcp.client import SpotifyClient
from spotify_mcp.server import create_server


def _write_test_tokens(tmp_path: Path, *, expires_in: float = 3600.0) -> Path:
    token_path = tmp_path / "spotify-mcp" / "tokens.json"
    save_tokens_atomic(
        TokenData(
            access_token="access-initial",
            refresh_token="refresh-initial",
            expires_at=time.time() + expires_in,
            scope="playlist-modify-private playlist-read-private",
        ),
        token_path=token_path,
    )
    return token_path


def _make_test_server(token_path: Path):
    def factory() -> SpotifyClient:
        manager = TokenManager(
            token_path=token_path,
            client_id="test-client-id",
        )
        return SpotifyClient(token_manager=manager)

    return create_server(client_factory=factory)


@pytest.mark.asyncio
async def test_list_tools_exposes_all_seven_tools(tmp_path: Path) -> None:
    token_path = _write_test_tokens(tmp_path)
    server = _make_test_server(token_path)

    async with Client(server) as client:
        result = await client.list_tools()

    names = {tool.name for tool in result.tools}
    assert names == {
        "search_tracks",
        "get_my_playlists",
        "get_playlist_items",
        "create_playlist",
        "add_playlist_items",
        "save_to_library",
        "build_playlist_from_list",
    }


@pytest.mark.asyncio
@respx.mock
async def test_search_tracks_and_validation(tmp_path: Path) -> None:
    token_path = _write_test_tokens(tmp_path)
    server = _make_test_server(token_path)

    respx.get("https://api.spotify.com/v1/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "uri": "spotify:track:abc123",
                            "name": "Strong",
                            "artists": [{"name": "Romy"}, {"name": "Fred again.."}],
                            "album": {"name": "Mid Air"},
                            "duration_ms": 234567,
                            "external_urls": {"spotify": "https://open.spotify.com/track/abc123"},
                        }
                    ]
                }
            },
        )
    )

    async with Client(server) as client:
        res = await client.call_tool("search_tracks", {"query": "Romy Strong", "limit": 5})
        assert not res.isError
        payload = json.loads(res.content[0].text)
        assert payload[0]["uri"] == "spotify:track:abc123"
        assert payload[0]["artists"] == ["Romy", "Fred again.."]

        bad = await client.call_tool("search_tracks", {"query": "Romy", "limit": 11})
        assert bad.isError


@pytest.mark.asyncio
@respx.mock
async def test_2026_playlist_endpoints_and_chunking(tmp_path: Path) -> None:
    token_path = _write_test_tokens(tmp_path)
    server = _make_test_server(token_path)

    respx.get("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "pl1",
                        "name": "My Mix",
                        "public": False,
                        "items": {"total": 12},
                        "external_urls": {"spotify": "https://open.spotify.com/playlist/pl1"},
                    }
                ]
            },
        )
    )
    respx.get("https://api.spotify.com/v1/playlists/pl1/items").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "item": {
                            "type": "track",
                            "uri": "spotify:track:t1",
                            "name": "Atlas",
                            "artists": [{"name": "Bicep"}],
                            "album": {"name": "Isles"},
                        }
                    },
                    {"item": None},
                    {"item": {"type": "episode", "uri": "spotify:episode:ep1", "name": "Ep"}},
                ]
            },
        )
    )
    respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(
            201,
            json={
                "id": "newpl",
                "name": "Created",
                "public": False,
                "external_urls": {"spotify": "https://open.spotify.com/playlist/newpl"},
            },
        )
    )
    add_route = respx.post("https://api.spotify.com/v1/playlists/newpl/items").mock(
        return_value=httpx.Response(201, json={"snapshot_id": "snap1"})
    )
    lib_route = respx.put("https://api.spotify.com/v1/me/library").mock(
        return_value=httpx.Response(200)
    )

    uris_150 = [f"spotify:track:track{i:03d}" for i in range(150)]

    async with Client(server) as client:
        playlists_res = await client.call_tool("get_my_playlists", {})
        playlists = json.loads(playlists_res.content[0].text)
        assert playlists == [
            {
                "id": "pl1",
                "name": "My Mix",
                "public": False,
                "item_count": 12,
                "url": "https://open.spotify.com/playlist/pl1",
            }
        ]

        items_res = await client.call_tool("get_playlist_items", {"playlist_id": "pl1"})
        items = json.loads(items_res.content[0].text)
        assert items == [
            {
                "uri": "spotify:track:t1",
                "name": "Atlas",
                "artists": ["Bicep"],
                "album": "Isles",
            }
        ]

        create_res = await client.call_tool("create_playlist", {"name": "Created"})
        created = json.loads(create_res.content[0].text)
        assert created["id"] == "newpl"

        add_res = await client.call_tool(
            "add_playlist_items",
            {"playlist_id": "newpl", "uris": uris_150},
        )
        added = json.loads(add_res.content[0].text)
        assert added["added_count"] == 150
        assert add_route.call_count == 2
        first_body = json.loads(add_route.calls[0].request.content)
        second_body = json.loads(add_route.calls[1].request.content)
        assert len(first_body["uris"]) == 100
        assert len(second_body["uris"]) == 50

        bad_uri = await client.call_tool(
            "add_playlist_items",
            {"playlist_id": "newpl", "uris": ["not-a-uri"]},
        )
        assert bad_uri.isError

        save_res = await client.call_tool(
            "save_to_library",
            {"uris": ["spotify:track:t1", "spotify:album:a1"]},
        )
        saved = json.loads(save_res.content[0].text)
        assert saved["saved_count"] == 2
        assert lib_route.called


@pytest.mark.asyncio
@respx.mock
async def test_build_playlist_from_list_fallback_and_401_refresh(tmp_path: Path) -> None:
    token_path = _write_test_tokens(tmp_path)
    server = _make_test_server(token_path)

    respx.post("https://accounts.spotify.com/api/token").mock(
        return_value=httpx.Response(
            200,
            json={
                "access_token": "access-refreshed",
                "refresh_token": "refresh-rotated",
                "expires_in": 3600,
            },
        )
    )

    search_calls = 0

    def search_handler(request: httpx.Request) -> httpx.Response:
        nonlocal search_calls
        search_calls += 1
        if search_calls == 1:
            return httpx.Response(401, json={"error": {"status": 401, "message": "Expired"}})
        q = request.url.params.get("q", "")
        if 'track:"Strong"' in q:
            return httpx.Response(200, json={"tracks": {"items": []}})
        if "Romy, Fred again.. Strong" in q:
            return httpx.Response(
                200,
                json={
                    "tracks": {
                        "items": [
                            {
                                "uri": "spotify:track:strong1",
                                "name": "Strong",
                                "artists": [{"name": "Romy"}, {"name": "Fred again.."}],
                            }
                        ]
                    }
                },
            )
        return httpx.Response(200, json={"tracks": {"items": []}})

    respx.get("https://api.spotify.com/v1/search").mock(side_effect=search_handler)
    respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(
            201,
            json={
                "id": "built1",
                "name": "UK House",
                "public": False,
                "external_urls": {"spotify": "https://open.spotify.com/playlist/built1"},
            },
        )
    )
    respx.post("https://api.spotify.com/v1/playlists/built1/items").mock(
        return_value=httpx.Response(201, json={"snapshot_id": "snap-built"})
    )

    async with Client(server) as client:
        res = await client.call_tool(
            "build_playlist_from_list",
            {
                "name": "UK House",
                "tracks": [
                    {"artist": "Romy, Fred again..", "title": "Strong"},
                    {"artist": "Missing Artist", "title": "Missing Track"},
                ],
            },
        )
        assert not res.isError
        data = json.loads(res.content[0].text)
        assert data["added_count"] == 1
        assert data["matched"][0]["uri"] == "spotify:track:strong1"
        assert data["unmatched"] == ["Missing Artist - Missing Track"]

    persisted = json.loads(token_path.read_text(encoding="utf-8"))
    assert persisted["refresh_token"] == "refresh-rotated"
    assert (os.stat(token_path).st_mode & 0o777) == 0o600
