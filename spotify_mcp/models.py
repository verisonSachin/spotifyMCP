"""Compact TypedDict schemas for Spotify MCP tool inputs and outputs."""

from __future__ import annotations

from typing import TypedDict


class TrackResult(TypedDict):
    """Compact track representation returned by search_tracks."""

    uri: str
    name: str
    artists: list[str]
    album: str
    duration_ms: int
    url: str


class PlaylistSummary(TypedDict):
    """Compact playlist summary returned by get_my_playlists."""

    id: str
    name: str
    public: bool | None
    item_count: int
    url: str


class PlaylistItemResult(TypedDict):
    """Compact track item returned by get_playlist_items."""

    uri: str
    name: str
    artists: list[str]
    album: str


class CreatedPlaylist(TypedDict):
    """Compact playlist metadata returned by create_playlist."""

    id: str
    name: str
    url: str
    public: bool | None


class AddPlaylistItemsResult(TypedDict):
    """Summary returned by add_playlist_items."""

    playlist_id: str
    added_count: int
    snapshot_id: str


class SaveToLibraryResult(TypedDict):
    """Summary returned by save_to_library."""

    saved_count: int
    uris: list[str]


class TrackQuery(TypedDict):
    """Input track specification for build_playlist_from_list."""

    artist: str
    title: str


class MatchedTrack(TypedDict):
    """Matched track entry in build_playlist_from_list output."""

    requested: str
    matched: str
    uri: str


class PlaylistRef(TypedDict):
    """Compact playlist reference in build_playlist_from_list output."""

    id: str
    name: str
    url: str


class BuildPlaylistResult(TypedDict):
    """Summary returned by build_playlist_from_list."""

    playlist: PlaylistRef
    added_count: int
    matched: list[MatchedTrack]
    unmatched: list[str]
