"""Unit tests for title/artist normalization and remix-aware track matching."""

from __future__ import annotations

from spotify_mcp.tools import normalize_text, score_track_candidate, select_best_track


def _make_candidate(
    uri: str,
    name: str,
    artists: list[str],
) -> dict:
    return {
        "uri": uri,
        "name": name,
        "artists": [{"name": a} for a in artists],
    }


def test_normalize_text_handles_accents_and_punctuation() -> None:
    assert normalize_text("Fred again..") == "fred again"
    assert normalize_text("Ben Böhmer") == "ben bohmer"
    assert normalize_text("Delilah (pull me out of this)") == "delilah pull me out of this"


def test_select_best_track_prefers_exact_original_over_remix() -> None:
    remix = _make_candidate(
        "spotify:track:remix123",
        "Strong (Club Remix)",
        ["Romy", "Fred again.."],
    )
    original = _make_candidate(
        "spotify:track:orig123",
        "Strong",
        ["Romy", "Fred again.."],
    )
    best = select_best_track("Romy, Fred again..", "Strong", [remix, original])
    assert best is not None
    assert best["uri"] == "spotify:track:orig123"


def test_select_best_track_prefers_requested_remix() -> None:
    original = _make_candidate(
        "spotify:track:orig456",
        "Nothing Left To Lose",
        ["Everything But The Girl"],
    )
    other_remix = _make_candidate(
        "spotify:track:other789",
        "Nothing Left To Lose (Radio Edit)",
        ["Everything But The Girl"],
    )
    four_tet = _make_candidate(
        "spotify:track:fourtet123",
        "Nothing Left To Lose (Four Tet Remix)",
        ["Everything But The Girl", "Four Tet"],
    )

    best = select_best_track(
        "Everything But The Girl",
        "Nothing Left To Lose (Four Tet Remix)",
        [original, other_remix, four_tet],
    )
    assert best is not None
    assert best["uri"] == "spotify:track:fourtet123"


def test_select_best_track_rejects_unrelated_or_invalid_candidates() -> None:
    unrelated = _make_candidate(
        "spotify:track:bad1",
        "Completely Different Song",
        ["Unknown Artist"],
    )
    invalid_uri = _make_candidate(
        "spotify:episode:ep123",
        "Strong",
        ["Romy"],
    )
    assert select_best_track("Romy", "Strong", [unrelated, invalid_uri]) is None
    assert select_best_track("Romy", "Strong", []) is None


def test_score_track_candidate_handles_multi_artist_and_accent_folding() -> None:
    cand = _make_candidate(
        "spotify:track:ben123",
        "Run Away",
        ["Ben Böhmer", "Tinlicker", "Felix Raphael"],
    )
    score = score_track_candidate("Ben Bohmer, Tinlicker", "Run Away", cand)
    assert score >= 80.0
