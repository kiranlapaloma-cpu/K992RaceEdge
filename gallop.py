"""Gallop race-card client for Race Edge.

This module talks to the JSON feeds observed behind Gallop's fixtures page.
It deliberately keeps HTTP/network concerns separate from the Race Card UI.

Observed feed patterns:
  meeting: /php/gallop.php?feed=meeting&club=<club>&date=<YYYYMMDD>
  event:   /php/gallop.php?feed=event&date=<YYYYMMDD>&club=<club>&event=<event>

These are website feeds rather than a documented public API, so all callers
should handle GallopFeedError gracefully if the site changes or is unavailable.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import requests

BASE_URL = "https://www.gallop.co.za/fixtures/php/gallop.php"
DEFAULT_TIMEOUT = 15


class GallopFeedError(RuntimeError):
    """Raised when a Gallop feed cannot be fetched or interpreted."""


def _date_param(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y%m%d")
    if isinstance(value, date):
        return value.strftime("%Y%m%d")

    text = str(value or "").strip()
    if not text:
        raise GallopFeedError("A race date is required.")

    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 8:
        return digits

    try:
        return datetime.fromisoformat(text).strftime("%Y%m%d")
    except Exception as exc:
        raise GallopFeedError(f"Could not understand Gallop date: {value}") from exc


def _positive_int(value: Any, label: str) -> int:
    try:
        out = int(value)
    except Exception as exc:
        raise GallopFeedError(f"{label} must be a number.") from exc
    if out <= 0:
        raise GallopFeedError(f"{label} must be greater than zero.")
    return out


def _session() -> requests.Session:
    s = requests.Session()
    # Gallop is a browser-facing website. A normal browser UA avoids some
    # generic anti-bot responses while still making a standard GET request.
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/126.0 Safari/537.36"
        ),
        "Accept": "application/json,text/plain,*/*",
        "Referer": "https://www.gallop.co.za/fixtures/fixtureIframeLink",
    })
    return s


def _get_json(params: dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Any:
    try:
        with _session() as s:
            response = s.get(BASE_URL, params=params, timeout=timeout)
            response.raise_for_status()
    except requests.Timeout as exc:
        raise GallopFeedError("Gallop took too long to respond.") from exc
    except requests.RequestException as exc:
        raise GallopFeedError(f"Could not connect to Gallop: {exc}") from exc

    try:
        return response.json()
    except ValueError as exc:
        preview = (response.text or "").strip().replace("\n", " ")[:180]
        extra = f" Response started with: {preview}" if preview else ""
        raise GallopFeedError(f"Gallop did not return JSON.{extra}") from exc


def fetch_meeting(club: int, race_date: date | datetime | str) -> Any:
    """Return Gallop's raw meeting feed for one club/date."""
    club_id = _positive_int(club, "Club ID")
    date_text = _date_param(race_date)
    return _get_json({"feed": "meeting", "club": club_id, "date": date_text})


def fetch_event(club: int, race_date: date | datetime | str, event: int) -> dict:
    """Return one full Gallop event/race card including the runners array."""
    club_id = _positive_int(club, "Club ID")
    event_id = _positive_int(event, "Event ID")
    date_text = _date_param(race_date)

    payload = _get_json({
        "feed": "event",
        "date": date_text,
        "club": club_id,
        "event": event_id,
    })
    if not isinstance(payload, dict):
        raise GallopFeedError("Gallop's event feed did not return a race object.")
    if not isinstance(payload.get("runners"), list):
        raise GallopFeedError("Gallop's event feed did not contain a runners list.")
    return payload


def _walk_dicts(value: Any):
    """Yield nested dictionaries so the meeting parser tolerates feed wrappers."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def meeting_races(payload: Any) -> list[dict]:
    """Extract unique race/event summaries from Gallop's meeting response.

    Gallop's website feed is not publicly documented, so this intentionally
    recognises race dictionaries by their stable fields rather than relying on
    one exact wrapper shape.
    """
    found: dict[int, dict] = {}

    for item in _walk_dicts(payload):
        event = item.get("event")
        race = item.get("race")
        if event is None or race is None:
            continue
        try:
            event_id = int(event)
            race_no = int(race)
        except Exception:
            continue
        if event_id <= 0 or race_no <= 0:
            continue

        found[event_id] = {
            "event": event_id,
            "race": race_no,
            "time": str(item.get("time") or "").strip(),
            "name": str(item.get("name") or "").strip(),
            "distance": _safe_int(item.get("distance")),
            "club": _safe_int(item.get("club")),
            "clubName": str(item.get("clubName") or "").strip(),
            "surfaceDescr": str(item.get("surfaceDescr") or "").strip(),
        }

    return sorted(found.values(), key=lambda r: (r.get("race") or 999, r.get("time") or ""))


def _safe_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return None


def race_option_label(race: dict) -> str:
    race_no = race.get("race")
    time_text = race.get("time") or "—"
    distance = race.get("distance")
    name = str(race.get("name") or "").strip()

    pieces = [f"Race {race_no if race_no is not None else '—'}", time_text]
    if distance:
        pieces.append(f"{distance}m")
    label = " · ".join(pieces)
    if name:
        label += f" — {name}"
    return label
