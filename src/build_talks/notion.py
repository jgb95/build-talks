"""
Notion integration.

NotionFetcher loads a Notion database once and provides lookups by talk ID,
including downloading title card images to a local cache directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

import requests
from notion_client import Client as NotionClient
from notion_client.client import ClientOptions as NotionClientOptions

from build_talks.config import DO_SPACES_BASE_URL, NOTION_CLIPART_PROP, NOTION_SOCIAL_CARD_PROP

NOTION_EVENT_PROP = "Event"

log = logging.getLogger(__name__)


class NotionFetcher:
    """Loads a Notion database once and provides lookups by talk ID."""

    def __init__(self, token: str, database_id: str) -> None:
        # Pin to Notion-Version 2022-06-28 — the 2025-09-03 version moved
        # /databases/{id}/query to a stricter data_sources API requiring
        # additional sharing steps. 2022-06-28 is the last stable version
        # that supports the standard databases query endpoint.
        self.client = NotionClient(
            options=NotionClientOptions(auth=token, notion_version="2022-06-28")
        )
        self.database_id = database_id
        self._pages: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Query the full database and index pages by their Clipart ID."""
        log.info("[notion] loading database...")
        cursor = None
        while True:
            body: dict = {"page_size": 100}
            if cursor:
                body["start_cursor"] = cursor
            # notion-client v3 removed databases.query(); use the raw request method.
            response = self.client.request(
                f"databases/{self.database_id}/query",
                method="POST",
                body=body,
            )
            for page in response["results"]:
                talk_id = self._extract_clipart_id(page)
                if talk_id:
                    self._pages[talk_id] = page

            if not response.get("has_more"):
                break
            cursor = response.get("next_cursor")

        log.info("[notion] loaded %d pages", len(self._pages))

    def _extract_clipart_id(self, page: dict) -> str | None:
        """
        Pull the talk ID from the Clipart property.
        The field contains '{ID}.{ext}' — only the ID part is returned.
        """
        prop = page["properties"].get(NOTION_CLIPART_PROP)
        if not prop:
            return None

        ptype = prop["type"]
        if ptype == "title":
            raw = "".join(t["plain_text"] for t in prop["title"]).strip()
        elif ptype == "rich_text":
            raw = "".join(t["plain_text"] for t in prop["rich_text"]).strip()
        elif ptype == "formula":
            raw = prop["formula"].get("string", "")
        else:
            return None

        if not raw:
            return None

        # Strip the file extension, e.g. "talk-01.png" → "talk-01"
        return raw.rsplit(".", 1)[0]

    def _resolve_url(self, url: str) -> str:
        """
        Resolve a URL that may be a full URL or a relative path.

        Relative paths (starting with '/') are prepended with DO_SPACES_BASE_URL
        so both formats reference the same Digital Ocean Spaces endpoint.
        """
        if url.startswith(("http://", "https://")):
            return url
        return DO_SPACES_BASE_URL.rstrip("/") + "/" + url.lstrip("/")

    def get_event(self, talk_id: str) -> str:
        """
        Return the event name (e.g. 'durham') for the given talk ID.

        Reads the 'Event' property; returns an empty string if absent.
        """
        page = self._pages.get(talk_id)
        if not page:
            raise KeyError(
                f"No Notion page found with {NOTION_CLIPART_PROP} ID='{talk_id}'"
            )

        prop = page["properties"].get(NOTION_EVENT_PROP)
        if not prop:
            return ""

        ptype = prop["type"]
        if ptype == "select":
            sel = prop["select"]
            return sel["name"].strip() if sel else ""
        elif ptype == "rich_text":
            return "".join(t["plain_text"] for t in prop["rich_text"]).strip()
        elif ptype == "title":
            return "".join(t["plain_text"] for t in prop["title"]).strip()
        elif ptype == "formula":
            return prop["formula"].get("string", "").strip()
        return ""

    def get_social_card_url(self, talk_id: str) -> str:
        """Return the SocialCard image download URL for the given talk ID."""
        page = self._pages.get(talk_id)
        if not page:
            raise KeyError(
                f"No Notion page found with {NOTION_CLIPART_PROP} ID='{talk_id}'"
            )

        prop = page["properties"].get(NOTION_SOCIAL_CARD_PROP)
        if not prop:
            raise KeyError(
                f"Page '{talk_id}' has no '{NOTION_SOCIAL_CARD_PROP}' property"
            )

        ptype = prop["type"]
        if ptype == "url":
            url = prop["url"]
        elif ptype == "files":
            files = prop["files"]
            if not files:
                raise ValueError(
                    f"No file in '{NOTION_SOCIAL_CARD_PROP}' for '{talk_id}'"
                )
            f = files[0]
            url = f["external"]["url"] if f["type"] == "external" else f["file"]["url"]
        elif ptype == "rich_text":
            url = "".join(t["plain_text"] for t in prop["rich_text"]).strip()
        else:
            raise ValueError(
                f"Unsupported property type '{ptype}' for '{NOTION_SOCIAL_CARD_PROP}'"
            )

        if not url:
            raise ValueError(f"Empty SocialCard URL for '{talk_id}'")
        return url

    def download_title_card(self, talk_id: str, titles_dir: Path) -> Path:
        """
        Return the local path to the title card image for talk_id.

        Downloads from Notion if not already cached in titles_dir.
        Raises ValueError if the downloaded file is empty (partial download).
        """
        titles_dir.mkdir(parents=True, exist_ok=True)

        # Return immediately if a non-video image is already cached.
        for existing in titles_dir.glob(f"{talk_id}.*"):
            if existing.suffix.lower() != ".mp4":
                return existing

        url = self._resolve_url(self.get_social_card_url(talk_id))
        ext = Path(urlparse(url).path).suffix or ".png"
        dest = titles_dir / f"{talk_id}{ext}"

        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                fh.write(chunk)

        if dest.stat().st_size == 0:
            dest.unlink()
            raise ValueError(f"Downloaded title card for '{talk_id}' is empty")

        return dest
