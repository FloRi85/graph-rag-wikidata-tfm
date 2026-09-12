"""Wikidata entity-search candidates for the interactive demo.

The user chooses a candidate; the demo does not automatically disambiguate.
Controlled evaluation bypasses this module and uses Mintaka's gold QIDs.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

from src.retrieval.wikidata import MEDIAWIKI_API, USER_AGENT


def _search_entity(name: str) -> list[dict]:
    """
    Up to 5 Wikidata candidates for `name` via the MediaWiki wbsearchentities
    API (cheaper than SPARQL for entity lookup).

    Each dict has keys: qid, label, description.
    """
    params = urllib.parse.urlencode({
        "action": "wbsearchentities",
        "search": name,
        "language": "en",
        "format": "json",
        "limit": 5,
    })
    req = urllib.request.Request(f"{MEDIAWIKI_API}?{params}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())

    return [
        {"qid": item["id"], "label": item.get("label", ""),
         "description": item.get("description", "")}
        for item in data.get("search", [])
    ]
