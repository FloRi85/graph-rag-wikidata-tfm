"""Offline tests of the demo's candidate search and evaluation boundary."""

import json
import os
import sys
import urllib.parse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.retrieval import entity_linking


class TestCandidateSearch:

    def test_preserves_candidate_order_for_user_selection(self, monkeypatch):
        requests = []

        class Response:
            def read(self):
                return json.dumps({"search": [
                    {"id": "Q1", "label": "Foo", "description": "a film"},
                    {"id": "Q2", "label": "Foo", "description": "a scientist"},
                    {"id": "Q3"},
                ]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def urlopen(request, timeout=None):
            requests.append((request, timeout))
            return Response()

        monkeypatch.setattr(entity_linking.urllib.request, "urlopen", urlopen)
        assert entity_linking._search_entity("Foo & Bar") == [
            {"qid": "Q1", "label": "Foo", "description": "a film"},
            {"qid": "Q2", "label": "Foo", "description": "a scientist"},
            {"qid": "Q3", "label": "", "description": ""},
        ]
        request, timeout = requests[0]
        params = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
        assert params["search"] == ["Foo & Bar"]
        assert params["action"] == ["wbsearchentities"]
        assert params["language"] == ["en"]
        assert params["limit"] == ["5"]
        assert timeout == 10


class TestTheEvalPathCannotReachThis:

    def test_no_pipeline_imports_the_linker(self):
        import pathlib
        import re
        root = pathlib.Path(__file__).resolve().parents[2]
        importer = re.compile(r"^\s*(from|import)\s+.*entity_linking", re.M)
        for path in (root / "src" / "pipelines").glob("*.py"):
            assert not importer.search(path.read_text(encoding="utf-8")), \
                f"{path.name} imports the demo linker"

    def test_wikidata_no_longer_exposes_the_linker(self):
        from src.retrieval import wikidata
        for name in ("resolve_entity", "_search_entity", "_pick_best_entity",
                     "retrieve_context", "_format_statements"):
            assert not hasattr(wikidata, name), \
                f"wikidata.{name} came back; the eval path is QID-in, statements-out"
