"""
Unit tests for src/retrieval/wikidata.py — pure functions and SPARQL filtering.
No network calls; SPARQL is mocked.

Run from repo root:
    venv/Scripts/python -m pytest tests/unit/test_wikidata.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from unittest.mock import patch

import numpy as np

from src.retrieval import wikidata
from src.retrieval.wikidata import _is_noise_prop
from src.retrieval.embedding_retriever import _topk_indices


# ---------------------------------------------------------------------------
# _topk_indices — per-group normalized selection (pure, numpy only)
# ---------------------------------------------------------------------------

class TestTopKIndices:
    def test_plain_topk_without_groups(self):
        scores = np.array([0.1, 0.9, 0.5, 0.3])
        assert _topk_indices(scores, None, 2) == [1, 2]

    def test_single_group_is_noop(self):
        """One group => monotonic transform => identical to plain top-k."""
        scores = np.array([0.1, 0.9, 0.5, 0.3])
        groups = ["A", "A", "A", "A"]
        assert _topk_indices(scores, groups, 3) == _topk_indices(scores, None, 3)

    def test_normalization_rescues_quiet_group(self):
        """
        Group B has systematically lower raw scores (the 'quiet entity'), but its
        best item is a strong within-group outlier. Plain top-2 picks both A items;
        normalized top-2 must surface B's best item.
        """
        #            A0    A1    B0    B1
        scores = np.array([0.70, 0.68, 0.40, 0.20])
        groups = ["A", "A", "B", "B"]
        assert _topk_indices(scores, None, 2) == [0, 1]        # both from A
        assert 2 in _topk_indices(scores, groups, 2)           # B0 rescued


# ---------------------------------------------------------------------------
# _is_noise_prop — datatype-driven noise filter (uses shipped label→datatype map)
# ---------------------------------------------------------------------------

class TestIsNoiseProp:
    """
    ⚠️ Re-expressed 2026-08-12 against DATATYPES rather than bare labels.

    Until then the datatype came from a shipped 6,631-entry label→datatype map,
    so `_is_noise_prop("ISNI")` could answer on the label alone. The map is
    deleted: the statement fetch resolves `wikibase:propertyType` in-query, for
    statement AND qualifier properties, so the datatype now arrives with the
    data and cannot drift from Wikidata.

    The parametrisations below therefore carry the datatype the fetch supplies.
    Cases that must still work WITHOUT one are in `TestNoDatatypeResidual`.
    """

    @pytest.mark.parametrize("label,datatype", [
        ("IMDb ID", "ExternalId"), ("ISNI", "ExternalId"),
        ("VIAF cluster ID", "ExternalId"), ("X (Twitter) username", "ExternalId"),
        ("image", "CommonsMedia"), ("logo image", "CommonsMedia"),
        ("official website", "Url"),
    ])
    def test_identifier_media_url_datatypes_dropped(self, label, datatype):
        assert _is_noise_prop(label, datatype) is True

    @pytest.mark.parametrize("label,datatype", [
        ("instance of", "WikibaseItem"), ("capital", "WikibaseItem"),
        ("genre", "WikibaseItem"), ("occupation", "WikibaseItem"),
        ("date of birth", "Time"),
        ("postal code", "String"), ("review score", "String"),
    ])
    def test_fact_datatypes_kept(self, label, datatype):
        assert _is_noise_prop(label, datatype) is False

    @pytest.mark.parametrize("label,datatype", [
        ("ISNI", "ExternalId"),             # no " id", no "identifier"
        ("image", "CommonsMedia"),
        ("official website", "Url"),
        ("X (Twitter) username", "ExternalId"),
    ])
    def test_the_datatype_catches_what_no_label_rule_can(self, label, datatype):
        """
        The reason tier 1 exists, and the reason dropping the datatype argument
        is not a harmless omission: none of these match any label rule.
        """
        assert _is_noise_prop(label, datatype) is True
        assert _is_noise_prop(label) is False, "label rules alone must NOT catch it"

    @pytest.mark.parametrize("label", [
        "Commons category", "Commons gallery",       # String plumbing (via "commons")
        "topic's main category", "topic has template",  # WikibaseItem plumbing
        "described by source", "different from",      # exact blocklist
        "maintained by WikiProject",                  # wikiproject rule
    ])
    def test_fact_typed_plumbing_dropped_by_label(self, label):
        assert _is_noise_prop(label) is True

    def test_unknown_datatype_identifier_fallback(self):
        # Tier 3: a property with no datatype still gets caught by label shape.
        assert _is_noise_prop("Made-up registry ID") is True
        assert _is_noise_prop("some external identifier") is True

    def test_unknown_datatype_plain_fact_kept(self):
        assert _is_noise_prop("totally novel relation") is False


class TestNoDatatypeResidual:
    """
    What survives when no datatype is supplied — the free-form demo path, and any
    caller that forgets. Pinned so the DEGRADATION is explicit rather than
    discovered later as a mysterious change in pool size.
    """

    @pytest.mark.parametrize("label", [
        "IMDb ID",                    # " id" suffix
        "some external identifier",   # "identifier" substring
        "maintained by WikiProject",
        "topic's main category",
        "Commons category",
        "described by source",
    ])
    def test_label_rules_still_fire(self, label):
        assert _is_noise_prop(label) is True

    @pytest.mark.parametrize("label", ["ISNI", "image", "official website"])
    def test_datatype_only_noise_is_NOT_caught(self, label):
        """
        Not a bug — a measured consequence of deleting the static label→datatype
        map. High-volume noise whose label matches no rule is caught by datatype
        alone, so a caller that omits it gets a materially weaker filter.
        """
        assert _is_noise_prop(label) is False


class TestTruthyPathIsGone:
    """
    The truthy retrieval was deleted 2026-08-16 (dead since the statement-model
    rebuild, and guarded before that because it would have dropped LESS than it
    did when every recorded C3/C4 number was produced).

    Pinned as ABSENCE: reintroducing any of these names would mean a second
    retrieval path exists again, which is what made "which system produced this
    number?" ambiguous in the first place.
    """

    @pytest.mark.parametrize("name", [
        "fetch_triples", "_fetch_triples", "_clean", "_format_triples",
        "ALLOW_TRUTHY_PATH", "CACHE_VERSION", "REVERSE_LIMIT",
        "_REVERSE_STATEMENTS",
    ])
    def test_the_symbol_is_not_reintroduced(self, name):
        assert not hasattr(wikidata, name)

    # test_the_demo_path_uses_statements moved to
    # tests/unit/test_entity_linking.py (2026-08-16), with the code it covers.



# ---------------------------------------------------------------------------
# NOISE_FILTER_MODE — selectable 2026-07-19 label heuristic
#
# The two modes are a measured coverage-vs-faithfulness tradeoff, not a fix and
# its bug, so both must stay working. These tests pin the DIFFERENCES between
# them — the cases where switching mode changes the verdict are exactly the
# cases that move the eval numbers.
#
# Modes were renamed 2026-07-27 ("datatype"→"property_type", "string"→
# "label_heuristic") because "datatype" read as Mintaka's answer_type and
# "string" as either the String property type or the string answer_type. The
# old names remain accepted as aliases; the alias tests below pin that.
# ---------------------------------------------------------------------------

@pytest.fixture
def label_mode():
    """Run the body with NOISE_FILTER_MODE = "label_heuristic"."""
    with patch("src.retrieval.wikidata.NOISE_FILTER_MODE", "label_heuristic"):
        yield


class TestNoiseFilterModeLabelHeuristic:
    def test_default_mode_is_property_type(self):
        assert wikidata.NOISE_FILTER_MODE == "property_type"

    @pytest.mark.parametrize("label,datatype", [
        ("image", "CommonsMedia"), ("logo image", "CommonsMedia"),
        ("official website", "Url"),
        ("ISNI", "ExternalId"),
        ("X (Twitter) username", "ExternalId"),
    ])
    def test_label_mode_keeps_what_only_property_type_catches(self, label, datatype, label_mode):
        # The blind spot that motivated the property_type filter: high-volume
        # noise whose label matches none of the label rules. The label mode
        # ignores the datatype entirely, which is exactly the difference.
        assert _is_noise_prop(label, datatype) is False

    @pytest.mark.parametrize("label,datatype", [
        ("image", "CommonsMedia"), ("official website", "Url"), ("ISNI", "ExternalId"),
    ])
    def test_property_type_mode_drops_those_same_labels(self, label, datatype):
        assert _is_noise_prop(label, datatype) is True

    @pytest.mark.parametrize("label", [
        "Commons category",     # no "commons" rule in the 07-19 heuristic
        "topic has template",   # no "topic has" prefix in the 07-19 heuristic
    ])
    def test_label_mode_lacks_the_later_label_rules(self, label, label_mode):
        assert _is_noise_prop(label) is False

    @pytest.mark.parametrize("label", [
        "IMDb ID", "VIAF cluster ID",       # " id" suffix, ungated by property type
        "some external identifier",         # "identifier" substring
        "maintained by WikiProject",        # wikiproject rule
        "topic's main category",            # prefix rule
        "described by source", "facet of",  # exact blocklist
    ])
    def test_label_mode_still_drops_its_own_targets(self, label, label_mode):
        assert _is_noise_prop(label) is True

    @pytest.mark.parametrize("label", [
        "instance of", "capital", "genre", "date of birth", "postal code",
    ])
    def test_label_mode_keeps_real_facts(self, label, label_mode):
        assert _is_noise_prop(label) is False

    def test_mode_is_read_per_call_not_frozen_at_import(self):
        # Sweep tools flip the module attribute between runs; the filter must
        # observe the change without a reimport.
        assert _is_noise_prop("image", "CommonsMedia") is True
        with patch("src.retrieval.wikidata.NOISE_FILTER_MODE", "label_heuristic"):
            assert _is_noise_prop("image", "CommonsMedia") is False
        assert _is_noise_prop("image", "CommonsMedia") is True


class TestLegacyModeAliases:
    """Pre-2026-07-27 mode names must keep selecting the same filter.

    A stale command or result file carrying "datatype"/"string" must not
    silently fall through to the default and mislabel an entire run.
    """

    def test_resolve_maps_old_names(self):
        assert wikidata._resolve_mode("datatype") == "property_type"
        assert wikidata._resolve_mode("string") == "label_heuristic"

    def test_resolve_passes_through_canonical_names(self):
        assert wikidata._resolve_mode("property_type") == "property_type"
        assert wikidata._resolve_mode("label_heuristic") == "label_heuristic"

    def test_legacy_string_alias_still_selects_label_heuristic(self):
        # "image" is the discriminating case: dropped by property type, kept by
        # the label rules.
        with patch("src.retrieval.wikidata.NOISE_FILTER_MODE", "string"):
            assert _is_noise_prop("image", "CommonsMedia") is False

    def test_legacy_datatype_alias_still_selects_property_type(self):
        with patch("src.retrieval.wikidata.NOISE_FILTER_MODE", "datatype"):
            assert _is_noise_prop("image", "CommonsMedia") is True


