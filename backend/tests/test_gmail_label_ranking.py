"""
Gmail label-aware retrieval and ranking (Phase B, feature 1).

Covers:
  - infer_gmail_label: word->label-id mapping, gated on email context.
  - _label_match_score: card.labels vs bias["label"], case-insensitive.
  - _metadata_bias_score ignores the "label" key (no double counting).
  - rerank_chunks wires W_LABEL_MATCH independently in hybrid mode and
    uses label as a tiebreaker in exact/default modes, while preserving
    existing ordering when no label bias is present.
  - main._resolve_query_rewrite folds an inferred label into metadata_bias.
"""

from unittest.mock import patch

from search_utils import (
    W_LABEL_MATCH,
    _label_match_score,
    _metadata_bias_score,
    infer_gmail_label,
    rerank_chunks,
)


# =====================================================================
# infer_gmail_label
# =====================================================================
class TestInferGmailLabel:
    def test_examples_from_spec(self):
        assert infer_gmail_label("emails in inbox") == "INBOX"
        assert infer_gmail_label("messages in promotions") == "CATEGORY_PROMOTIONS"
        assert infer_gmail_label("important emails") == "IMPORTANT"
        assert infer_gmail_label("starred emails") == "STARRED"
        assert infer_gmail_label("mail from updates") == "CATEGORY_UPDATES"

    def test_requires_email_context(self):
        # No email-ish token -> no label inferred (avoids false positives).
        assert infer_gmail_label("an important decision was made") is None
        assert infer_gmail_label("our social media strategy") is None
        assert infer_gmail_label("send it to the forum") is None

    def test_none_and_empty(self):
        assert infer_gmail_label("") is None
        assert infer_gmail_label("what did Alice say about Kafka") is None

    def test_category_words(self):
        assert infer_gmail_label("show me social emails") == "CATEGORY_SOCIAL"
        assert infer_gmail_label("forum messages") == "CATEGORY_FORUMS"

    def test_inbox_token_is_both_context_and_label(self):
        assert infer_gmail_label("inbox") == "INBOX"


# =====================================================================
# _label_match_score
# =====================================================================
class TestLabelMatchScore:
    def test_match_case_insensitive(self):
        card = {"labels": ["INBOX", "CATEGORY_PROMOTIONS"]}
        assert _label_match_score(card, {"label": "inbox"}) == 1
        assert _label_match_score(card, {"label": "CATEGORY_PROMOTIONS"}) == 1

    def test_no_match(self):
        card = {"labels": ["INBOX"]}
        assert _label_match_score(card, {"label": "STARRED"}) == 0

    def test_slack_card_without_labels(self):
        card = {"channel": "general"}  # no labels key
        assert _label_match_score(card, {"label": "INBOX"}) == 0

    def test_no_label_in_bias(self):
        card = {"labels": ["INBOX"]}
        assert _label_match_score(card, {"channel": "x"}) == 0
        assert _label_match_score(card, None) == 0


# =====================================================================
# _metadata_bias_score must ignore the label key
# =====================================================================
class TestBiasIgnoresLabel:
    def test_label_does_not_inflate_generic_bias(self):
        card = {"labels": ["INBOX"], "channel": "general"}
        # Only the channel match should count; the label key is skipped.
        assert _metadata_bias_score(card, {"channel": "general", "label": "INBOX"}) == 1
        # Label alone contributes nothing to the generic bias.
        assert _metadata_bias_score(card, {"label": "INBOX"}) == 0


# =====================================================================
# rerank_chunks: label wiring
# =====================================================================
def _chunk(idx, text, labels=None, ts=0.0):
    card = {"index": idx, "document_type": "email"}
    if labels is not None:
        card["labels"] = labels
    return {
        "text": text,
        "source_card": card,
        "original_index": idx,
        "timestamp_float": ts,
    }


class TestRerankLabelWiring:
    def test_hybrid_promotes_label_match(self):
        # Two chunks, identical otherwise; the one in the biased label
        # should sort first in hybrid mode via W_LABEL_MATCH.
        chunks = [
            _chunk(1, "quarterly report attached", labels=["CATEGORY_PROMOTIONS"]),
            _chunk(2, "quarterly report attached", labels=["INBOX"]),
        ]
        ranked, _ = rerank_chunks(
            chunks,
            terms=["quarterly", "report"],
            mode="hybrid",
            top_k=5,
            metadata_bias={"label": "INBOX"},
        )
        assert ranked[0]["source_card"]["labels"] == ["INBOX"]
        # The label contribution shows up in the debug breakdown.
        assert ranked[0]["_debug_score"]["label_match"] == 1
        assert ranked[1]["_debug_score"]["label_match"] == 0

    def test_hybrid_label_weight_value(self):
        chunks = [_chunk(1, "hello world", labels=["INBOX"])]
        ranked, _ = rerank_chunks(
            chunks,
            terms=[],
            mode="hybrid",
            top_k=5,
            metadata_bias={"label": "INBOX"},
        )
        # No keyword/subject/recency signal -> score is exactly the label term.
        assert ranked[0]["_debug_score"]["hybrid_score"] == float(W_LABEL_MATCH)

    def test_default_mode_uses_label_tiebreaker(self):
        chunks = [
            _chunk(1, "alpha", labels=["INBOX"]),
            _chunk(2, "beta", labels=["STARRED"]),
        ]
        ranked, _ = rerank_chunks(
            chunks,
            terms=[],
            mode="default",
            top_k=5,
            metadata_bias={"label": "STARRED"},
        )
        # The STARRED chunk (original_index 2) is promoted above index 1.
        assert ranked[0]["original_index"] == 2

    def test_no_label_bias_preserves_order(self):
        # When no label is biased, default-mode order is unchanged
        # (collapses to original_index) -- regression guard.
        chunks = [
            _chunk(1, "alpha", labels=["INBOX"]),
            _chunk(2, "beta", labels=["STARRED"]),
            _chunk(3, "gamma", labels=["IMPORTANT"]),
        ]
        ranked, _ = rerank_chunks(
            chunks, terms=[], mode="default", top_k=5, metadata_bias=None
        )
        assert [c["original_index"] for c in ranked] == [1, 2, 3]

    def test_exact_mode_label_is_secondary_to_hits(self):
        # Body keyword hits dominate; label only breaks ties. Chunk 2 has
        # MORE distinct-term hits, so it wins even though chunk 1 matches
        # the biased label.
        chunks = [
            _chunk(1, "deadline", labels=["INBOX"]),
            _chunk(2, "deadline on friday", labels=["STARRED"]),
        ]
        ranked, matched = rerank_chunks(
            chunks,
            terms=["deadline", "friday"],
            mode="exact",
            top_k=5,
            metadata_bias={"label": "INBOX"},
        )
        assert matched == 2
        assert ranked[0]["original_index"] == 2  # more hits wins over label

    def test_exact_mode_label_breaks_ties(self):
        # Equal hits -> the label match decides ordering.
        chunks = [
            _chunk(1, "deadline", labels=["STARRED"]),
            _chunk(2, "deadline", labels=["INBOX"]),
        ]
        ranked, _ = rerank_chunks(
            chunks,
            terms=["deadline"],
            mode="exact",
            top_k=5,
            metadata_bias={"label": "INBOX"},
        )
        assert ranked[0]["original_index"] == 2  # INBOX-labeled tiebreak win


# =====================================================================
# main._resolve_query_rewrite folds the inferred label into bias
# =====================================================================
class TestResolveQueryRewriteLabel:
    def _no_inference_rewrite(self):
        return {
            "inferred_channel": None,
            "channel_confidence": None,
            "inferred_person": None,
            "person_confidence": None,
            "retrieval_biases_applied": [],
            "metadata_bias": None,
        }

    def test_label_added_to_metadata_bias(self):
        import main
        from main import QueryRequest

        req = QueryRequest(question="important emails about the launch")
        with patch("main.rewrite_query", return_value=self._no_inference_rewrite()):
            out = main._resolve_query_rewrite(req)
        assert out["metadata_bias"] == {"label": "IMPORTANT"}

    def test_no_label_leaves_bias_none(self):
        import main
        from main import QueryRequest

        req = QueryRequest(question="what did Alice decide about Kafka")
        with patch("main.rewrite_query", return_value=self._no_inference_rewrite()):
            out = main._resolve_query_rewrite(req)
        assert out["metadata_bias"] is None