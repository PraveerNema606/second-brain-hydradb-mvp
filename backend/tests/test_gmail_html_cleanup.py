"""
Gmail HTML cleanup + body trimming (audit item 3).

Covers _strip_html (newline-preserving, script/style removal) and
_clean_email_text (signature / mobile-footer / unsubscribe / legal
trimming) with a strong bias toward PRESERVING real content and a
safety net that never empties a non-empty body.
"""

from gmail_oauth import _clean_email_text, _strip_html


# =====================================================================
# _strip_html
# =====================================================================
class TestStripHtml:
    def test_empty(self):
        assert _strip_html("") == ""

    def test_drops_script_and_style(self):
        raw = "<style>.x{color:red}</style><p>Hello</p><script>alert(1)</script><p>World</p>"
        out = _strip_html(raw)
        assert "Hello" in out
        assert "World" in out
        assert "color:red" not in out
        assert "alert(1)" not in out
        assert "<" not in out

    def test_block_tags_become_newlines(self):
        out = _strip_html("<p>Line one</p><p>Line two</p>")
        assert out == "Line one\nLine two"

    def test_br_becomes_newline(self):
        out = _strip_html("a<br>b<br/>c")
        assert out == "a\nb\nc"

    def test_entities_unescaped(self):
        out = _strip_html("<p>Tom &amp; Jerry</p>")
        assert "Tom & Jerry" in out

    def test_inline_tags_stripped_but_text_joined(self):
        out = _strip_html("<p>Hello <b>bold</b> world</p>")
        assert "Hello bold world" in out


# =====================================================================
# _clean_email_text
# =====================================================================
class TestCleanEmailText:
    def test_empty_passthrough(self):
        assert _clean_email_text("") == ""
        assert _clean_email_text("   ") == "   "

    def test_signature_delimiter_cut(self):
        text = "Hi team,\nShipping Friday.\n--\nAlice Smith\nAcme Corp"
        out = _clean_email_text(text)
        assert "Shipping Friday." in out
        assert "Alice Smith" not in out
        assert "Acme Corp" not in out

    def test_does_not_cut_double_dash_with_text(self):
        # "--Alice" is NOT a standalone signature delimiter.
        text = "Hello team,\nQuarterly review on Friday.\n--Alice"
        out = _clean_email_text(text)
        assert "Quarterly review on Friday." in out
        assert "--Alice" in out

    def test_mobile_footer_removed(self):
        text = "Can you review the doc?\nSent from my iPhone"
        out = _clean_email_text(text)
        assert "Can you review the doc?" in out
        assert "iPhone" not in out

    def test_get_outlook_footer_removed(self):
        text = "Approved.\nGet Outlook for iOS"
        out = _clean_email_text(text)
        assert "Approved." in out
        assert "Outlook" not in out

    def test_unsubscribe_trailer_cut_in_latter_half(self):
        text = "\n".join(
            [
                "Hello team",
                "Here is the weekly update",
                "Thanks for reading",
                "Unsubscribe here",
                "Acme Inc",
                "123 Main St",
            ]
        )
        out = _clean_email_text(text)
        assert "weekly update" in out
        assert "Unsubscribe" not in out
        assert "123 Main St" not in out

    def test_top_of_body_unsubscribe_link_preserves_content(self):
        # Newsletters often put "View in browser | Unsubscribe" at the
        # very top. That must NOT nuke the body below it.
        text = "\n".join(
            [
                "View in browser Unsubscribe",
                "Headline of the week",
                "Body paragraph one",
                "Body paragraph two",
                "Closing line",
            ]
        )
        out = _clean_email_text(text)
        assert "Headline of the week" in out
        assert "Body paragraph one" in out
        assert "Closing line" in out
        assert "View in browser" not in out  # the stray link line is dropped

    def test_legal_disclaimer_cut(self):
        text = "\n".join(
            [
                "Quarterly numbers attached.",
                "Let me know if you have questions.",
                "Best,",
                "Alice",
                "This email is confidential and intended for the named recipient only.",
                "Please do not forward.",
            ]
        )
        out = _clean_email_text(text)
        assert "Quarterly numbers attached." in out
        assert "confidential" not in out
        assert "do not forward" not in out

    def test_safety_net_returns_original_when_cleanup_empties(self):
        # A body that is ONLY a mobile footer would clean to empty; the
        # safety net returns the original rather than dropping the doc.
        text = "Sent from my iPhone"
        out = _clean_email_text(text)
        assert out == "Sent from my iPhone"

    def test_real_content_preserved(self):
        text = "Line one\nLine two\nLine three"
        assert _clean_email_text(text) == "Line one\nLine two\nLine three"
