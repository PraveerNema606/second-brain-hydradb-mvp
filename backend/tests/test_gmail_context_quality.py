"""
Email content-quality improvements (Phase B, feature 2).

Extends the Phase-A _clean_email_text trimming with retrieval-noise
removal: zero-width characters, "view in browser" lines, decorative
separators, "[image: ...]" placeholders, and tracking-only URL lines —
while preserving real content (including plain content URLs) and the
existing signature/footer/unsubscribe/legal behavior.
"""

from gmail_oauth import _clean_email_text


class TestZeroWidthStripping:
    def test_zero_width_chars_removed(self):
        text = "Hello\u200b team,\nthe\u00ad report is ready"
        out = _clean_email_text(text)
        assert "\u200b" not in out
        assert "\u00ad" not in out
        assert "Hello team," in out
        assert "report is ready" in out


class TestViewInBrowser:
    def test_view_in_browser_line_removed(self):
        text = "View this email in your browser\nHere is the weekly digest\nThanks"
        out = _clean_email_text(text)
        assert "View this email in your browser" not in out
        assert "weekly digest" in out

    def test_having_trouble_line_removed(self):
        text = "Having trouble viewing this email?\nReal content here"
        out = _clean_email_text(text)
        assert "Having trouble" not in out
        assert "Real content here" in out


class TestSeparatorsAndPlaceholders:
    def test_decorative_separator_removed(self):
        text = "Top line\n==========\nBottom line"
        out = _clean_email_text(text)
        assert "Top line" in out
        assert "Bottom line" in out
        assert "====" not in out

    def test_dashes_separator_removed_but_not_signature_dashes(self):
        # A 4-dash decorative rule is removed; "--Alice" (content) stays.
        text = "Intro\n----\n--Alice mentioned this"
        out = _clean_email_text(text)
        assert "Intro" in out
        assert "--Alice mentioned this" in out
        assert "----" not in out

    def test_image_placeholder_removed(self):
        text = "[image: company-logo.png]\nThe meeting is at 3pm"
        out = _clean_email_text(text)
        assert "company-logo" not in out
        assert "The meeting is at 3pm" in out


class TestTrackingUrls:
    def test_tracking_url_line_removed(self):
        text = "Click below to confirm\nhttps://click.example.com/CL0/aHR0cHM?utm_source=news\nThanks"
        out = _clean_email_text(text)
        assert "click.example.com" not in out
        assert "Click below to confirm" in out
        assert "Thanks" in out

    def test_plain_content_url_preserved(self):
        text = "Repo is here:\nhttps://github.com/PraveerNema606/second-brain-hydradb-mvp\nPlease review"
        out = _clean_email_text(text)
        assert "github.com/PraveerNema606/second-brain-hydradb-mvp" in out
        assert "Please review" in out

    def test_url_with_text_not_removed(self):
        # A sentence that merely contains a tracking-ish URL is NOT a
        # standalone URL line, so it is preserved.
        text = "See https://click.example.com/CL0/x?utm_source=a for details"
        out = _clean_email_text(text)
        assert "for details" in out


class TestPhaseANotRegressed:
    def test_signature_still_cut(self):
        text = "Body content here\n--\nSig Name\nCompany"
        out = _clean_email_text(text)
        assert "Body content here" in out
        assert "Sig Name" not in out

    def test_mobile_footer_still_removed(self):
        text = "Please review\nSent from my iPhone"
        out = _clean_email_text(text)
        assert "Please review" in out
        assert "iPhone" not in out

    def test_safety_net_still_returns_original(self):
        text = "Sent from my iPhone"
        assert _clean_email_text(text) == "Sent from my iPhone"

    def test_plain_content_unchanged(self):
        text = "Line one\nLine two\nLine three"
        assert _clean_email_text(text) == "Line one\nLine two\nLine three"
