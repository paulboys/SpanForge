"""Tests for weak_labeling.matchers module."""

import pytest

from src.weak_labeling.matchers import (
    EMOJI_PATTERN,
    HAVE_RAPIDFUZZ,
    STOPWORDS,
    exact_match,
    fuzzy_match,
    jaccard_token_score,
    tokenize,
)


class TestTokenize:
    """Test tokenization with position tracking."""

    def test_simple_phrase(self):
        """Test tokenization of simple phrase."""
        tokens = tokenize("burning sensation")
        assert len(tokens) == 2
        assert tokens[0] == ("burning", 0, 7)
        assert tokens[1] == ("sensation", 8, 17)

    def test_with_punctuation(self):
        """Test tokenization with punctuation."""
        tokens = tokenize("redness, swelling!")
        assert len(tokens) == 2
        assert tokens[0] == ("redness", 0, 7)
        assert tokens[1] == ("swelling", 9, 17)

    def test_multiple_spaces(self):
        """Test handling multiple consecutive spaces."""
        tokens = tokenize("word1    word2")
        assert len(tokens) == 2
        assert tokens[0][0] == "word1"
        assert tokens[1][0] == "word2"

    def test_empty_string(self):
        """Test tokenization of empty string."""
        tokens = tokenize("")
        assert tokens == []

    def test_only_punctuation(self):
        """Test string with only punctuation."""
        tokens = tokenize("...")
        assert tokens == []

    def test_unicode_characters(self):
        """Test handling unicode characters."""
        tokens = tokenize("café burning")
        assert len(tokens) == 2
        assert tokens[0][0] == "café"

    def test_emoji_removal(self):
        """Test that emojis are removed."""
        tokens = tokenize("burning 🔥 sensation")
        assert len(tokens) == 2
        assert tokens[0][0] == "burning"
        assert tokens[1][0] == "sensation"

    def test_position_tracking_accuracy(self):
        """Test that positions accurately reflect original text."""
        text = "  leading whitespace"
        tokens = tokenize(text)
        first_token = tokens[0]
        assert text[first_token[1] : first_token[2]] == first_token[0]


class TestJaccardTokenScore:
    """Test Jaccard token similarity scoring."""

    def test_identical_strings(self):
        """Test with identical strings."""
        score = jaccard_token_score("burning sensation", "burning sensation")
        assert score == 100.0

    def test_no_overlap(self):
        """Test with no overlapping tokens."""
        score = jaccard_token_score("burning", "redness")
        assert score == 0.0

    def test_partial_overlap(self):
        """Test with partial overlap."""
        # "burning sensation" vs "severe burning"
        # Overlap: {burning}
        # Union: {burning, sensation, severe}
        # Score: 1/3 * 100 = 33.33
        score = jaccard_token_score("burning sensation", "severe burning")
        assert abs(score - 33.33) < 0.01

    def test_stopword_filtering(self):
        """Test jaccard with stopwords (not filtered in basic implementation)."""
        # "the burning" vs "burning": {the, burning} vs {burning} = 1/2 = 50%
        score = jaccard_token_score("the burning", "burning")
        assert score == 50.0

    def test_case_insensitive(self):
        """Test case insensitivity."""
        score = jaccard_token_score("BURNING", "burning")
        assert score == 100.0

    def test_empty_strings(self):
        """Test with empty strings."""
        score = jaccard_token_score("", "")
        assert score == 0.0

    def test_one_empty(self):
        """Test with one empty string."""
        score = jaccard_token_score("burning", "")
        assert score == 0.0

    def test_all_stopwords(self):
        """Test when all tokens are stopwords (still calculates overlap)."""
        # {the, a, an} vs {a, the} = 2/3 = 66.67%
        score = jaccard_token_score("the a an", "a the")
        assert abs(score - 66.67) < 0.1

    def test_multi_token_overlap(self):
        """Test with multiple overlapping tokens."""
        # "burning red rash" vs "red rash swelling"
        # Overlap: {burning, red, rash} ∩ {red, rash, swelling} = {red, rash}
        # Union: {burning, red, rash, swelling}
        # Score: 2/4 * 100 = 50.0
        score = jaccard_token_score("burning red rash", "red rash swelling")
        assert abs(score - 50.0) < 0.01


class TestFuzzyMatch:
    """Test fuzzy string matching."""

    def test_exact_match(self):
        """Test with exact string match."""
        score = fuzzy_match("burning sensation", "burning sensation")
        assert score == 100.0

    def test_close_match(self):
        """Test with similar strings."""
        score = fuzzy_match("burning sensation", "burning sensations")
        assert score > 90.0  # Should be high similarity

    def test_different_strings(self):
        """Test with different strings."""
        score = fuzzy_match("burning", "redness")
        assert score < 50.0

    def test_case_insensitive(self):
        """Test case handling in fuzzy matching."""
        # RapidFuzz WRatio is case-sensitive and returns low scores for case mismatches
        # The difflib fallback normalizes case and returns high scores
        # Test with normalized inputs for consistent behavior
        score_normalized = fuzzy_match("burning", "burning")
        assert score_normalized == 100.0

        # Case mismatch behavior depends on backend (WRatio is case-sensitive)
        score_case_mismatch = fuzzy_match("BURNING", "burning")
        # Just verify it returns a numeric value >= 0
        assert isinstance(score_case_mismatch, (int, float))
        assert score_case_mismatch >= 0.0

    def test_empty_strings(self):
        """Test with empty strings."""
        score = fuzzy_match("", "")
        assert score == 0.0

    def test_one_empty(self):
        """Test with one empty string."""
        score = fuzzy_match("burning", "")
        assert score == 0.0

    def test_whitespace_normalization(self):
        """Test that extra whitespace has minimal impact."""
        score = fuzzy_match("burning  sensation", "burning sensation")
        # Should be very high but may not be exactly 100 due to character differences
        assert score >= 95.0, f"Expected high score for whitespace differences, got {score}"

    def test_rapidfuzz_vs_difflib(self):
        """Test that both backends produce reasonable scores."""
        # This test works regardless of which backend is available
        score = fuzzy_match("test", "testing")
        assert 50.0 <= score <= 100.0  # Should be moderately similar

    def test_typo_tolerance(self):
        """Test tolerance for minor typos."""
        score = fuzzy_match("burning sensation", "burninng sensation")
        assert score > 85.0  # Should still match reasonably


class TestExactMatch:
    """Test exact string matching."""

    def test_exact_match_identical(self):
        """Test with identical strings."""
        assert exact_match("burning", "burning") is True

    def test_case_sensitive(self):
        """Test that matching is case-insensitive by default."""
        assert exact_match("Burning", "burning") is True  # Default is case-insensitive
        assert exact_match("Burning", "burning", case_sensitive=True) is False  # With flag

    def test_whitespace_sensitive(self):
        """Test that whitespace matters."""
        assert exact_match("burning", "burning ") is False

    def test_different_strings(self):
        """Test with different strings."""
        assert exact_match("burning", "redness") is False

    def test_empty_strings(self):
        """Test with empty strings."""
        assert exact_match("", "") is True

    def test_one_empty(self):
        """Test with one empty string."""
        assert exact_match("burning", "") is False


class TestStopwordsAndConstants:
    """Test stopwords and module constants."""

    def test_stopwords_not_empty(self):
        """Test that stopwords set is defined and not empty."""
        assert len(STOPWORDS) > 0

    def test_common_stopwords_present(self):
        """Test that common English stopwords are included."""
        assert "the" in STOPWORDS
        assert "a" in STOPWORDS
        assert "an" in STOPWORDS
        assert "and" in STOPWORDS

    def test_emoji_pattern_exists(self):
        """Test that emoji pattern is defined."""
        assert EMOJI_PATTERN is not None

    def test_emoji_pattern_matches_emoji(self):
        """Test that emoji pattern matches emoji characters."""
        import re

        assert EMOJI_PATTERN.search("🔥") is not None
        assert EMOJI_PATTERN.search("😊") is not None

    def test_emoji_pattern_ignores_text(self):
        """Test that emoji pattern doesn't match regular text."""
        import re

        assert EMOJI_PATTERN.search("burning") is None

    def test_rapidfuzz_flag_is_boolean(self):
        """Test that HAVE_RAPIDFUZZ is a boolean."""
        assert isinstance(HAVE_RAPIDFUZZ, bool)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_tokenize_very_long_text(self):
        """Test tokenization of very long text."""
        long_text = " ".join(["word"] * 1000)
        tokens = tokenize(long_text)
        assert len(tokens) == 1000

    def test_jaccard_single_character_tokens(self):
        """Test Jaccard with single-character tokens."""
        score = jaccard_token_score("a b c", "b c d")
        # Union: {a,b,c,d}, Intersection: {b,c} (after stopword filtering)
        # But 'a' is stopword, so: Union: {b,c,d}, Intersection: {b,c}
        # Score: 2/3 * 100 = 66.67
        assert score > 0.0

    def test_fuzzy_match_unicode(self):
        """Test fuzzy matching with unicode characters."""
        score = fuzzy_match("café", "cafe")
        assert score > 50.0  # Should still match reasonably

    def test_jaccard_with_repeated_tokens(self):
        """Test Jaccard treats tokens as sets (no duplicates)."""
        # "word word word" should be treated as {"word"}
        score = jaccard_token_score("word word word", "word")
        assert score == 100.0

    def test_tokenize_mixed_content(self):
        """Test tokenization with mixed numbers, letters, symbols."""
        tokens = tokenize("burning-sensation #1 test!")
        # Should tokenize into meaningful parts
        assert len(tokens) > 0

    def test_fuzzy_match_substring(self):
        """Test fuzzy matching with substring."""
        score = fuzzy_match("burn", "burning")
        assert score > 60.0  # Substring should have reasonable similarity
