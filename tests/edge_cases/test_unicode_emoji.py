"""Edge case tests for unicode and emoji handling.

Tests span extraction with unicode characters, combining marks,
emojis, and multi-byte encodings.
"""
import pytest
import unittest
from tests.base import WeakLabelTestBase
from tests.assertions import SpanAsserter
from src.weak_label import weak_label


class TestUnicodeHandling(WeakLabelTestBase):
    """Test span extraction with unicode text."""
    
    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()
        self.span_asserter = SpanAsserter(self)
    
    def test_accented_characters(self):
        """Test text with accented latin characters."""
        text = "Patiënt reports démangeaisons (itching)"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        # Should detect "itching" regardless of surrounding unicode
        symptom_texts = [s.text.lower() for s in spans if s.label == "SYMPTOM"]
        self.assertTrue(any("itching" in t for t in symptom_texts),
                       "Should detect 'itching' despite unicode context")
    
    def test_cjk_characters(self):
        """Test text with CJK (Chinese, Japanese, Korean) characters."""
        text = "患者说用后发痒 (itching reported)"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        # Should still extract English symptom
        if spans:
            for span in spans:
                self.span_asserter.assert_boundaries_valid(text, 
                    [{"start": span.start, "end": span.end, "label": span.label}])
    
    def test_combining_marks(self):
        """Test text with unicode combining marks (e.g., diacritics)."""
        text = "Redness and burning sensation"  # Could be "Réndess" with combining acute
        base_text_with_combining = "Re\u0301dness"  # R + e + combining acute + dness
        
        spans = [{"start": 0, "end": len(base_text_with_combining), 
                 "label": "SYMPTOM", "text": base_text_with_combining}]
        
        # Should handle combining marks in boundary validation
        self.span_asserter.assert_boundaries_valid(base_text_with_combining, spans)


class TestEmojiHandling(WeakLabelTestBase):
    """Test span extraction with emoji characters."""
    
    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()
        self.product_lexicon = self.create_product_lexicon()
        self.span_asserter = SpanAsserter(self)
    
    def test_emoji_in_text(self):
        """Test text containing emoji."""
        text = "Patient reports itching 🔥 after cream use 😢"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        # Verify boundaries are valid despite emoji presence
        for span in spans:
            span_dict = {"start": span.start, "end": span.end, 
                        "label": span.label, "text": span.text}
            self.span_asserter.assert_boundaries_valid(text, [span_dict])
            self.span_asserter.assert_text_slices_match(text, [span_dict])
    
    def test_emoji_adjacent_to_span(self):
        """Test span immediately adjacent to emoji."""
        text = "🤒itching🔥"
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        
        if spans:
            for span in spans:
                # Span should not include emoji unless intentional
                self.assertNotIn("🤒", span.text, "Span should not capture leading emoji")
                self.assertNotIn("🔥", span.text, "Span should not capture trailing emoji")


@pytest.mark.parametrize("text,expected_symptoms", [
    ("Patiënt reports itching", ["itching"]),
    ("Démangeaisons sévères (severe itching)", ["itching"]),
    ("Rötung und Brennen", []),  # German: "Redness and burning" - no English match
    ("患者说用后发痒", []),  # Chinese - no English match expected
    ("Patient has 🔥 burning sensation", ["burning"]),
    ("Pruritus 😢", []),  # May match if lexicon contains "pruritus"
])
def test_unicode_symptom_detection(text, expected_symptoms):
    """Parametrized unicode symptom detection."""
    from pathlib import Path
    from src.weak_label import load_symptom_lexicon, load_product_lexicon
    
    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)
    
    spans = weak_label(text, symptom_lexicon, product_lexicon)
    symptom_texts = [s.text.lower() for s in spans if s.label == "SYMPTOM"]
    
    for expected in expected_symptoms:
        assert any(expected in t for t in symptom_texts), \
            f"Expected symptom '{expected}' not found in {symptom_texts}"


@pytest.mark.parametrize("text,valid", [
    ("Normal ASCII text", True),
    ("Tëxt with äccënts", True),
    ("中文字符", True),
    ("Emoji 🔥🤒😢", True),
    ("Mixed 中文 and English", True),
    ("Combining: e\u0301", True),  # e + combining acute
    ("Zalgo t̴̡̢̛̗̙̫̪̺̯̞̻̠̓̇̇̀̋͒̿̓̉̚̚͜͠ͅë̸̛̜̰̙̝̱́̋̈́̿̽x̷̢̧̛̣̝̺̪̞̬̳͈͙̐́̏̐̈́̐̇̓̃͘͝t̸̨̨̺̩̯̜̟̪̱̠̻̱̥̆̅͋̄̌̚͝", True),
])
def test_unicode_boundary_validation(text, valid):
    """Parametrized unicode boundary validation."""
    test_case = unittest.TestCase()
    test_case.setUp = lambda: None
    span_asserter = SpanAsserter(test_case)
    
    # Span covering full text
    spans = [{"start": 0, "end": len(text), "label": "SYMPTOM", "text": text}]
    
    if valid:
        span_asserter.assert_boundaries_valid(text, spans)
        span_asserter.assert_text_slices_match(text, spans)
    else:
        with pytest.raises(AssertionError):
            span_asserter.assert_boundaries_valid(text, spans)
