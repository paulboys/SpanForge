from pathlib import Path
import sys
import pathlib
import json
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.weak_label import load_symptom_lexicon, load_product_lexicon, weak_label, persist_weak_labels_jsonl


def test_weak_label_symptom_and_negation():
    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)
    assert symptom_lexicon, "Symptom lexicon should not be empty for test"
    text = "I had a mild headache after using the cream but no skin rash developed."
    spans = weak_label(text, symptom_lexicon, product_lexicon)
    symptom_texts = {s.text.lower(): s for s in spans if s.label == "SYMPTOM"}
    assert "headache" in symptom_texts, "Should detect headache"
    if "skin rash" in symptom_texts:
        assert symptom_texts["skin rash"].negated, "Skin rash should be negated"


def test_product_lexicon_matching():
    text = "This moisturizing cream caused a headache."
    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)
    spans = weak_label(text, symptom_lexicon, product_lexicon)
    labels = [s.label for s in spans]
    assert "PRODUCT" in labels, "PRODUCT span expected"
    product_spans = [s for s in spans if s.label == "PRODUCT"]
    assert any("moisturizing cream" in s.text.lower() for s in product_spans), "Should match moisturizing cream"
    if product_spans:
        assert product_spans[0].sku is not None, "Product should have SKU"


def test_jsonl_persistence(tmp_path):
    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)
    texts = ["I had a headache.", "The serum caused itching."]
    spans_batch = [weak_label(t, symptom_lexicon, product_lexicon) for t in texts]
    output_file = tmp_path / "weak_labels.jsonl"
    persist_weak_labels_jsonl(texts, spans_batch, output_file)
    assert output_file.exists(), "JSONL file should be created"
    with output_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 2, "Should have 2 JSONL records"
        record = json.loads(lines[0])
        assert "text" in record and "spans" in record, "Record should have text and spans"


def test_no_short_spurious_matches():
    """Test that very short tokens (1-2 chars) don't produce spurious fuzzy matches."""
    symptom_lex_path = Path("data/lexicon/symptoms.csv")
    product_lex_path = Path("data/lexicon/products.csv")
    symptom_lexicon = load_symptom_lexicon(symptom_lex_path)
    product_lexicon = load_product_lexicon(product_lex_path)
    
    # Text with single-letter words that shouldn't match anything
    text = "I got a terrible headache after using the moisturizing cream."
    spans = weak_label(text, symptom_lexicon, product_lexicon)
    
    # Verify no single-letter or two-letter spurious matches
    for span in spans:
        if span.confidence < 1.0:  # fuzzy match
            assert len(span.text) >= 3, f"Spurious short match detected: '{span.text}' -> {span.canonical} (conf: {span.confidence})"
    
    # Should still find valid matches
    symptom_texts = [s.text.lower() for s in spans if s.label == "SYMPTOM"]
    assert any("headache" in t for t in symptom_texts), "Should still find 'headache' symptom"

