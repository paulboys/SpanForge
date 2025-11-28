# Phase 4: Module Architecture

## Package Structure

```
src/weak_labeling/
│
├── __init__.py (101 lines)
│   └── Public API exports
│       ├── Types: LexiconEntry, Span
│       ├── Loaders: load_symptom_lexicon, load_product_lexicon
│       ├── Main: WeakLabeler, weak_label, weak_label_batch
│       ├── Matchers: fuzzy_match, jaccard_token_score, tokenize
│       ├── Negation: detect_negated_regions, is_negated
│       ├── Confidence: compute_confidence, calibrate_threshold
│       └── Validators: deduplicate_spans, is_anatomy_only
│
├── types.py (143 lines)
│   ├── @dataclass LexiconEntry
│   ├── @dataclass Span
│   ├── load_symptom_lexicon()
│   └── load_product_lexicon()
│
├── matchers.py (175 lines)
│   ├── tokenize() → List[(token, start, end)]
│   ├── tokenize_clean() → List[str]
│   ├── jaccard_token_score() → float (0-100)
│   ├── fuzzy_match() → (match, score, index)
│   ├── exact_match() → bool
│   └── Constants: WORD_PATTERN, EMOJI_PATTERN, STOPWORDS
│
├── negation.py (130 lines)
│   ├── detect_negated_regions() → List[(start, end)]
│   ├── is_negated() → bool
│   ├── get_negation_cues() → set
│   └── Constants: NEGATION_TOKENS
│
├── confidence.py (143 lines)
│   ├── compute_confidence() → float (0-1)
│   ├── align_spans() → List[aligned_spans]
│   ├── adjust_confidence_for_negation() → float
│   └── calibrate_threshold() → float
│
├── validators.py (189 lines)
│   ├── is_anatomy_only() → bool
│   ├── validate_span_alignment() → bool
│   ├── deduplicate_spans() → List[Span]
│   ├── filter_overlapping_spans() → List[Span]
│   ├── get_anatomy_tokens() → set
│   └── Constants: ANATOMY_TOKENS
│
└── labeler.py (494 lines)
    ├── class WeakLabeler:
    │   ├── __init__()
    │   ├── label_text() → List[Span]
    │   └── label_batch() → List[List[Span]]
    │
    ├── _match_entities() (core algorithm)
    ├── match_symptoms() → List[Span]
    ├── match_products() → List[Span]
    ├── assemble_spans() → List[Span]
    ├── weak_label() → List[Span]
    ├── weak_label_batch() → List[List[Span]]
    └── persist_weak_labels_jsonl()
```

## Dependency Graph

```
types.py (no dependencies)
    ↓
matchers.py (no dependencies)
    ↓
negation.py
    ├── imports: matchers.tokenize
    │
confidence.py (no dependencies)
    │
validators.py
    ├── imports: types.Span
    │
labeler.py (orchestrator)
    ├── imports: types (LexiconEntry, Span)
    ├── imports: matchers (tokenize, jaccard_token_score, STOPWORDS, EMOJI_PATTERN, HAVE_RAPIDFUZZ)
    ├── imports: negation (detect_negated_regions)
    ├── imports: validators (deduplicate_spans, is_anatomy_only, ANATOMY_TOKENS)
    └── imports: confidence (compute_confidence)
```

## Data Flow

### Input → Output Pipeline

```
User Input
    │
    ├─→ "I have burning sensation after using cream"
    │
    ▼
┌────────────────────────────────────────────────┐
│ 1. Lexicon Loading (types.py)                 │
│    load_symptom_lexicon()                      │
│    load_product_lexicon()                      │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 2. Text Preprocessing (matchers.py)           │
│    - Remove emojis (for symptom matching)     │
│    - Tokenize with positions                  │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 3. Negation Detection (negation.py)           │
│    - Detect negation cues (no, not, never)    │
│    - Build ±5 token windows                   │
│    - Mark negated regions                     │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 4a. Exact Matching (labeler.py)               │
│     - Find exact phrase matches                │
│     - Check word boundaries                    │
│     - Confidence = 1.0                         │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 4b. Fuzzy Matching (labeler.py + matchers.py) │
│     - Sliding window (1-6 tokens)              │
│     - Filter candidates (first token match)    │
│     - Jaccard gate (≥40%)                      │
│     - Last-token alignment check               │
│     - RapidFuzz WRatio (≥88)                   │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 5. Confidence Scoring (confidence.py)          │
│    score = 0.8×fuzzy + 0.2×jaccard             │
└────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────┐
│ 6. Validation (validators.py)                  │
│    - Filter anatomy-only spans                 │
│    - Check span alignment                      │
│    - Deduplicate exact matches                 │
└────────────────────────────────────────────────┘
    │
    ▼
Output: List[Span]
    │
    ├─→ Span(text="burning sensation", start=7, end=24,
    │        label="SYMPTOM", confidence=0.92, negated=False)
    │
    └─→ Span(text="cream", start=37, end=42,
             label="PRODUCT", confidence=1.0, negated=False)
```

## API Comparison

### Old (Function-Based)

```python
from src.weak_label import (
    load_symptom_lexicon,
    load_product_lexicon,
    weak_label,
)

# Manual lexicon loading
symptom_lex = load_symptom_lexicon(Path("data/lexicon/symptoms.csv"))
product_lex = load_product_lexicon(Path("data/lexicon/products.csv"))

# Process text
text = "I have burning sensation"
spans = weak_label(text, symptom_lex, product_lex)
```

### New (Class-Based)

```python
from src.weak_labeling import WeakLabeler
from pathlib import Path

# One-time setup
labeler = WeakLabeler(
    symptom_lexicon_path=Path("data/lexicon/symptoms.csv"),
    product_lexicon_path=Path("data/lexicon/products.csv"),
    fuzzy_threshold=88.0,
    negation_window=5,
)

# Process text (lexicons cached)
text = "I have burning sensation"
spans = labeler.label_text(text)

# Batch processing
spans_batch = labeler.label_batch([text1, text2, text3])
```

## Testing Strategy

```
Unit Tests (Module-Specific)
    ├── test_types.py (data structures & loaders)
    ├── test_matchers.py (fuzzy/exact matching)
    ├── test_negation.py (negation detection)
    ├── test_confidence.py (confidence scoring)
    ├── test_validators.py (span filtering)
    └── test_labeler.py (orchestration logic)

Integration Tests
    ├── test_weak_label.py (backward compatibility)
    ├── test_pipeline.py (pipeline integration)
    └── test_end_to_end.py (full workflow)

Edge Cases
    ├── test_boundary_conditions.py
    ├── test_malformed_inputs.py
    ├── test_negation_edge.py
    ├── test_overlap_scenarios.py
    └── test_unicode_emoji.py
```

## Performance Characteristics

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| **Lexicon Loading** | O(n) | 20-50ms |
| **Tokenization** | O(m) | 0.1-0.5ms |
| **Negation Detection** | O(m×k) | 0.2-1ms |
| **Exact Matching** | O(n×m) | 1-5ms |
| **Fuzzy Matching** | O(m²×n) | 5-20ms |
| **Validation** | O(s²) | 0.1-0.5ms |
| **Total (1 text)** | - | **8-30ms** |

Where:
- n = lexicon size (~500-2000 entries)
- m = text length in tokens (~20-100 tokens)
- k = negation window size (5 tokens)
- s = number of candidate spans (~5-20 spans)

## Memory Usage

```
Component                Memory
─────────────────────────────────
Lexicon (symptoms)      ~200 KB
Lexicon (products)      ~50 KB
Tokenizer cache         ~100 KB
RapidFuzz state         ~50 KB
Span objects (10)       ~5 KB
─────────────────────────────────
Total per instance      ~400 KB
```

**Note**: WeakLabeler reuses lexicons across calls → efficient batching.

---

**Summary**: Phase 4 delivered a clean, modular architecture with zero breaking changes and 100% test pass rate.
