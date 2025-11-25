# Testing Guide

Comprehensive guide to SpanForge's test infrastructure.

## Overview

SpanForge implements **144 tests** across 4 categories:

| Category | Count | Purpose |
|----------|-------|---------|
| **Core Tests** | 16 | Basic functionality validation |
| **Edge Case Tests** | 98 | Boundary conditions, unicode, negation |
| **Integration Tests** | 26 | End-to-end workflows, scale, performance |
| **Curation Tests** | 4 | Annotation pipeline, Label Studio export |

**Test Status:** ✅ 144/144 passing (100%)

## Quick Start

### Run All Tests

```bash
# Run full suite
pytest

# With verbose output
pytest -v

# With coverage
pytest --cov=src --cov-report=html
```

### Run Specific Categories

```bash
# Core tests only
pytest tests/test_weak_label_core.py -v

# Edge cases only
pytest tests/test_weak_label_edge.py -v

# Integration tests only
pytest tests/test_integration.py -v

# Curation tests only
pytest tests/test_curation.py -v
```

### Run Specific Tests

```bash
# Single test function
pytest tests/test_weak_label_core.py::test_match_symptoms_basic -v

# Pattern matching
pytest -k "negation" -v  # All negation tests
pytest -k "emoji" -v     # All emoji tests
pytest -k "unicode" -v   # All unicode tests
```

## Test Architecture

### Directory Structure

```
tests/
├── conftest.py                    # Pytest fixtures
├── test_weak_label_core.py        # 16 core tests
├── test_weak_label_edge.py        # 98 edge case tests
├── test_integration.py            # 26 integration tests
└── test_curation.py               # 4 curation tests
```

### Test Composition Pattern

Tests use **composition** for shared setup:

```python
# Base test class
class WeakLabelTestBase:
    """Shared fixtures and utilities."""
    
    @pytest.fixture
    def symptom_lexicon(self):
        return ["itching", "redness", "burning sensation"]
    
    @pytest.fixture
    def product_lexicon(self):
        return ["Lotion X", "Cream Y"]

# Edge case test class (inherits base)
class TestWeakLabelEdgeCases(WeakLabelTestBase):
    """Edge case tests with shared fixtures."""
    
    def test_emoji_handling(self, symptom_lexicon):
        text = "Patient has 😊 severe itching"
        spans = match_symptoms(text, symptom_lexicon)
        assert len(spans) > 0
```

**Benefits:**
- Avoid duplicate fixture code
- Easy to extend with new test categories
- Clear inheritance hierarchy

## Test Categories

### 1. Core Tests (16)

**Purpose:** Validate basic functionality

**Examples:**

```python
def test_match_symptoms_basic(symptom_lexicon):
    """Test basic symptom matching."""
    text = "Patient has severe itching"
    spans = match_symptoms(text, symptom_lexicon)
    
    assert len(spans) == 1
    assert spans[0]["text"] in ["itching", "severe itching"]
    assert spans[0]["label"] == "SYMPTOM"

def test_match_products_basic(product_lexicon):
    """Test basic product matching."""
    text = "Used Lotion X twice daily"
    spans = match_products(text, product_lexicon)
    
    assert len(spans) == 1
    assert spans[0]["text"] == "Lotion X"
    assert spans[0]["label"] == "PRODUCT"

def test_negation_forward():
    """Test forward negation detection."""
    text = "No history of itching"
    spans = match_symptoms(text, ["itching"])
    
    assert len(spans) == 1
    assert spans[0].get("negated", False) is True
```

**Coverage:**
- Symptom matching
- Product matching
- Fuzzy matching
- Negation detection (forward/backward)
- Confidence scoring

### 2. Edge Case Tests (98)

**Purpose:** Validate boundary conditions and special cases

**Categories:**

#### Unicode & Emoji (12 tests)

```python
def test_emoji_within_text():
    """Test emoji doesn't break span detection."""
    text = "Patient has 😊 severe itching and 🌡️ redness"
    spans = match_symptoms(text, ["itching", "redness"])
    assert len(spans) == 2

def test_unicode_medical_symbols():
    """Test medical unicode symbols."""
    text = "Patient has ≥3 episodes of severe itching"
    spans = match_symptoms(text, ["itching"])
    assert len(spans) == 1
```

#### Negation Patterns (24 tests)

```python
def test_bidirectional_negation():
    """Test both forward and backward negation."""
    # Forward
    text1 = "No history of itching"
    spans1 = match_symptoms(text1, ["itching"])
    assert spans1[0].get("negated", False) is True
    
    # Backward
    text2 = "Itching was denied"
    spans2 = match_symptoms(text2, ["itching"])
    assert spans2[0].get("negated", False) is True

def test_negation_out_of_scope():
    """Test negation beyond window."""
    text = "Patient denies fever but reports itching"
    spans = match_symptoms(text, ["itching"], negation_window=5)
    itching_span = next(s for s in spans if s["text"] == "itching")
    assert itching_span.get("negated", False) is False
```

#### Boundary Cases (18 tests)

```python
def test_last_token_alignment():
    """Test last-token alignment filter."""
    text = "Patient has severe itching today"
    spans = match_symptoms(text, ["itch"])  # Partial match
    # Should NOT match "itch" (doesn't end at token boundary)
    assert not any(s["text"] == "itch" for s in spans)

def test_sentence_boundary():
    """Test span doesn't cross sentence."""
    text = "Patient has redness. New sentence with itching."
    spans = match_symptoms(text, ["redness", "itching"])
    assert len(spans) == 2
```

#### Anatomy Filter (15 tests)

```python
def test_anatomy_singleton_rejection():
    """Test single anatomy token rejected."""
    text = "Apply to skin twice daily"
    spans = match_symptoms(text, ["skin"])
    assert len(spans) == 0  # Generic anatomy alone rejected

def test_anatomy_with_symptom_keyword():
    """Test anatomy accepted with symptom."""
    text = "Patient has skin redness"
    spans = match_symptoms(text, ["skin"])
    assert len(spans) > 0  # Accepted due to "redness" co-occurrence
```

#### Validation & Errors (29 tests)

```python
def test_empty_lexicon():
    """Test empty lexicon handling."""
    text = "Patient has itching"
    spans = match_symptoms(text, [])
    assert spans == []

def test_empty_text():
    """Test empty text handling."""
    spans = match_symptoms("", ["itching"])
    assert spans == []

def test_confidence_bounds():
    """Test confidence clamped to [0, 1]."""
    text = "Patient has severe itching"
    spans = match_symptoms(text, ["severe itching"])
    for span in spans:
        assert 0.0 <= span["confidence"] <= 1.0
```

### 3. Integration Tests (26)

**Purpose:** Validate end-to-end workflows

**Categories:**

#### Pipeline Integration (11 tests)

```python
def test_pipeline_end_to_end():
    """Test full pipeline from text to entities."""
    from src.pipeline import simple_inference
    
    text = "Patient used Lotion X and experienced severe itching"
    results = simple_inference([text])
    
    assert len(results) == 1
    assert "text" in results[0]
    assert "entities" in results[0]
    assert len(results[0]["entities"]) >= 2  # SYMPTOM + PRODUCT

def test_pipeline_jsonl_export():
    """Test JSONL persistence."""
    import tempfile
    from src.pipeline import simple_inference
    
    texts = ["Text 1 with itching", "Text 2 with redness"]
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        output_path = f.name
    
    results = simple_inference(texts, output_jsonl=output_path)
    
    # Verify file created
    with open(output_path) as f:
        lines = f.readlines()
    assert len(lines) == len(texts)
```

#### Scale Tests (9 tests)

```python
def test_large_batch_processing():
    """Test batch processing 100+ texts."""
    from src.pipeline import simple_inference
    
    texts = [f"Patient {i} has itching" for i in range(100)]
    results = simple_inference(texts)
    
    assert len(results) == 100
    assert all("entities" in r for r in results)

def test_long_text_handling():
    """Test very long texts (>1000 chars)."""
    text = "Patient has itching. " * 50  # ~1000 chars
    spans = match_symptoms(text, ["itching"])
    assert len(spans) > 0
```

#### Performance Tests (6 tests)

```python
import time

def test_inference_speed():
    """Test inference time per text."""
    from src.pipeline import simple_inference
    
    texts = [f"Patient {i} has itching" for i in range(10)]
    
    start = time.time()
    results = simple_inference(texts)
    elapsed = time.time() - start
    
    # Should process <1 sec/text on CPU
    assert elapsed / len(texts) < 1.0
```

### 4. Curation Tests (4)

**Purpose:** Validate annotation pipeline

```python
def test_weak_label_export_format():
    """Test weak labels exportable to Label Studio."""
    from src.pipeline import simple_inference
    import json
    
    text = "Patient has itching"
    results = simple_inference([text])
    
    # Convert to Label Studio format
    task = {
        "data": {"text": results[0]["text"]},
        "predictions": [{
            "result": [
                {
                    "value": {
                        "start": e["start"],
                        "end": e["end"],
                        "text": e["text"],
                        "labels": [e["label"]]
                    },
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels"
                }
                for e in results[0]["entities"]
            ]
        }]
    }
    
    # Validate JSON serializable
    json_str = json.dumps(task)
    assert json_str
```

## Fixtures

### Shared Fixtures (conftest.py)

```python
import pytest

@pytest.fixture
def symptom_lexicon():
    """Standard symptom lexicon."""
    return [
        "itching", "redness", "burning", "swelling",
        "severe itching", "burning sensation", "dry skin"
    ]

@pytest.fixture
def product_lexicon():
    """Standard product lexicon."""
    return ["Lotion X", "Cream Y", "Soap Z"]

@pytest.fixture
def sample_texts():
    """Sample complaint texts."""
    return [
        "Patient has severe itching",
        "No redness reported",
        "Used Lotion X twice daily"
    ]
```

### Test-Specific Fixtures

```python
@pytest.fixture
def negation_config():
    """Config with extended negation window."""
    from src.config import AppConfig
    return AppConfig(negation_window=7)

@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
```

## Assertions & Validation

### Entity Assertions

```python
def assert_entity_valid(entity):
    """Validate entity structure."""
    assert "text" in entity
    assert "start" in entity
    assert "end" in entity
    assert "label" in entity
    assert entity["label"] in ["SYMPTOM", "PRODUCT"]
    assert 0 <= entity.get("confidence", 0.0) <= 1.0

def assert_span_bounds(text, entity):
    """Validate span boundaries."""
    assert 0 <= entity["start"] < len(text)
    assert entity["start"] < entity["end"] <= len(text)
    assert text[entity["start"]:entity["end"]] == entity["text"]
```

### Negation Assertions

```python
def assert_negated(text, entity_text, lexicon):
    """Assert entity is negated."""
    spans = match_symptoms(text, lexicon)
    entity = next(s for s in spans if s["text"] == entity_text)
    assert entity.get("negated", False) is True

def assert_not_negated(text, entity_text, lexicon):
    """Assert entity is NOT negated."""
    spans = match_symptoms(text, lexicon)
    entity = next(s for s in spans if s["text"] == entity_text)
    assert entity.get("negated", False) is False
```

## Coverage

### Current Coverage (Example)

```
Name                        Stmts   Miss  Cover
-----------------------------------------------
src/__init__.py                 0      0   100%
src/config.py                  45      2    96%
src/model.py                   52      3    94%
src/weak_label.py             187     12    94%
src/pipeline.py                78      5    94%
src/llm_agent.py               34     28    18%  # Stub implementation
-----------------------------------------------
TOTAL                         396     50    87%
```

### Generate Coverage Report

```bash
# HTML report
pytest --cov=src --cov-report=html

# Open in browser
start htmlcov/index.html  # Windows
open htmlcov/index.html   # macOS

# Terminal report
pytest --cov=src --cov-report=term-missing
```

## Continuous Integration

### GitHub Actions Workflow

SpanForge runs tests on **6 configurations** (2 OS × 3 Python versions):

**.github/workflows/test.yml:**

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ['3.9', '3.10', '3.11']
    
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest -v --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Pre-commit Hooks

**.pre-commit-config.yaml:**

```yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
```

**Install hooks:**

```bash
pip install pre-commit
pre-commit install
```

Now tests run automatically before each commit.

## Best Practices

1. **Write tests first** (TDD) - define expected behavior
2. **Use descriptive names** - `test_negation_forward_window_5` not `test1`
3. **One assertion per test** - easier to debug failures
4. **Use fixtures** - avoid duplicate setup code
5. **Test edge cases** - empty inputs, boundary values, unicode
6. **Mock external calls** - don't hit HuggingFace API in tests
7. **Run locally before push** - ensure CI will pass
8. **Track coverage** - aim for ≥90% on core modules

## Debugging Failed Tests

### Verbose Output

```bash
# Show print statements
pytest -v -s

# Show locals on failure
pytest -v -l

# Stop on first failure
pytest -x
```

### Debugging with pdb

```python
def test_example():
    text = "Patient has itching"
    spans = match_symptoms(text, ["itching"])
    
    # Drop into debugger
    import pdb; pdb.set_trace()
    
    assert len(spans) == 1
```

### Parametrized Tests

```python
import pytest

@pytest.mark.parametrize("text,expected", [
    ("No itching", True),
    ("Patient has itching", False),
    ("Itching was denied", True),
])
def test_negation_parametrized(text, expected):
    """Test negation with multiple cases."""
    spans = match_symptoms(text, ["itching"])
    assert spans[0].get("negated", False) == expected
```

## Performance Benchmarks

### Benchmark Suite

```python
import pytest
import time

@pytest.mark.benchmark
def test_match_symptoms_speed(benchmark):
    """Benchmark symptom matching."""
    text = "Patient has severe itching and redness"
    lexicon = ["itching", "redness", "burning"]
    
    result = benchmark(match_symptoms, text, lexicon)
    assert len(result) > 0

# Run benchmarks
pytest -v -m benchmark --benchmark-only
```

### Expected Performance

| Operation | Texts | Time (CPU) | Time (GPU) |
|-----------|-------|------------|------------|
| `match_symptoms` | 1 | ~10ms | N/A |
| `simple_inference` | 1 | ~200ms | ~50ms |
| `simple_inference` (batch=32) | 32 | ~5s | ~1s |

## Next Steps

- [Development: Contributing](contributing.md) - Contribution guidelines
- [User Guide: Weak Labeling](../user-guide/weak-labeling.md) - Understand tested features
- [API Reference](../api/weak_label.md) - Function signatures
