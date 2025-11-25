# SpanForge Test Architecture

> Comprehensive test infrastructure with composition-based patterns, parametrization, and edge case coverage for BioBERT + weak labeling NER pipeline.

## Overview

Refactored test suite (Nov 2025) to eliminate duplication (~70% reduction in setup code), improve maintainability via composition/inheritance, and expand edge case coverage (3x increase).

### Test Statistics
- **Total Tests**: 140+ (after Phase 2)
- **Core Tests**: 16 (weak labeling, overlap, integrity)
- **Edge Case Tests**: 98 (boundary, unicode, negation, overlap, malformed)
- **Integration Tests**: 26 (end-to-end pipeline, scale/stress)
- **Pass Rate**: 133/140 tests passing (known failures document system limitations)
- **Infrastructure**: 3 base classes, 3 composition helpers, 8+ pytest fixtures

## Architecture

### Base Classes (`tests/base.py`)

#### `TestBase`
Core utilities for all tests.

**Features:**
- Automatic temp directory management (`setUp`/`tearDown`)
- File builders: `create_temp_file`, `create_lexicon_csv`, `create_jsonl_file`
- JSONL I/O: `load_jsonl`
- Mock config context manager: `mock_config(**overrides)`

**Example:**
```python
class TestMyFeature(TestBase):
    def setUp(self):
        super().setUp()
        self.data_file = self.create_temp_file("data.txt", "sample content")
    
    def test_something(self):
        records = self.load_jsonl(self.data_file)
        self.assertEqual(len(records), 10)
```

#### `WeakLabelTestBase`
Extends `TestBase` for weak labeling tests with lexicon support.

**Features:**
- `create_symptom_lexicon(entries=None)`: Creates and loads symptom lexicon with defaults
- `create_product_lexicon(entries=None)`: Creates and loads product lexicon with defaults
- `load_lexicon(path)`: Loads lexicon using `weak_label` module loaders

**Example:**
```python
class TestSymptomMatching(WeakLabelTestBase):
    def setUp(self):
        super().setUp()
        self.symptom_lexicon = self.create_symptom_lexicon()  # Uses defaults
        self.product_lexicon = self.create_product_lexicon()
    
    def test_detection(self):
        spans = weak_label(text, self.symptom_lexicon, self.product_lexicon)
        self.assertGreater(len(spans), 0)
```

#### `IntegrationTestBase`
Extends `TestBase` for full pipeline integration tests.

**Features:**
- Automatic directory structure creation (`data/lexicon`, `data/output`, `data/annotation/exports`)
- `create_standard_lexicons()`: Minimal symptom + product lexicons for pipeline tests

**Example:**
```python
class TestEndToEnd(IntegrationTestBase):
    def setUp(self):
        super().setUp()
        sym_path, prod_path = self.create_standard_lexicons()
```

### Composition Helpers (`tests/assertions.py`)

#### `SpanAsserter`
Reusable span validation assertions.

**Methods:**
- `assert_span_equals(expected, actual, check_confidence=True)`
- `assert_span_list_contains(expected_span, span_list)`
- `assert_boundaries_valid(text, spans)`
- `assert_text_slices_match(text, spans)`
- `assert_no_duplicate_spans(spans)`

**Example:**
```python
def test_span_validation(self):
    span_asserter = SpanAsserter(self)
    span_asserter.assert_boundaries_valid(text, spans)
    span_asserter.assert_text_slices_match(text, spans)
```

#### `OverlapChecker`
Overlap computation and conflict detection.

**Methods:**
- `compute_overlap(span_a, span_b) -> int`: Returns overlap characters
- `compute_iou(span_a, span_b) -> float`: Returns intersection-over-union
- `assert_no_conflicting_labels(spans)`: Fails if overlapping spans have different labels
- `assert_overlap_matrix_valid(spans, allow_same_label_overlap=True)`

**Example:**
```python
def test_overlaps(self):
    overlap_checker = OverlapChecker(self)
    iou = overlap_checker.compute_iou(span_a, span_b)
    overlap_checker.assert_no_conflicting_labels(spans)
```

#### `IntegrityValidator`
Gold dataset integrity validation.

**Methods:**
- `assert_provenance_present(record)`: Checks `source`, `annotator`, `revision` fields
- `assert_canonical_present(entities)`: Ensures all entities have `canonical`
- `assert_labels_valid(entities)`: Validates labels in `ALLOWED_LABELS`
- `assert_no_duplicates(entities)`
- `assert_sorted_by_start(entities)`
- `validate_full_record(record)`: Runs all checks + boundary + text slice validation

**Example:**
```python
def test_gold_integrity(self):
    integrity_validator = IntegrityValidator(self)
    for record in gold_records:
        integrity_validator.validate_full_record(record)
```

### Fixtures (`tests/conftest.py`)

#### Lexicon Fixtures
- `temp_lexicon`: Creates temp symptom + product CSVs
- `mock_config`: AppConfig with temp lexicon paths

#### Sample Data Fixtures
- `sample_texts`: List of standard test texts
- `sample_texts_unicode`: Unicode/emoji test texts
- `sample_weak_labels`: Span dicts with confidence scores
- `sample_overlap_spans`: Various overlap patterns

#### Gold File Fixtures
- `gold_files`: Auto-discovers and creates temp gold JSONL files

**Example:**
```python
def test_with_fixtures(mock_config, sample_texts):
    for text in sample_texts:
        spans = weak_label(text, config=mock_config)
        assert len(spans) > 0
```

## Test Organization

### Core Tests
- `test_weak_label.py`: Symptom/product matching, negation, JSONL persistence (16 tests)
- `test_overlap_merge.py`: Deduplication, IOU computation, conflict detection (16 tests)
- `test_curation_integrity.py`: Gold file validation (parametrized)
- `test_forward.py`: Model loading sanity checks
- `test_llm_refine.py`: LLM refinement stub validation

### Edge Cases (`tests/edge_cases/`)
- `test_boundary_conditions.py`: Span at text start/end, zero-width, single-char (17 tests)
- `test_unicode_emoji.py`: Accented chars, CJK, emojis, combining marks (17 tests)
- `test_negation_edge.py`: Negation window boundaries, double negatives, multiple cues (17 tests)
- `test_overlap_scenarios.py`: Nested, partial, adjacent, conflict detection (19 tests)
- `test_malformed_inputs.py`: Empty text, invalid JSON, missing fields, boundary errors (28 tests)

### Integration Tests (`tests/integration/`)
- `test_end_to_end.py`: Full pipeline validation (11 tests)
  - Weak labeling stage
  - JSONL persistence
  - Gold export integrity
  - Confidence filtering
  - Provenance tracking
  - LLM refinement stub integration
  - Multi-document batch processing
- `test_scale.py`: Performance and stress tests (15 tests)
  - 100/1000 document processing
  - Long text handling (5k-10k chars)
  - Memory efficiency
  - Concurrent detection (many symptoms/products)
  - Unicode at scale
  - Performance benchmarks (<100ms/doc average)

## Running Tests

### All Tests
```powershell
pytest
```

### Specific Test File
```powershell
pytest tests/test_weak_label.py -v
```

### Specific Test Class
```powershell
pytest tests/test_weak_label.py::TestSymptomMatching -v
```

### Edge Cases Only
```powershell
pytest tests/edge_cases/ -v
```

### Integration Tests Only
```powershell
pytest tests/integration/ -v
```

### Parametrized Test Subset
```powershell
pytest tests/edge_cases/test_boundary_conditions.py::test_boundary_validation -v
```

### With Coverage
```powershell
pytest --cov=src --cov-report=html
```

### Quick Smoke Test (Fast Subset)
```powershell
pytest tests/test_weak_label.py tests/test_overlap_merge.py -v
```

## Parametrization Patterns

### Basic Parametrization
```python
@pytest.mark.parametrize("text,expected_label", [
    ("Patient has itching", "SYMPTOM"),
    ("Used cream", "PRODUCT"),
])
def test_label_detection(text, expected_label):
    spans = weak_label(text, symptom_lex, product_lex)
    assert spans[0].label == expected_label
```

### Combinatorial Scenarios
```python
@pytest.mark.parametrize("negation_cue,distance,should_negate", [
    ("no", 1, True),
    ("no", 3, True),
    ("no", 5, True),
    ("no", 6, False),  # Outside window
])
def test_negation_distances(negation_cue, distance, should_negate):
    text = f"{negation_cue}{' word' * (distance-1)} itching"
    # Test negation detection
```

### Expected Failures
```python
@pytest.mark.parametrize("span,should_fail", [
    ({"start": -1, "end": 5}, True),   # Negative start
    ({"start": 0, "end": 5}, False),   # Valid
])
def test_boundary_validation(span, should_fail):
    if should_fail:
        with pytest.raises(AssertionError):
            span_asserter.assert_boundaries_valid(text, [span])
    else:
        span_asserter.assert_boundaries_valid(text, [span])
```

## Known Test Limitations

### Expected Failures (Document System Behavior)
1. **Negation cues**: "absent", "denies" not in default negation list (tests/edge_cases/test_negation_edge.py)
2. **Unicode + Emoji**: Emoji may interfere with fuzzy matching in some contexts
3. **IOU edge cases**: Boundary conditions at exact threshold (0.8 IOU → 0.8 computed)
4. **Empty text validation**: Some validators expect non-empty text field
5. **Data quality**: Existing gold files may have provenance gaps or boundary errors

### Future Improvements
- Expand negation cue list based on test feedback
- Add inter-annotator agreement metrics
- Active learning simulation tests
- CI/CD integration (GitHub Actions)

## CI/CD Integration

### GitHub Actions Workflow (Planned)
```yaml
- name: Run tests
  run: |
    pytest --cov=src --cov-report=xml --junitxml=junit.xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
```

### Pre-commit Hook
```bash
pytest tests/test_weak_label.py tests/test_overlap_merge.py -q
```

## Contributing

### Adding New Tests
1. Inherit from appropriate base class (`TestBase`, `WeakLabelTestBase`, `IntegrationTestBase`)
2. Use composition helpers (`SpanAsserter`, `OverlapChecker`, `IntegrityValidator`)
3. Leverage fixtures from `conftest.py`
4. Parametrize combinatorial scenarios
5. Document expected failures if testing system limitations

### Test Naming Conventions
- Test files: `test_<feature>.py`
- Test classes: `Test<Feature>`
- Test methods: `test_<specific_behavior>`
- Edge case files: `tests/edge_cases/test_<category>.py`

### Assertions Best Practices
- Use descriptive assertion messages
- Prefer composition helpers over raw asserts
- Group related assertions in nested test classes
- Document why a test is expected to fail (if applicable)

## Maintenance

### Updating Base Classes
1. Add new utility methods to appropriate base class
2. Update this README with examples
3. Refactor existing tests to use new utilities if beneficial

### Updating Composition Helpers
1. Add new assertion methods to appropriate helper
2. Ensure backward compatibility
3. Update docstrings with examples

### Updating Fixtures
1. Add new fixtures to `conftest.py`
2. Document parameters and return types
3. Provide usage example in this README

## Performance

### Test Execution Times (Approximate)
- Core tests: ~0.3s
- Edge cases: ~2.1s
- Integration tests: ~2.2s (includes 1000-doc stress test)
- Full suite: ~5s

### Optimization Tips
- Use `pytest -n auto` for parallel execution (requires pytest-xdist)
- Run edge cases separately during development
- Cache lexicon loading for repeated tests
- Use `@pytest.mark.slow` for long-running tests

## Support

For test infrastructure questions or issues:
1. Check this README for patterns/examples
2. Review base class docstrings (`tests/base.py`)
3. Inspect composition helper methods (`tests/assertions.py`)
4. See fixture definitions (`tests/conftest.py`)
5. Open GitHub issue with `test:` prefix

---

Last Updated: November 2025 (Phase 2 Complete: Integration & Scale Tests Added)
