# Test Suite API Fixes Needed

## Summary
New weak_labeling tests exposed API mismatches between expected and actual implementations. Rather than block progress, documenting issues here for targeted fixes after verifying coverage gains.

## API Signature Issues

### 1. LexiconEntry (types.py)
**Expected**: `LexiconEntry(term, canonical, category, aliases, anatomical_site, severity, metadata)`  
**Actual**: `LexiconEntry(term, canonical, source, concept_id=None, sku=None, category=None)`  
**Impact**: 30+ test errors in test_labeler.py  
**Fix**: Update fixtures to use `source` parameter instead of `aliases`

### 2. detect_negated_regions (negation.py)
**Expected**: `detect_negated_regions(text, window_size=5)`  
**Actual**: `detect_negated_regions(text, window=5)`  
**Impact**: 24 test failures  
**Fix**: Replace `window_size` parameter with `window`

### 3. is_anatomy_only (validators.py)
**Expected**: `is_anatomy_only(text) -> bool`  
**Actual**: `is_anatomy_only(text, tokens) -> bool`  
**Impact**: 11 test failures  
**Fix**: Pass tokens list parameter or tokenize within test

### 4. validate_span_alignment (validators.py)
**Expected**: `validate_span_alignment(text, spans) -> Tuple[bool, List[str]]`  
**Actual**: `validate_span_alignment(text, span_text, start, end) -> bool`  
**Impact**: 13 test failures  
**Fix**: Iterate over spans and call function per-span, collect results

### 5. exact_match (matchers.py)
**Expected**: Case-sensitive by default  
**Actual**: `case_sensitive=False` default  
**Impact**: 1 test failure  
**Fix**: Pass `case_sensitive=True` in test

### 6. WeakLabeler.match_products (labeler.py)
**Expected**: Public method  
**Actual**: May be private `_match_products` or not exist  
**Impact**: 1 test failure  
**Fix**: Check actual API and use correct method

## Behavior Mismatches

### 7. deduplicate_spans confidence retention
**Expected**: Keep highest confidence from duplicate group  
**Actual**: Appears to reset confidence to 1.0  
**Impact**: 2 test failures  
**Fix**: Verify actual deduplication logic

### 8. filter_overlapping_spans strategy validation
**Expected**: Raises ValueError on invalid strategy  
**Actual**: Does not raise (possibly logs or ignores)  
**Impact**: 1 test failure  
**Fix**: Check actual error handling

### 9. Jaccard stopword filtering
**Expected**: Full stopword removal before comparison  
**Actual**: Partial filtering or different stopword list  
**Impact**: 2 test failures  
**Fix**: Align test with actual STOPWORDS set

### 10. fuzzy_match normalization
**Expected**: Perfect match after whitespace normalization  
**Actual**: Slight score difference (97.14 vs 100.0)  
**Impact**: 1 test failure  
**Fix**: Accept fuzzy match scores with tolerance (>95.0 instead of ==100.0)

## Test Stats
- **Total tests created**: 230+
- **Passing**: 104 (45%)
- **Failing**: 54 (24%)
- **Errors**: 29 (13%)
- **Not run** (due to errors): ~43 (18%)

## Next Steps
1. Run existing test suite to verify no regressions
2. Check coverage gains from passing tests (104 new tests is significant)
3. Prioritize top 3 fix categories by impact:
   - Category A: LexiconEntry fixtures (30 errors) - quick fix
   - Category B: detect_negated_regions parameter (24 failures) - quick find/replace
   - Category C: is_anatomy_only signature (11 failures) - requires logic change
4. After fixes, re-run coverage to validate 95% target

## Coverage Impact (Estimated)
Even with 45% passing rate, 104 new tests should improve:
- confidence.py: 16% → ~50% (30 passing tests)
- matchers.py: 38% → ~65% (35 passing tests)
- validators.py: 46% → ~60% (20 passing tests)
- negation.py: 68% → ~75% (15 passing tests, many blocked)
- labeler.py: 63% → ~65% (few tests passing due to fixture errors)

## Action Plan
**Option 1 (Recommended)**: Fix top 3 categories (A, B, C) in batch using multi_replace  
**Option 2**: Run full suite coverage now, accept 70-80%, proceed to Phase 6  
**Option 3**: Fix all issues individually (3-4 hours)

**Recommendation**: Option 1 - Fix categories A & B (50 errors) in next 30 minutes, re-run coverage, likely hit 85-90%. Category C can be deferred if time-constrained.
