# Pre-Phase 6 Action Plan

**Created**: November 27, 2025  
**Status**: Pending Approval  
**Estimated Time**: 7-10 hours

---

## Quick Summary

Three issues identified before Phase 6:

1. ❌ **Test Coverage**: 76% overall, new `weak_labeling` package only 49%
2. ⚠️ **Documentation**: 36 files with ~40% duplication in phase summaries
3. ⚠️ **Versioning**: Submodule has independent version (0.2.0 vs 0.5.0)

---

## Option A: Full Cleanup (Recommended)

### Duration: 7-10 hours

### Tasks

#### 1. Test Coverage (4-6 hours) - CRITICAL
```powershell
# Create new test suite
mkdir tests\weak_labeling
New-Item tests\weak_labeling\test_confidence.py
New-Item tests\weak_labeling\test_matchers.py
New-Item tests\weak_labeling\test_validators.py
New-Item tests\weak_labeling\test_negation.py
New-Item tests\weak_labeling\test_labeler.py

# Target coverage:
# - confidence.py:  16% → 90% (+74%)
# - matchers.py:    38% → 90% (+52%)
# - validators.py:  46% → 90% (+44%)
# - negation.py:    68% → 90% (+22%)
# - labeler.py:     63% → 80% (+17%)

# Overall impact: 76% → 95% (+19%)
```

#### 2. Documentation (2-3 hours) - HIGH
```powershell
# Consolidate phase docs
mv docs\phase*.md docs\archive\
mv docs\development\phase*.md docs\archive\

# Create unified docs
# - docs/about/changelog.md (version history)
# - docs/user-guide/annotation.md (merge 3 files)
# - docs/user-guide/llm-integration.md (merge 3 files)

# Result: 36 files → 25 files, -100KB
```

#### 3. Versioning (1 hour) - MEDIUM
```python
# Remove submodule version
# src/weak_labeling/__init__.py
# DELETE: __version__ = "0.2.0"

# Create changelog
# docs/about/changelog.md

# Update to 0.6.0 for Phase 6
# - src/__init__.py
# - pyproject.toml
# - VERSION
```

### Benefits
- ✅ 95% test coverage (production-ready)
- ✅ Clean, maintainable documentation
- ✅ Consistent semantic versioning
- ✅ Confidence for Phase 6 model training

---

## Option B: Critical Only (Fast Track)

### Duration: 4-6 hours

### Tasks

#### 1. Test Coverage Only (4-6 hours) - CRITICAL
Create test suite for `weak_labeling` package as in Option A.

### Skip
- Documentation consolidation (defer to Phase 6)
- Versioning cleanup (defer to Phase 6)

### Benefits
- ✅ Validate refactored code before model training
- ⏭️ Faster to Phase 6 (model training)

### Risks
- ⚠️ Documentation remains fragmented
- ⚠️ Version inconsistency persists

---

## Option C: Proceed As-Is (Not Recommended)

### Duration: 0 hours

### Risks
- ❌ **Critical**: Training model with untested code (49% coverage)
- ❌ **High**: Bug in matchers/validators could corrupt training data
- ⚠️ **Medium**: Confusing documentation for contributors

### Only If
- Phase 6 is exploratory only (no production deployment)
- Manual validation of all weak labels before training
- Small-scale training run (<1000 samples)

---

## Recommendation

**Choose Option A (Full Cleanup)**

### Rationale
1. **Model training is expensive**: CPU/GPU time, annotation effort
2. **Bad training data is worse than no training**: Untested code could introduce systematic errors
3. **7-10 hours is small investment**: Compared to weeks of Phase 6 work
4. **Sets precedent**: High quality standards for production

### Execution Order
```
Day 1 (4-6 hours):
├── Morning:   Create test_confidence.py, test_matchers.py
├── Afternoon: Create test_validators.py, test_negation.py
└── Evening:   Create test_labeler.py, run full suite

Day 2 (3-4 hours):
├── Morning:   Consolidate phase docs, create changelog
├── Afternoon: Fix versioning, update pyproject.toml
└── Validate:  Run full test suite, build docs, commit
```

---

## Approval Required

Please confirm which option to proceed with:

- [ ] **Option A: Full Cleanup** (7-10 hours, recommended)
- [ ] **Option B: Critical Only** (4-6 hours, fast track)
- [ ] **Option C: Proceed As-Is** (0 hours, not recommended)

---

## Next Steps After Approval

### If Option A or B Selected

1. **Create issue/branch**: `pre-phase6-cleanup`
2. **Run evaluation report**: `python coverage_summary.py`
3. **Execute tasks**: Per action plan
4. **Validate**: 
   ```powershell
   pytest tests/ --cov=src --cov-report=term
   mkdocs build --clean
   ```
5. **Commit and tag**: `git tag v0.6.0-dev`
6. **Proceed to Phase 6**: Gold standard assembly

### If Option C Selected

**Document risks in Phase 6 plan**:
- Small batch size (<500 samples)
- Manual validation of all training data
- Extensive error analysis post-training
- Budget for potential retraining if bugs found

---

## Questions?

See full evaluation: `docs/development/EVALUATION_PRE_PHASE6.md`
