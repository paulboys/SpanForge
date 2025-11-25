# SpanForge Roadmap

Project roadmap and feature planning for biomedical NER pipeline.

## Project Status

**Current Phase:** Documentation & Infrastructure (Phase 4 Complete) ✅  
**Test Coverage:** 144/144 tests passing (100%)  
**CI/CD:** Active GitHub Actions workflows  
**Next Phase:** Annotation & Curation (Phase 5)

## Completed Phases

### Phase 1: Bootstrap & Lexicon ✅

**Completed:** November 2024

**Objectives:**
- Repository setup and project structure
- BioBERT model loading
- Initial lexicon-based weak labeling
- Core functionality implementation

**Deliverables:**
- ✅ `src/config.py` - Configuration management
- ✅ `src/model.py` - BioBERT loader
- ✅ `src/weak_label.py` - Basic fuzzy matching
- ✅ `src/pipeline.py` - End-to-end inference
- ✅ `data/lexicon/symptoms.csv` - Symptom lexicon (MedDRA-derived)
- ✅ `data/lexicon/products.csv` - Product lexicon
- ✅ `requirements.txt` - Dependencies

**Key Achievements:**
- Fuzzy matching with RapidFuzz (WRatio ≥88)
- Jaccard token-set filtering (≥40)
- Basic negation detection (forward only)
- JSONL persistence

---

### Phase 2: Weak Label Refinement ✅

**Completed:** December 2024

**Objectives:**
- Improve weak labeling accuracy
- Add advanced filters and heuristics
- Expand negation detection

**Deliverables:**
- ✅ Bidirectional negation (forward + backward windows)
- ✅ Last-token alignment filter
- ✅ Anatomy singleton filter
- ✅ Emoji and unicode handling
- ✅ Confidence scoring (0.8×fuzzy + 0.2×jaccard)
- ✅ Expanded negation token list (22 cues)

**Key Achievements:**
- Negation recall improved by ~30% (backward detection)
- Emoji handling prevents span breakage
- Anatomy filter reduces false positives by ~15%
- Confidence scores well-calibrated for active learning

---

### Phase 3: Test Infrastructure & Edge Cases ✅

**Completed:** January 2025 (Commit 8307485)

**Objectives:**
- Comprehensive test coverage
- Edge case validation
- Test composition patterns

**Deliverables:**
- ✅ 144 tests (16 core, 98 edge cases, 26 integration, 4 curation)
- ✅ Test base classes with composition
- ✅ Unicode and emoji edge case tests
- ✅ Negation pattern tests (24 tests)
- ✅ Boundary condition tests (18 tests)
- ✅ Anatomy filter tests (15 tests)
- ✅ Validation and error tests (29 tests)
- ✅ Scale and performance tests (15 tests)

**Key Achievements:**
- 100% test pass rate
- Edge cases documented and validated
- Test composition eliminates duplication
- Performance benchmarks established

---

### Phase 4: CI/CD Integration ✅

**Completed:** January 2025 (Commit dbc2ad8)

**Objectives:**
- Automated testing on push/PR
- Pre-commit hooks
- Configuration management
- Documentation infrastructure

**Deliverables:**
- ✅ GitHub Actions workflows (test.yml, pre-commit.yml)
- ✅ 6 CI configurations (2 OS × 3 Python versions)
- ✅ Pre-commit hooks (pytest, formatting)
- ✅ pyproject.toml configuration
- ✅ README updates with badges
- ✅ MkDocs infrastructure with Material theme
- ✅ Comprehensive docstrings and type hints

**Key Achievements:**
- CI/CD pipeline fully automated
- Pre-commit hooks enforce quality
- Test matrix covers Python 3.9-3.11 on Ubuntu/Windows
- Professional documentation site

---

## Current Phase

### Phase 5: Annotation & Curation 🚧

**Status:** Planned (In Progress: Documentation Complete)  
**Target:** Q1 2025

**Objectives:**
- Integrate Label Studio for human annotation
- Build annotation workflow and tooling
- Implement provenance tracking
- Quality assurance and agreement metrics

**Planned Deliverables:**

#### Label Studio Setup
- [ ] `data/annotation/config/label_config.xml` - Label config (SYMPTOM, PRODUCT)
- [ ] `scripts/annotation/init_label_studio_project.py` - Project bootstrap
- [ ] Telemetry disabled (privacy-safe setup)

#### Import/Export Pipeline
- [ ] `scripts/annotation/import_weak_to_labelstudio.py` - Weak label import
- [ ] `scripts/annotation/convert_labelstudio.py` - Export to gold JSONL
- [ ] Consensus/adjudication logic (majority vote, longest span)

#### Quality Assurance
- [ ] `scripts/annotation/quality_report.py` - Per-annotator stats, agreement
- [ ] `scripts/annotation/adjudicate.py` - Conflict resolution
- [ ] Inter-annotator agreement (IOU ≥0.5, Cohen's kappa)

#### Provenance & Registry
- [ ] `scripts/annotation/register_batch.py` - Track batches, annotators
- [ ] `data/annotation/registry.csv` - Batch metadata
- [ ] Conflict collection (`data/annotation/conflicts/`)

#### Documentation
- ✅ `docs/annotation_guide.md` - Annotation guidelines (COMPLETE)
- [ ] `docs/tutorial_labeling.md` - Step-by-step tutorial
- [ ] `scripts/AnnotationWalkthrough.ipynb` - Interactive tutorial
- [ ] Glossary of symptom synonyms

#### CLI Tooling
- [ ] `scripts/annotation/cli.py` - Unified CLI (`bootstrap`, `import-weak`, `export-convert`, `quality`, `adjudicate`, `register`)

**Success Criteria:**
- <5% conflicting overlaps after consensus
- ≥90% canonical coverage
- Annotator agreement (IOU ≥0.5) >0.75
- Clean gold JSONL passes integrity tests

**Risks & Mitigations:**
- Annotator overlap confusion → Visual examples & highlight guidance
- Bias from pre-annotated spans → Option to hide weak spans
- Inconsistent boundaries → Boundary rules + integrity tests
- Privacy concerns → Local-only data, telemetry disabled

---

## Upcoming Phases

### Phase 6: Gold Standard Assembly

**Status:** Planned  
**Target:** Q1-Q2 2025

**Objectives:**
- Curate high-quality gold annotations (≥500 samples)
- Define label schema (`labels.json`)
- Split train/dev/test sets
- Establish evaluation baselines

**Planned Work:**
- Annotate 500-1000 complaint texts
- Consensus annotation (2-3 annotators/task)
- Adjudication of conflicts
- Dataset splits (70/15/15 train/dev/test)
- Baseline weak labeling evaluation (P/R/F1)

**Deliverables:**
- `data/gold/train.jsonl` - Training set
- `data/gold/dev.jsonl` - Development set
- `data/gold/test.jsonl` - Test set (held out)
- `data/labels.json` - Label schema (SYMPTOM, PRODUCT, O)
- Evaluation metrics: precision, recall, F1 per label

---

### Phase 7: Token Classification Fine-Tune

**Status:** Planned  
**Target:** Q2 2025

**Objectives:**
- Add classification head to BioBERT
- Fine-tune on gold annotations
- Hyperparameter tuning
- Model evaluation and selection

**Planned Work:**
- `AutoModelForTokenClassification` wrapper
- Training script with AdamW, learning rate scheduling
- Hyperparameter search (LR, batch size, epochs)
- Evaluation on dev/test sets
- Model checkpointing and versioning

**Deliverables:**
- `src/trainer.py` - Training loop
- `models/biobert-ner-v1/` - Fine-tuned checkpoint
- `scripts/train.py` - Training CLI
- `scripts/evaluate.py` - Evaluation script
- Training logs and metrics

**Expected Metrics:**
- SYMPTOM: P ~85%, R ~80%, F1 ~82%
- PRODUCT: P ~90%, R ~85%, F1 ~87%
- Macro F1: ~84%

---

### Phase 8: Domain Adaptation

**Status:** Planned  
**Target:** Q2-Q3 2025

**Objectives:**
- Continued MLM pre-training on complaints corpus
- Adapt BioBERT to colloquial language
- Compare adapted vs. base BioBERT

**Planned Work:**
- De-identify complaint corpus (≥10k texts)
- Masked language modeling (MLM) on complaints
- 1-3 epochs, batch size 16-32
- Evaluation: perplexity, downstream NER F1

**Deliverables:**
- `scripts/pretrain_mlm.py` - MLM training script
- `models/biobert-complaints-adapted/` - Adapted checkpoint
- Perplexity comparison report
- NER performance before/after adaptation

**Expected Gains:**
- Perplexity reduction: ~10-15%
- NER F1 improvement: ~2-4%
- Better handling of misspellings, colloquialisms

---

### Phase 9: Baseline Comparison

**Status:** Planned  
**Target:** Q3 2025

**Objectives:**
- Train RoBERTa-base for comparison
- Benchmark against BioBERT
- Analyze trade-offs (biomedical vs. general domain)

**Planned Work:**
- Fine-tune `roberta-base` on same gold data
- Compare BioBERT vs. RoBERTa on dev/test
- Error analysis (medical terms, colloquialisms)

**Deliverables:**
- `models/roberta-ner-v1/` - RoBERTa checkpoint
- Comparison report (P/R/F1, inference speed)
- Error analysis notebook

**Expected Results:**
- BioBERT: Better on medical terms
- RoBERTa: Better on colloquial language
- Final choice: Ensemble or BioBERT-adapted

---

### Phase 10: Evaluation & Calibration

**Status:** Planned  
**Target:** Q3 2025

**Objectives:**
- Comprehensive error analysis
- Confidence calibration
- Threshold tuning for production

**Planned Work:**
- Error categorization (false positives, false negatives)
- Confidence calibration curves
- Threshold optimization (maximize F1 or balance P/R)
- Per-label performance analysis

**Deliverables:**
- `docs/error_analysis.md` - Error taxonomy
- Calibration plots
- Optimal threshold recommendations
- Per-label performance breakdown

---

### Phase 11: Educational Docs Expansion

**Status:** Partially Complete  
**Target:** Q4 2025

**Objectives:**
- Comprehensive user and developer docs
- API reference
- Tutorials and walkthroughs

**Planned Work:**
- ✅ `docs/overview.md` - Architecture overview (COMPLETE)
- ✅ `docs/annotation_guide.md` - Annotation guidelines (COMPLETE)
- ✅ `docs/user-guide/weak-labeling.md` - Weak labeling guide (COMPLETE)
- ✅ `docs/user-guide/negation.md` - Negation guide (COMPLETE)
- ✅ `docs/user-guide/pipeline.md` - Pipeline guide (COMPLETE)
- ✅ `docs/development/testing.md` - Testing guide (COMPLETE)
- [ ] `docs/heuristic.md` - Heuristic tuning guide
- [ ] `docs/model_strategy.md` - Model selection guide
- [ ] `docs/deployment.md` - Production deployment

---

### Phase 12: Continuous Improvement & Active Learning

**Status:** Planned  
**Target:** Ongoing (Q4 2025+)

**Objectives:**
- Active learning pipeline
- Model monitoring and retraining
- Feedback loop for continuous improvement

**Planned Work:**
- Active learning: prioritize uncertain samples
- Human-in-the-loop annotation for edge cases
- Periodic model retraining (monthly/quarterly)
- Performance monitoring dashboard

**Deliverables:**
- `scripts/active_learning.py` - Uncertainty sampling
- Monitoring dashboard (Streamlit or Grafana)
- Retraining automation scripts
- Performance drift detection

---

## Feature Wishlist

### Near-Term (2025)

- [ ] Multi-language support (Spanish, French)
- [ ] Relation extraction (symptom-product links)
- [ ] Severity classification (mild/moderate/severe)
- [ ] Temporal extraction (onset, duration)

### Long-Term (2026+)

- [ ] Real-time inference API (FastAPI + Docker)
- [ ] Web-based annotation interface (custom UI)
- [ ] Integration with FAERS database
- [ ] Ensemble models (BioBERT + RoBERTa + ClinicalBERT)
- [ ] Zero-shot entity recognition (GPT-4 integration)

---

## Contribution Opportunities

Looking for contributors in:

1. **Annotation** - Help curate gold standard dataset
2. **Documentation** - Expand tutorials and examples
3. **Testing** - Add edge cases and integration tests
4. **Feature Development** - Implement roadmap items
5. **Research** - Experiment with new models/techniques

See [Contributing Guide](../development/contributing.md) for details.

---

## Version History

| Version | Date | Phase | Highlights |
|---------|------|-------|------------|
| **v0.1.0** | Nov 2024 | Phase 1 | Initial release, weak labeling |
| **v0.2.0** | Dec 2024 | Phase 2 | Bidirectional negation, filters |
| **v0.3.0** | Jan 2025 | Phase 3 | 144 tests, 100% pass rate |
| **v0.4.0** | Jan 2025 | Phase 4 | CI/CD, MkDocs, docstrings |
| **v0.5.0** | Q1 2025 | Phase 5 | Label Studio, annotation (planned) |
| **v1.0.0** | Q2 2025 | Phase 7 | Fine-tuned NER model (planned) |

---

## Contact & Feedback

**Questions or suggestions?**  
Open an issue on GitHub: [SpanForge Issues](#)

**Want to contribute?**  
See [Contributing Guide](../development/contributing.md)

---

*Last updated: January 2025*
