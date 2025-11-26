# Copilot / AI Agent Instructions (Python + BioBERT NER)

> Scope: Active biomedical NER + weak labeling prototype using BioBERT, lexicon-driven span extraction, JSONL persistence, and upcoming annotation + education workflows.

## Project Overview
Implements configuration, BioBERT loading, lexicon-based weak labeling, exploratory notebook, and tests. Moving toward supervised token classification and human curation.

## Current State
- Model loader: `src/model.py` (BioBERT `dmis-lab/biobert-base-cased-v1.1`)
- Config: `src/config.py` (fields: `model_name`, `max_seq_len`, `device`, `seed`, `negation_window`, `fuzzy_scorer`)
- Weak labeling: `src/weak_label.py` (symptom/product lexicons, fuzzy 0.88, Jaccard ≥40, bidirectional negation window 5, emoji handling, anatomy skip, last‑token alignment)
- Pipeline: `src/pipeline.py` (simple inference + optional JSONL persistence)
- LLM Refinement: `src/llm_agent.py` (multi-provider support: OpenAI, Azure OpenAI, Anthropic; boundary correction, negation validation, canonical normalization; 15/15 tests passing)
- Evaluation Harness: `src/evaluation/metrics.py` (10 evaluation functions: IOU, boundary precision, correction rate, calibration, stratification, P/R/F1; 27/27 tests passing)
- Evaluation Script: `scripts/annotation/evaluate_llm_refinement.py` (CLI tool for 3-way weak→LLM→gold comparison with JSON/Markdown reports)
- Visualization: `scripts/annotation/plot_llm_metrics.py` (matplotlib/seaborn plots: IOU uplift, calibration curves, correction breakdown, P/R/F1 comparison, stratified analysis)
- Lexicon build: `scripts/build_meddra_symptom_lexicon.py`
- Notebook: `scripts/Workbook.ipynb` (exploration & education)
- Environment check: `scripts/verify_env.py`
- Lexicons: `data/lexicon/symptoms.csv` (includes burning/burning sensation), `data/lexicon/products.csv`
- Output artifact: `data/output/notebook_test.jsonl`
- Test Suite: 171 tests (16 core, 98 edge cases, 26 integration, 4 curation, 27 evaluation) - 100% passing
- CI/CD: GitHub Actions workflows (test.yml, pre-commit.yml), pre-commit hooks, pyproject.toml config
- Dependencies: `requirements.txt`, `requirements-llm.txt` (openai, anthropic, tenacity), `requirements-viz.txt` (matplotlib, seaborn, numpy - optional)

## Roadmap Phases
1. Bootstrap & Lexicon (DONE)
2. Weak Label Refinement (DONE - bidirectional negation, emoji handling)
3. Test Infrastructure & Edge Cases (DONE - 171/171 tests passing)
4. CI/CD Integration (DONE - GitHub Actions, pre-commit hooks)
4.5. LLM-Based Refinement (DONE - multi-provider agent with evaluation harness)
5. Annotation & Curation (Label Studio integration) (UPCOMING)
6. Gold Standard Assembly
7. Token Classification Fine‑Tune (BioBERT + classification head)
8. Domain Adaptation (MLM on complaints corpus)
9. Baseline Comparison (RoBERTa)
10. Evaluation & Calibration (precision/recall/F1, confidence thresholds) (PARTIAL - evaluation metrics complete)
11. Educational Docs Expansion
12. Continuous Improvement & Active Learning

## Annotation & Curation (Phase 5 - IN PROGRESS)
- Tool: Label Studio (local, telemetry disabled)
- Workflow: weak labels → LLM refinement → Label Studio → gold JSONL → evaluation
- Privacy: strip PII before ingestion; keep raw complaints outside repo (`data/raw/` ignored)
- Config: `data/annotation/config/label_config.xml` (SYMPTOM/PRODUCT with hotkeys s/p, word granularity, colorblind-safe palette)
- Tutorial: `scripts/AnnotationWalkthrough.ipynb` (7 sections: intro, data prep, LLM demo, Label Studio setup, 5 practice examples, export/eval, common mistakes + glossary)
- Production Guide: `docs/production_workflow.md` (step-by-step batch evaluation workflow)
- Scripts Completed: `evaluate_llm_refinement.py`, `plot_llm_metrics.py`, `cli.py` (evaluate-llm, plot-metrics)
- Scripts Pending: `prepare_production_batch.py` (stratified sampling, de-identification)

## Label Studio Implementation & Tutorial Plan (Expanded)
### Objectives
- Local, privacy-safe annotation (telemetry disabled).
- Simple import of weak labels, human correction, consensus/adjudication, export to gold JSONL.
- Clear lay-user tutorial (non-technical annotators) with boundary, negation, ambiguity guidance.
- Provenance tracking (source, annotator(s), revision, consensus strategy).

### Components
1. Label Config: `data/annotation/config/label_config.xml` defining SYMPTOM & PRODUCT spans.
2. Project Bootstrap Script: `scripts/annotation/init_label_studio_project.py` (sets `LABEL_STUDIO_DISABLE_TELEMETRY=1`, creates project, uploads initial tasks).
3. Weak Import Script: `scripts/annotation/import_weak_to_labelstudio.py` (reads weak JSONL, posts tasks via API; optional confidence filter).
4. Conversion + Adjudication: Extend `convert_labelstudio.py` with majority vote consensus (`--consensus min_agree`), conflict collection when overlapping spans carry different labels.
5. Quality / Agreement: `scripts/annotation/quality_report.py` (per-annotator counts, disagreement rate, simple Cohen's kappa using IOU ≥0.5 span overlap).
6. Adjudication Tool: `scripts/annotation/adjudicate.py` (resolves conflicts; longest span or manual escalation; writes conflict JSON under `data/annotation/conflicts/`).
7. Registry & Provenance: `data/annotation/registry.csv` appended by `scripts/annotation/register_batch.py` after each conversion (batch_id, n_tasks, annotators, revision, consensus_done).
8. Tutorial Notebook: `scripts/AnnotationWalkthrough.ipynb` (intro, example text, weak vs gold comparison, practice exercise cell).
9. CLI Orchestration: `scripts/annotation/cli.py` subcommands (`bootstrap`, `import-weak`, `export-convert`, `quality`, `adjudicate`, `register`).

### Documentation Additions
- `docs/annotation_guide.md`: Expanded definitions, boundary rules, negation policy (annotate even if negated; mark flag), examples table.
- `docs/tutorial_labeling.md`: Step-by-step (launch, open project, select task, highlight, assign label, submit, export). Includes FAQ & keyboard shortcuts.
- Glossary: Common symptom synonyms (redness/erythema, pruritus/itching) to unify canonical usage.
- Rationale Section: Explains downstream model improvement & avoidance of noisy weak labels.

### Annotator Rules (Summary)
- Include full clinical phrase; exclude trailing punctuation.
- Annotate negated symptoms (mark NEGATED flag) to support separate modeling.
- Avoid single generic anatomy tokens unless part of explicit symptom phrase.
- Prefer specific terms ("dryness" over "dry" unless only "dry" present).
- Overlapping suggestions: choose most semantically complete span; if uncertain keep both for adjudication.

### Consensus Strategy
- Majority vote on identical (start,end,label).
- Tie-breaker: longest span; if label conflict on overlap → escalate to adjudication.
- Conflicts serialized to `data/annotation/conflicts/` for manual resolution.

### Quality Metrics
- Per-annotator span density (spans/task).
- Disagreement rate (#conflicts / total spans).
- Pairwise agreement (IOU ≥0.5) for top annotator pairs.
- Drift detection: sudden spike in PRODUCT vs SYMPTOM ratio flagged in report.

### Implementation Sequence
1. Add label config + bootstrap script.
2. Import weak script & initial tutorial docs.
3. Conversion consensus extension + registry.
4. Quality report & adjudication tools.
5. Notebook walkthrough & CLI wrapper.
6. Iterate with real annotations; refine thresholds post first 100 gold tasks.

### Risks & Mitigations
- Annotator Overlap Confusion → Provide visual examples & highlight guidance.
- Bias from pre-annotated weak spans → Option to hide weak spans on initial pass.
- Inconsistent boundaries → Boundary rules + integrity tests (canonical presence, text slice alignment).
- Privacy concerns → Local-only data; no raw complaints committed; telemetry off.

### Success Criteria
- <5% conflicting overlaps after consensus phase.
- ≥90% canonical coverage (symptom spans map to canonical lexicon entry or deterministic fallback).
- Annotator agreement (IOU ≥0.5) >0.75 after calibration round.
- Clean gold JSONL passes integrity & quality reports with zero blocking conflicts.

### Next Actions (Agent)
- Scaffold label config & bootstrap script.
- Add import weak script & consensus extension.
- Draft `docs/tutorial_labeling.md` with examples & glossary.

## Educational Docs Plan (To Create)
- `docs/overview.md` (architecture & flow)
- `docs/annotation_guide.md` (label definitions, negation, boundary rules)
- `docs/heuristic.md` (fuzzy/Jaccard thresholds, anatomy gating, examples)
- Later: `docs/error_analysis.md`, `docs/model_strategy.md`
## Repository Structure
```
src/
  evaluation/               # Evaluation metrics module (DONE)
    __init__.py
    metrics.py              # 10 evaluation functions
scripts/
  annotation/               # Annotation workflow scripts
    evaluate_llm_refinement.py  # CLI evaluation tool (DONE)
    plot_llm_metrics.py     # Visualization helper (DONE)
    cli.py                  # Unified CLI (TO ADD evaluate-llm subcommand)
    # Planned: export_weak_labels.py, convert_labelstudio.py
data/
  annotation/
    reports/                # Evaluation JSON + Markdown reports
    plots/                  # Generated visualizations
    exports/                # Converted gold JSONL (planned)
    conflicts/              # Adjudication workspace (planned)
tests/
  fixtures/
    annotation/             # Test fixtures for evaluation (DONE)
      weak_baseline.jsonl
      gold_with_llm_refined.jsonl
      gold_standard.jsonl
  test_evaluate_llm.py      # 27 evaluation tests (DONE)
docs/                       # Educational documentation (planned)
models/                     # Fine-tuned checkpoints (gitignored, planned)
```els/ (fine‑tuned checkpoints, gitignored)
```

## Model Strategy
- Primary encoder: BioBERT (retain biomedical semantic richness)
- Domain adaptation: Continued MLM on de‑identified complaints (seq len ≤256, 1–3 epochs CPU/GPU)
- Baseline comparator: RoBERTa‑base for colloquial phrasing contrast
- Head addition: `AutoModelForTokenClassification` wrapper once `labels.json` defined

## Heuristic & Threshold Summary
- Fuzzy threshold: 0.88 (WRatio)
- Jaccard token-set gate: ≥40
- Last-token alignment required for multi-token fuzzy spans
- Anatomy single-token skip (e.g., "skin", "face") unless symptom keyword co-occurs
- Negation window: 5 tokens; ≥50% overlap marks span negated
- Confidence: 0.8*fuzzy + 0.2*jaccard (clamped ≤1.0)

## Contributor Onboarding Checklist
1. Create virtual env & install (`pip install -r requirements.txt`)
2. Run `python scripts/verify_env.py` (device + model sanity)
## LLM Refinement & Evaluation (NEW - Phase 4.5 Complete)

### LLM Agent (`src/llm_agent.py`)
**Purpose**: Automated refinement of weak labels using LLM reasoning before human annotation.

**Capabilities**:
- **Boundary Correction**: Removes superfluous adjectives (severe, mild, slight) and determiners
- **Negation Validation**: Confirms negated spans maintain semantic accuracy
- **Canonical Normalization**: Maps colloquial terms to medical lexicon entries
- **Multi-Provider Support**: OpenAI (GPT-4), Azure OpenAI, Anthropic (Claude)
- **Resilience**: Exponential backoff retry (3 attempts), caching, structured output validation

**Configuration** (`AppConfig.llm_provider`):
```python
llm_provider: Optional[str] = "openai"  # or "azure", "anthropic"
llm_model: str = "gpt-4"  # or "gpt-4o-mini", "claude-3-5-sonnet-20241022"
llm_temperature: float = 0.1
llm_max_retries: int = 3
```

**Environment Variables**:
- OpenAI: `OPENAI_API_KEY`
- Azure: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`
- Anthropic: `ANTHROPIC_API_KEY`

**Test Coverage**: 15 tests (13 passed, 2 skipped for missing openai package in CI)

### Evaluation Harness (`src/evaluation/metrics.py`)

**Core Functions** (10 total):
1. `compute_overlap(span1, span2) -> int`: Token overlap count
2. `compute_iou(span1, span2) -> float`: Intersection over Union (0-1)
3. `compute_boundary_precision(pred_spans, gold_spans) -> Dict`: Exact match rate + mean/median IOU
4. `compute_iou_delta(weak_spans, llm_spans, gold_spans) -> Dict`: Tracks weak→LLM improvement
5. `compute_correction_rate(weak_spans, llm_spans, gold_spans) -> Dict`: Categorizes improved/worsened/unchanged
6. `calibration_curve(spans_with_confidence, gold_spans, n_bins) -> Dict`: Confidence reliability analysis
7. `stratify_by_confidence(spans, n_bins) -> Dict`: Bucketize by confidence score
8. `stratify_by_label(spans) -> Dict`: Group by entity label (SYMPTOM/PRODUCT)
9. `stratify_by_span_length(spans, buckets) -> Dict`: Group by character length
10. `compute_precision_recall_f1(pred_spans, gold_spans) -> Dict`: Standard NER metrics

**Test Coverage**: 27 tests across 7 test classes (100% passing)
- TestBasicMetrics: overlap, IOU calculations
- TestBoundaryPrecision: exact match rate, mean IOU
- TestIOUDelta: weak→LLM improvement tracking
- TestCorrectionRate: improved/worsened/unchanged categorization
- TestCalibration: binned confidence vs actual IOU
- TestStratification: confidence/label/length grouping
- TestPrecisionRecallF1: P/R/F1 with FP/FN cases
- TestEndToEnd: fixture loading and alignment

### Evaluation Script (`scripts/annotation/evaluate_llm_refinement.py`)

**Usage**:
```bash
python scripts/annotation/evaluate_llm_refinement.py \
  --weak data/weak_labels.jsonl \
  --refined data/llm_refined.jsonl \
  --gold data/gold_standard.jsonl \
  --output data/annotation/reports/evaluation.json \
  --markdown \
  --stratify label confidence span_length
```

**Output**:
- **JSON Report**: Comprehensive metrics with stratified breakdowns
- **Markdown Summary**: Human-readable tables and percentages
- **Metrics Included**:
  - Overall: IOU delta, boundary precision, correction stats, P/R/F1, calibration
  - Stratified: by label, confidence bucket, span length

**Example Results** (from test fixtures):
- IOU Improvement: +13.4% (0.882 → 1.000)
- Exact Match Rate: 66.7% → 100.0%
- Correction Rate: 100% improved (2/2 modified spans)
- F1 Score: 1.000 (perfect precision/recall on gold)

### Visualization Script (`scripts/annotation/plot_llm_metrics.py`)

**Dependencies**: `pip install -r requirements-viz.txt` (matplotlib, seaborn, numpy - optional)

**Usage**:
```bash
python scripts/annotation/plot_llm_metrics.py \
  --report data/annotation/reports/evaluation.json \
  --output-dir data/annotation/plots/ \
  --formats png pdf \
  --dpi 300 \
  --plots all  # or: iou calibration correction prf stratified
```

**Generated Plots**:
1. **IOU Uplift** (`iou_uplift.{png,pdf}`): Weak vs LLM distribution by confidence bucket or overall bar chart
2. **Calibration Curve** (`calibration_curve.{png,pdf}`): Confidence score reliability (expected vs observed IOU)
3. **Correction Rate** (`correction_rate.{png,pdf}`): Pie + bar chart of improved/worsened/unchanged spans
4. **P/R/F1 Comparison** (`prf_comparison.{png,pdf}`): Side-by-side weak vs LLM metrics with delta annotations
5. **Stratified Label** (`stratified_label.{png,pdf}`): F1 by entity type (SYMPTOM/PRODUCT)
6. **Stratified Confidence** (`stratified_confidence.{png,pdf}`): IOU delta across confidence buckets

**Design**: Publication-quality (300 DPI default), colorblind-safe palette, annotated with counts and deltas

### Workflow Integration (Recommended)

**Before Annotation**:
1. Generate weak labels: `python -m src.pipeline --input raw_text.txt --output weak_labels.jsonl`
2. Refine with LLM: `python -m src.llm_agent --weak weak_labels.jsonl --output llm_refined.jsonl`
3. Import to Label Studio: `python scripts/annotation/import_weak_to_labelstudio.py llm_refined.jsonl` (planned)

**After Annotation**:
1. Export gold labels: Label Studio → `gold_standard.jsonl`
2. Evaluate: `python scripts/annotation/evaluate_llm_refinement.py --weak weak_labels.jsonl --refined llm_refined.jsonl --gold gold_standard.jsonl --output reports/eval.json --markdown --stratify label confidence`
3. Visualize: `python scripts/annotation/plot_llm_metrics.py --report reports/eval.json --output-dir plots/ --formats png pdf`
4. Iterate: Adjust LLM prompts, thresholds, or weak labeling heuristics based on correction patterns

### Performance Benchmarks (Test Fixtures)
- **Boundary Corrections**: "severe burning sensation" → "burning sensation" (+18.2% IOU)
- **Adjective Removal**: "mild redness" → "redness" (+8.3% IOU)
- **Negation Confirmation**: LLM validates "no swelling" remains accurate (0% delta, expected)
- **Overall**: +13.4% mean IOU improvement, 100% exact match after refinement

### Cost Considerations
- OpenAI GPT-4: ~$0.03/1K input tokens, ~$0.06/1K output tokens
- Anthropic Claude 3.5 Sonnet: ~$0.003/1K input, ~$0.015/1K output
- Typical span: 50-150 tokens per refinement request
- Batch processing recommended; use caching for repeated contexts

### Limitations & Future Work
- LLM may over-correct colloquial phrasing (monitor via correction_rate worsened %)
- Calibration curve requires ≥50 spans per bucket for statistical reliability
- Visualization requires optional dependencies (matplotlib/seaborn)
- CLI integration pending (`scripts/annotation/cli.py` evaluate-llm subcommand)

## Updating This File
- Revise when thresholds change or new scripts/docs added
- Update LLM provider list if new integrations added
- Refresh benchmarks after production annotation batches
- Remove provisional notes once classification head operational

## Next Agent Steps
1. Add `evaluate-llm` subcommand to `scripts/annotation/cli.py` for unified workflow
2. Generate production evaluation report with first 100 annotated gold spans
3. Implement Label Studio import/export scripts (planned Phase 5)
4. Draft `docs/llm_refinement.md` with prompt engineering guidance

---
**Latest Update**: November 25, 2025 - Phase 4.5 complete + Phase 5 planned + Production guide ready
**Test Status**: 186/186 passing (100%) - Added 15 LLM agent tests, 27 evaluation tests
**Artifacts**: Evaluation harness (10 metrics), CLI integration (evaluate-llm, plot-metrics), 4 comprehensive guides (2,000+ lines)
**Production Status**: Ready for real-world annotation batches with full evaluation workflow
- Do not commit raw proprietary complaints
- Automatic GPU fallback to CPU with clear log output
- Defer mixed precision & gradient checkpointing until training begins
- Keep seed handling consistent for reproducibility

## Conventions
- Pure functions; pass dependencies explicitly
- Central config in `AppConfig` (extend vs. ad‑hoc globals)
- Typing everywhere; incremental mypy adoption later
- Tests small & deterministic (no network calls beyond Hugging Face cache)

## Updating This File
- Revise when thresholds change or new scripts/docs added
- Add evaluation metrics section after first supervised run
- Remove provisional notes once classification head operational

## Next Agent Step
- Implement annotation scripts, docs stubs, and integrity test prior to adding classification head.

---
Feedback: Confirm threshold defaults & priority (scripts vs docs) for next PR.
