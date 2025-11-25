# Copilot / AI Agent Instructions (Python + BioBERT NER)

> Scope: Active biomedical NER + weak labeling prototype using BioBERT, lexicon-driven span extraction, JSONL persistence, and upcoming annotation + education workflows.

## Project Overview
Implements configuration, BioBERT loading, lexicon-based weak labeling, exploratory notebook, and tests. Moving toward supervised token classification and human curation.

## Current State
- Model loader: `src/model.py` (BioBERT `dmis-lab/biobert-base-cased-v1.1`)
- Config: `src/config.py` (fields: `model_name`, `max_seq_len`, `device`, `seed`, `negation_window`, `fuzzy_scorer`)
- Weak labeling: `src/weak_label.py` (symptom/product lexicons, fuzzy 0.88, Jaccard ≥40, bidirectional negation window 5, emoji handling, anatomy skip, last‑token alignment)
- Pipeline: `src/pipeline.py` (simple inference + optional JSONL persistence)
- Lexicon build: `scripts/build_meddra_symptom_lexicon.py`
- Notebook: `scripts/Workbook.ipynb` (exploration & education)
- Environment check: `scripts/verify_env.py`
- Lexicons: `data/lexicon/symptoms.csv` (includes burning/burning sensation), `data/lexicon/products.csv`
- Output artifact: `data/output/notebook_test.jsonl`
- Test Suite: 144 tests (16 core, 98 edge cases, 26 integration, 4 curation) - 100% passing
- CI/CD: GitHub Actions workflows (test.yml, pre-commit.yml), pre-commit hooks, pyproject.toml config
- Dependencies: `requirements.txt`

## Roadmap Phases
1. Bootstrap & Lexicon (DONE)
2. Weak Label Refinement (DONE - bidirectional negation, emoji handling)
3. Test Infrastructure & Edge Cases (DONE - 144/144 tests passing)
4. CI/CD Integration (DONE - GitHub Actions, pre-commit hooks)
5. Annotation & Curation (Label Studio integration) (UPCOMING)
6. Gold Standard Assembly
7. Token Classification Fine‑Tune (BioBERT + classification head)
8. Domain Adaptation (MLM on complaints corpus)
9. Baseline Comparison (RoBERTa)
10. Evaluation & Calibration (precision/recall/F1, confidence thresholds)
11. Educational Docs Expansion
12. Continuous Improvement & Active Learning

## Annotation & Curation Plan (Planned)
- Tool: Label Studio (local, telemetry disabled)
- Workflow: export weak labels → import → human refine → export gold → convert → integrity tests
- Privacy: strip PII before ingestion; keep raw complaints outside repo (`data/raw/` ignored)
- Disable analytics: PowerShell `setx LABEL_STUDIO_DISABLE_TELEMETRY 1` (or `$env:LABEL_STUDIO_DISABLE_TELEMETRY=1` for session)
- Scripts (to add): `scripts/annotation/export_weak_labels.py`, `scripts/annotation/convert_labelstudio.py`
- Gold format: JSONL lines with `{id, text, entities:[{start,end,label}]}`
- QC: Inter‑annotator agreement sample, coverage report, span integrity test

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

## Repository Structure Additions (Planned)
```
docs/
scripts/annotation/
data/annotation/exports/ (converted gold JSONL)
models/ (fine‑tuned checkpoints, gitignored)
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
3. Execute `pytest -q` (ensure baseline tests pass)
4. Inspect lexicon CSVs for formatting
5. Run notebook to view weak labels
6. Generate weak labels → JSONL
7. Start Label Studio (when scripts added) & import JSONL
8. Curate a sample batch; export and convert
9. Run integrity test on converted gold
10. Propose PR (small, focused) referencing roadmap phase

## Safety & Performance
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
