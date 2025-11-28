# Phase 6: Gold Standard Assembly Guide

**Status**: ACTIVE  
**Version**: 1.0  
**Date**: November 28, 2025  
**Prerequisites**: Phases 1-5 Complete (Weak Labeling, LLM Refinement, Annotation Infrastructure)

---

## Table of Contents

1. [Overview](#overview)
2. [Objectives](#objectives)
3. [Prerequisites Checklist](#prerequisites-checklist)
4. [Implementation Steps](#implementation-steps)
5. [Quality Assurance](#quality-assurance)
6. [Success Metrics](#success-metrics)
7. [Troubleshooting](#troubleshooting)
8. [Next Steps (Phase 7)](#next-steps-phase-7)

---

## Overview

Phase 6 focuses on assembling high-quality gold standard annotations through systematic human curation of weak labels and LLM-refined spans. This gold standard dataset will serve as:

- **Training data** for supervised BioBERT fine-tuning (Phase 7)
- **Evaluation benchmark** for measuring model performance
- **Quality reference** for ongoing active learning iterations

### Key Activities

1. ✅ **Batch Preparation**: Sample representative complaints using stratified sampling
2. ✅ **Weak Label Generation**: Apply lexicon-based NER to sampled texts
3. ✅ **LLM Refinement**: Enhance weak labels with LLM boundary correction (optional)
4. ✅ **Human Annotation**: Expert review and correction in Label Studio
5. ✅ **Consensus & Adjudication**: Resolve annotator disagreements
6. ✅ **Gold Export**: Convert validated annotations to training-ready JSONL
7. ✅ **Quality Validation**: Verify integrity, coverage, and agreement metrics

---

## Objectives

### Primary Goals

1. **Assemble 500-1000 gold-annotated complaints**
   - Target: 500 minimum, 1000 ideal for robust training
   - Balanced across span densities and product categories
   - Representative of FDA CAERS complaint language

2. **Achieve >0.75 inter-annotator agreement (IOU ≥ 0.5)**
   - Measured on overlapping annotation batches
   - Cohen's kappa for label agreement
   - Boundary precision >80% exact match

3. **Maintain >90% canonical coverage**
   - All SYMPTOM spans map to lexicon entries
   - Consistent terminology across annotators
   - Clear handling of negation and ambiguity

### Secondary Goals

- Document annotation guidelines and edge cases
- Build annotator calibration dataset (50-100 examples)
- Establish quality monitoring dashboards
- Create reproducible annotation SOP

---

## Prerequisites Checklist

Before starting Phase 6, ensure:

- ✅ **Weak Labeling Module**: `src/weak_labeling/` fully tested (531/531 tests passing)
- ✅ **LLM Agent**: `src/llm_agent.py` operational (stub/OpenAI/Anthropic support)
- ✅ **Evaluation Metrics**: `src/evaluation/metrics.py` implemented (10 functions)
- ✅ **Annotation Scripts**: All `scripts/annotation/*.py` scripts present
- ✅ **Label Studio Setup**: Local instance configured (telemetry disabled)
- ✅ **Documentation**: `docs/annotation_guide.md`, `docs/production_workflow.md`
- ✅ **CAERS Data**: FDA complaints downloaded and weak-labeled
- ✅ **Lexicons**: Symptom and product lexicons up-to-date

### Environment Setup

```powershell
# Activate environment
conda activate NER

# Verify all dependencies
pip install -r requirements.txt
pip install -r requirements-llm.txt  # Optional: for LLM refinement

# Check annotation scripts
python scripts/annotation/cli.py --help

# Verify Label Studio (if using)
label-studio --version
```

---

## Implementation Steps

### Step 1: Prepare Initial Batch (n=100)

**Purpose**: Create first annotation batch with stratified sampling

```powershell
# Download CAERS cosmetics data (if not already done)
python scripts/caers/download_caers.py `
  --output data/caers/cosmetics_1000.jsonl `
  --categories cosmetics `
  --limit 1000 `
  --min-spans 1

# Prepare annotation batch
python scripts/annotation/prepare_production_batch.py `
  --input data/caers/cosmetics_1000.jsonl `
  --output data/annotation/exports/batch_001.jsonl `
  --n-tasks 100 `
  --strategy stratified `
  --min-spans 1 `
  --max-text-len 500 `
  --stats-output data/annotation/reports/batch_001_stats.json

# Review batch statistics
Get-Content data/annotation/reports/batch_001_stats.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**Expected Output**:
- `batch_001.jsonl`: 100 tasks with weak labels
- Stratified by span density (low/medium/high)
- Avg 2-3 spans per task
- Avg text length 150-250 characters

---

### Step 2: Optional LLM Refinement

**Purpose**: Improve boundary precision before human annotation

```powershell
# Refine with OpenAI GPT-4 (requires OPENAI_API_KEY)
python scripts/annotation/refine_llm.py `
  --input data/annotation/exports/batch_001.jsonl `
  --output data/annotation/exports/batch_001_refined.jsonl `
  --provider openai `
  --model gpt-4o-mini `
  --temperature 0.1

# OR use stub mode (no API calls)
python scripts/annotation/refine_llm.py `
  --input data/annotation/exports/batch_001.jsonl `
  --output data/annotation/exports/batch_001_refined.jsonl `
  --provider stub
```

**Decision Matrix**:
- **Use LLM**: If boundary precision <70%, budget allows API costs
- **Skip LLM**: If weak labels already high quality, cost-sensitive, offline workflow

---

### Step 3: Import to Label Studio

**Purpose**: Load tasks into annotation interface

```powershell
# Initialize Label Studio project (first time only)
python scripts/annotation/init_label_studio_project.py `
  --project-name "SpanForge Gold Standard" `
  --config data/annotation/config/label_config.xml

# Import batch tasks
python scripts/annotation/import_weak_to_labelstudio.py `
  --input data/annotation/exports/batch_001_refined.jsonl `
  --project-id 1 `
  --confidence-filter 0.6
```

**Alternative**: Manual annotation without Label Studio (see `docs/annotation_guide.md`)

---

### Step 4: Human Annotation

**Purpose**: Expert review and span correction

**Workflow**:
1. Launch Label Studio: `label-studio start`
2. Navigate to project: http://localhost:8080
3. Review each task:
   - ✅ Correct span boundaries (remove extra words, fix punctuation)
   - ✅ Add missing spans (false negatives)
   - ✅ Remove incorrect spans (false positives)
   - ✅ Verify negation flags
   - ✅ Confirm canonical mappings
4. Submit completed tasks

**Guidelines** (see `docs/annotation_guide.md` for full details):
- Include full clinical phrase (e.g., "severe burning sensation" not just "burning")
- Exclude trailing punctuation
- Mark negated spans but don't delete them
- Prefer specific terms over generic anatomy
- When uncertain, add note in task comments

**Time Estimate**: 2-3 minutes per task (100 tasks = 3-5 hours)

---

### Step 5: Export & Convert

**Purpose**: Extract validated annotations from Label Studio

```powershell
# Export from Label Studio UI: Settings → Export → JSON

# Convert to SpanForge gold format
python scripts/annotation/convert_labelstudio.py `
  --input data/annotation/exports/label_studio_export.json `
  --output data/annotation/exports/batch_001_gold.jsonl `
  --consensus majority `
  --min-agree 2
```

**Validation Checks**:
- All tasks have at least 1 annotator
- Span boundaries align with text slices
- Canonical fields populated
- No duplicate spans

---

### Step 6: Quality Assurance

**Purpose**: Verify annotation quality and consistency

```powershell
# Generate quality report
python scripts/annotation/quality_report.py `
  --gold data/annotation/exports/batch_001_gold.jsonl `
  --output data/annotation/reports/batch_001_quality.md

# Compare weak vs LLM vs gold
python scripts/annotation/evaluate_llm_refinement.py `
  --weak data/annotation/exports/batch_001.jsonl `
  --refined data/annotation/exports/batch_001_refined.jsonl `
  --gold data/annotation/exports/batch_001_gold.jsonl `
  --output data/annotation/reports/batch_001_evaluation.json `
  --markdown `
  --stratify label confidence span_length

# Generate visualizations
python scripts/annotation/plot_llm_metrics.py `
  --report data/annotation/reports/batch_001_evaluation.json `
  --output-dir data/annotation/plots/batch_001/ `
  --formats png pdf `
  --dpi 300 `
  --plots all
```

**Review Metrics**:
- **IOU Improvement**: +8-15% (weak → LLM vs gold)
- **Exact Match Rate**: 70-85% (LLM boundaries vs gold)
- **Correction Rate**: >60% improved, <10% worsened
- **F1 Score**: >0.85 (LLM precision/recall vs gold)
- **Inter-Annotator Agreement**: >0.75 (if multiple annotators)

---

### Step 7: Adjudication (If Needed)

**Purpose**: Resolve annotator disagreements

```powershell
# Identify conflicts
python scripts/annotation/adjudicate.py `
  --gold data/annotation/exports/batch_001_gold.jsonl `
  --output data/annotation/conflicts/batch_001_conflicts.json `
  --strategy flag

# Manual review of conflicts
# Edit conflicts JSON to mark resolutions

# Apply resolutions
python scripts/annotation/adjudicate.py `
  --gold data/annotation/exports/batch_001_gold.jsonl `
  --conflicts data/annotation/conflicts/batch_001_conflicts_resolved.json `
  --output data/annotation/exports/batch_001_gold_adjudicated.jsonl `
  --strategy resolve
```

---

### Step 8: Register Batch

**Purpose**: Track batch provenance and metadata

```powershell
python scripts/annotation/register_batch.py `
  --gold-file data/annotation/exports/batch_001_gold.jsonl `
  --batch-id batch_001 `
  --n-tasks 100 `
  --annotators "annotator_1,annotator_2" `
  --revision 1 `
  --notes "Initial gold standard batch - cosmetics complaints"
```

**Registry Entry** (appended to `data/annotation/registry.csv`):
```csv
timestamp,batch_id,gold_file,n_tasks,annotators,revision,notes
2025-11-28T14:30:00,batch_001,data/annotation/exports/batch_001_gold.jsonl,100,annotator_1;annotator_2,1,Initial gold standard batch - cosmetics complaints
```

---

### Step 9: Iterate (Repeat for Additional Batches)

**Purpose**: Build up to 500-1000 gold annotations

**Recommended Batch Schedule**:
1. **Batch 001** (n=100): Cosmetics, initial calibration
2. **Batch 002** (n=100): Cosmetics, refined guidelines
3. **Batch 003** (n=100): Supplements, category expansion
4. **Batch 004** (n=100): Personal care, diverse language
5. **Batches 005-010** (n=100 each): Scale to 1000 total

**Iteration Strategy**:
- Review quality reports after each batch
- Update `docs/annotation_guide.md` with new edge cases
- Calibrate annotators on disagreements
- Adjust weak labeling thresholds if needed
- Track time-to-annotate and adjust task complexity

---

## Quality Assurance

### Integrity Checks

Run automated validation on all gold batches:

```powershell
# Test curation integrity
pytest tests/test_curation_integrity.py -v

# Expected: All spans validate, text alignment correct, no orphaned canonicals
```

### Manual Spot Checks

Periodically review:
- 10 random tasks from each batch
- All tasks with >5 disagreements
- Tasks with 0 spans (verify true negatives)
- Tasks flagged in quality reports

### Dashboard Monitoring

Track cumulative metrics:
- Total gold annotations: 0 → 500 → 1000
- Avg spans per task: 2-3 stable
- Label distribution: 85% SYMPTOM / 15% PRODUCT target
- Annotator throughput: 2-3 min/task stable
- Agreement trends: improving with calibration

---

## Success Metrics

### Phase 6 Complete When:

✅ **Quantity**: ≥500 gold-annotated complaints (1000 ideal)  
✅ **Quality**: Inter-annotator agreement >0.75 (IOU ≥ 0.5)  
✅ **Coverage**: >90% canonical mapping for symptom spans  
✅ **Balance**: Stratified by span density and product category  
✅ **Documentation**: Annotation guidelines finalized with examples  
✅ **Provenance**: All batches registered in `registry.csv`  
✅ **Validation**: All integrity tests passing

### Quantitative Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Total Gold Tasks | 500-1000 | 0 | 🔴 Not started |
| Inter-Annotator Agreement (IOU) | >0.75 | - | 🟡 Pending |
| Canonical Coverage | >90% | - | 🟡 Pending |
| SYMPTOM Spans | 400-900 | 0 | 🔴 Not started |
| PRODUCT Spans | 100-300 | 0 | 🔴 Not started |
| Avg Time/Task | <3 min | - | 🟡 Pending |

---

## Troubleshooting

### Issue: Low Inter-Annotator Agreement (<0.60)

**Causes**:
- Unclear boundary rules
- Ambiguous symptom definitions
- Inconsistent negation handling

**Solutions**:
- Conduct calibration session with annotators
- Add examples to `docs/annotation_guide.md`
- Create practice batch with known-correct annotations
- Use adjudication to establish precedents

---

### Issue: High False Positive Rate in Weak Labels

**Causes**:
- Fuzzy threshold too low (0.88 → 0.92)
- Jaccard threshold too low (40 → 50)
- Lexicon contains non-specific terms

**Solutions**:
- Adjust thresholds in `src/weak_labeling/labeler.py`
- Refine lexicons to remove ambiguous terms
- Use LLM refinement to filter low-confidence spans

---

### Issue: Slow Annotation Throughput (>5 min/task)

**Causes**:
- Tasks too long (>500 chars)
- Too many pre-annotated spans (>10)
- Complex medical terminology

**Solutions**:
- Reduce `--max-text-len` in batch preparation
- Filter to simpler language (Flesch-Kincaid >8th grade)
- Provide terminology glossary
- Split long tasks into shorter segments

---

### Issue: PII Exposure in Raw Complaints

**Causes**:
- CAERS data contains user-submitted PII
- De-identification patterns missed edge cases

**Solutions**:
- Use `--deidentify` flag in batch preparation
- Manual review of high-risk fields (age, gender, dates)
- Keep raw CAERS data in `data/raw/` (gitignored)
- Audit exported gold files before committing

---

## Next Steps (Phase 7)

Once Phase 6 is complete with ≥500 gold annotations:

### Phase 7: Token Classification Fine-Tuning

1. **Prepare Training Data**: Convert gold JSONL to BIO-tagged format
2. **Split Dataset**: 80% train / 10% validation / 10% test
3. **Add Classification Head**: Extend BioBERT with token classification layer
4. **Fine-Tune**: 3-5 epochs with learning rate 2e-5
5. **Evaluate**: Compare weak labels vs fine-tuned model
6. **Iterate**: Active learning to identify hard examples

**Target Metrics** (Phase 7):
- **Precision**: >0.90 (SYMPTOM), >0.85 (PRODUCT)
- **Recall**: >0.85 (SYMPTOM), >0.80 (PRODUCT)
- **F1**: >0.87 (SYMPTOM), >0.82 (PRODUCT)

**See**: `docs/phase_7_training.md` (to be created)

---

## Appendix A: CLI Quick Reference

```powershell
# Prepare batch
python scripts/annotation/prepare_production_batch.py --input CAERS.jsonl --output batch.jsonl --n-tasks 100

# LLM refinement
python scripts/annotation/refine_llm.py --input batch.jsonl --output refined.jsonl --provider openai

# Convert Label Studio export
python scripts/annotation/convert_labelstudio.py --input export.json --output gold.jsonl --consensus majority

# Quality report
python scripts/annotation/quality_report.py --gold gold.jsonl --output report.md

# Evaluation
python scripts/annotation/evaluate_llm_refinement.py --weak batch.jsonl --refined refined.jsonl --gold gold.jsonl --output eval.json --markdown

# Visualize
python scripts/annotation/plot_llm_metrics.py --report eval.json --output-dir plots/ --plots all

# Register batch
python scripts/annotation/register_batch.py --gold-file gold.jsonl --batch-id batch_001 --n-tasks 100 --annotators "user1,user2"
```

---

## Appendix B: Annotation Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 6 WORKFLOW                         │
└─────────────────────────────────────────────────────────────┘

Step 1: Batch Preparation
    ├─→ Download CAERS data (cosmetics, supplements, etc.)
    ├─→ Filter by quality (min spans, text length)
    ├─→ Stratified sampling (span density, category)
    └─→ Output: batch_XXX.jsonl (100 tasks)
         │
         ↓
Step 2: Weak Labeling (Already Done)
    ├─→ Lexicon matching (fuzzy + Jaccard)
    ├─→ Negation detection (bidirectional window)
    └─→ Confidence scoring
         │
         ↓
Step 3: LLM Refinement (Optional)
    ├─→ Boundary correction (remove adjectives)
    ├─→ Negation validation
    └─→ Canonical normalization
         │
         ↓
Step 4: Human Annotation
    ├─→ Load into Label Studio
    ├─→ Expert review + correction
    ├─→ Add missing, remove false positives
    └─→ Mark negation, verify canonicals
         │
         ↓
Step 5: Export & Convert
    ├─→ Export from Label Studio (JSON)
    ├─→ Convert to SpanForge format (JSONL)
    └─→ Apply consensus strategy
         │
         ↓
Step 6: Quality Validation
    ├─→ Integrity checks (span alignment, canonicals)
    ├─→ Agreement metrics (IOU, kappa)
    ├─→ Evaluation report (weak → LLM → gold)
    └─→ Visualizations (IOU uplift, calibration)
         │
         ↓
Step 7: Adjudication (If Conflicts)
    ├─→ Identify disagreements
    ├─→ Manual resolution
    └─→ Update gold file
         │
         ↓
Step 8: Register Batch
    ├─→ Add to registry.csv
    ├─→ Record provenance (annotators, date)
    └─→ Archive reports
         │
         ↓
Step 9: Iterate (Repeat Until 500-1000 Gold)
    └─→ Next batch with refined guidelines
```

---

## Appendix C: File Locations

```
data/
├── annotation/
│   ├── config/
│   │   └── label_config.xml          # Label Studio configuration
│   ├── exports/
│   │   ├── batch_001.jsonl            # Weak labels (input)
│   │   ├── batch_001_refined.jsonl    # LLM refined (optional)
│   │   ├── batch_001_gold.jsonl       # Gold standard (output)
│   │   └── *.json                     # Label Studio exports
│   ├── reports/
│   │   ├── batch_001_stats.json       # Batch statistics
│   │   ├── batch_001_quality.md       # Quality report
│   │   └── batch_001_evaluation.json  # Evaluation metrics
│   ├── plots/
│   │   └── batch_001/                 # Visualization outputs
│   ├── conflicts/
│   │   └── batch_001_conflicts.json   # Adjudication workspace
│   └── registry.csv                   # Batch provenance log
│
├── caers/
│   ├── raw/                           # Downloaded CAERS CSV (gitignored)
│   └── *.jsonl                        # Processed weak-labeled complaints
│
└── lexicon/
    ├── symptoms.csv                   # Symptom lexicon
    └── products.csv                   # Product lexicon
```

---

**Document Version**: 1.0  
**Last Updated**: November 28, 2025  
**Next Review**: After Batch 001 complete  
**Owner**: SpanForge Project Team
