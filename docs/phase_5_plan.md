# Phase 5 Implementation Plan: Label Studio Integration

**Version**: 1.0  
**Start Date**: November 25, 2025  
**Target**: Complete annotation workflow with LLM-refined weak labels  
**Status**: 🟡 PLANNING

---

## Overview

Phase 5 integrates the evaluation harness (Phase 4.5) with Label Studio for human annotation, establishing a complete pipeline: **weak labels → LLM refinement → human curation → gold standard → iterative improvement**.

---

## Current State Assessment

### ✅ Existing Infrastructure (Ready to Use)

**Scripts** (already implemented):
1. ✅ `import_weak_to_labelstudio.py` - Converts weak JSONL to Label Studio tasks
2. ✅ `convert_labelstudio.py` - Exports Label Studio JSON to normalized gold JSONL
3. ✅ `adjudicate.py` - Majority vote consensus across multiple annotators
4. ✅ `quality_report.py` - Annotation quality metrics (IAA, label distribution, conflicts)
5. ✅ `register_batch.py` - Provenance tracking in registry.csv
6. ✅ `init_label_studio_project.py` - Project bootstrap with telemetry disabled
7. ✅ `refine_llm.py` - LLM-based weak label refinement
8. ✅ `evaluate_llm_refinement.py` - Comprehensive evaluation harness
9. ✅ `plot_llm_metrics.py` - Visualization generator
10. ✅ `cli.py` - Unified CLI wrapper (now includes evaluate-llm, plot-metrics)

**Infrastructure**:
- LLM agent with multi-provider support (OpenAI, Azure, Anthropic)
- Evaluation metrics (10 functions: IOU, boundary precision, correction rate, calibration, P/R/F1)
- Test fixtures (171 tests passing)
- Comprehensive documentation

### 🟡 Gaps to Address

**Missing Components**:
1. **Tutorial Notebook** (`scripts/AnnotationWalkthrough.ipynb`) - Interactive guide for annotators
2. **Label Studio Config** (`data/annotation/config/label_config.xml`) - Entity type definitions
3. **Annotation Guidelines** (`docs/annotation_guide.md`) - Needs expansion with examples
4. **Workflow Orchestration** - End-to-end automation script
5. **Confidence Filtering** - Optimize LLM refinement by skipping high-confidence weak labels

---

## Phase 5 Objectives

### Primary Goals
1. **Enable Human Annotation** - Complete Label Studio integration for production use
2. **Establish Gold Standard** - Curate 100+ high-quality gold annotations
3. **Measure Improvement** - Quantify weak → LLM → gold quality gains
4. **Iterate Workflows** - Refine heuristics and LLM prompts based on evaluation

### Success Metrics
- ✅ 100+ gold-annotated spans with provenance
- ✅ Inter-annotator agreement (IOU ≥0.5) >0.75 for calibration set
- ✅ Evaluation report showing +10-15% IOU improvement from weak → LLM
- ✅ <10% "worsened" correction rate (LLM doesn't introduce errors)
- ✅ Annotation guidelines tested with 3+ annotators

---

## Implementation Tasks

### Task 1: Label Studio Configuration (Priority: HIGH)

**Files to Create**:
```xml
<!-- data/annotation/config/label_config.xml -->
<View>
  <Header value="Annotate Symptoms and Products"/>
  <Text name="text" value="$text"/>
  <Labels name="label" toName="text">
    <Label value="SYMPTOM" background="#2ca02c"/>
    <Label value="PRODUCT" background="#1f77b4"/>
  </Labels>
</View>
```

**Requirements**:
- Two entity types: SYMPTOM, PRODUCT
- Color-coded for visual distinction
- Simple interface (no nested entities initially)

**Testing**:
- Import config to Label Studio
- Verify text selection highlights correctly
- Confirm export JSON structure matches convert_labelstudio.py expectations

---

### Task 2: Tutorial Notebook (Priority: HIGH)

**File**: `scripts/AnnotationWalkthrough.ipynb`

**Sections**:
1. **Introduction** - Why annotation matters, pipeline overview
2. **Data Preparation** - Load weak labels, show examples with confidence scores
3. **LLM Refinement Demo** - Run refine_llm.py on sample, compare before/after
4. **Label Studio Setup** - Environment setup, telemetry disable, project creation
5. **Annotation Practice** - 5 example texts with correct/incorrect weak labels
6. **Export & Evaluation** - Convert exports, run evaluation harness, interpret results
7. **Common Mistakes** - Boundary errors, negation handling, anatomy tokens
8. **Glossary** - Canonical symptom terms (redness/erythema, pruritus/itching)

**Interactive Elements**:
- Code cells for running pipeline steps
- Markdown with screenshots of Label Studio interface
- Quiz questions (e.g., "Should 'no redness' be annotated?" - Yes, with negation flag)

---

### Task 3: Annotation Guidelines Expansion (Priority: HIGH)

**File**: `docs/annotation_guide.md` (expand existing)

**New Sections**:

**1. Boundary Rules** (with examples):
```
✅ CORRECT: "burning sensation" (span: 22-40)
❌ INCORRECT: "severe burning sensation" (span: 15-43) - adjective included
❌ INCORRECT: "burning" (span: 22-29) - incomplete medical term

✅ CORRECT: "redness and swelling" (two separate spans: "redness", "swelling")
❌ INCORRECT: "redness and swelling" (single span) - conjunction should be excluded
```

**2. Negation Policy**:
- Annotate negated symptoms (e.g., "no redness") as SYMPTOM spans
- Add metadata flag `"negated": true` if Label Studio supports custom attributes
- Rationale: Model can learn negation context; skipping loses training signal

**3. Anatomy Gating**:
- Single anatomy tokens ("skin", "face", "arm") → skip unless part of symptom phrase
- ✅ "facial swelling" → annotate "swelling" only (or "facial swelling" if lexicon has compound term)
- ❌ "face" alone → skip

**4. Ambiguous Cases**:
- "dry skin" vs "dryness" → prefer canonical term from lexicon (likely "dryness")
- Colloquial phrasing ("it hurt a lot") → map to canonical ("pain")
- Multi-word symptoms → include full phrase if in lexicon ("burning sensation", not "burning")

**5. Overlapping Suggestions**:
- When weak label overlaps gold boundary: choose semantically complete span
- If uncertain: annotate both, flag for adjudication

---

### Task 4: Workflow Orchestration Script (Priority: MEDIUM)

**File**: `scripts/annotation/workflow.py`

**Purpose**: End-to-end automation from raw text → gold standard

**Steps**:
```bash
# 1. Generate weak labels
python -m src.pipeline --input data/raw/complaints.txt --output data/weak/batch_001.jsonl

# 2. Filter by confidence (optional - optimize LLM cost)
python scripts/annotation/workflow.py filter-confidence \
  --input data/weak/batch_001.jsonl \
  --output data/weak/batch_001_filtered.jsonl \
  --threshold 0.80 --action below  # Only refine low-confidence spans

# 3. Refine with LLM
python scripts/annotation/cli.py refine-llm \
  --weak data/weak/batch_001_filtered.jsonl \
  --output data/llm/batch_001_refined.jsonl \
  --provider openai --model gpt-4o-mini

# 4. Import to Label Studio
python scripts/annotation/cli.py import-weak \
  --weak data/llm/batch_001_refined.jsonl \
  --out data/annotation/imports/batch_001_tasks.json \
  --include-preannotated \
  --push --project-id 42

# 5. Human annotation (manual step in Label Studio UI)

# 6. Export from Label Studio (manual or API)
# Produces: data/annotation/raw/batch_001_export.json

# 7. Convert to gold JSONL
python scripts/annotation/cli.py convert \
  --input data/annotation/raw/batch_001_export.json \
  --output data/gold/batch_001.jsonl \
  --source complaints_batch_001 \
  --annotator alice \
  --symptom-lexicon data/lexicon/symptoms.csv \
  --product-lexicon data/lexicon/products.csv

# 8. Evaluate
python scripts/annotation/cli.py evaluate-llm \
  --weak data/weak/batch_001.jsonl \
  --refined data/llm/batch_001_refined.jsonl \
  --gold data/gold/batch_001.jsonl \
  --output data/annotation/reports/batch_001_eval.json \
  --markdown \
  --stratify label confidence

# 9. Visualize
python scripts/annotation/cli.py plot-metrics \
  --report data/annotation/reports/batch_001_eval.json \
  --output-dir data/annotation/plots/batch_001/ \
  --formats png pdf

# 10. Register batch
python scripts/annotation/cli.py register \
  --batch-id batch_001 \
  --n-tasks 50 \
  --annotators alice \
  --registry data/annotation/registry.csv
```

**Orchestration Features**:
- Single command to run steps 1-4: `python scripts/annotation/workflow.py prepare-batch ...`
- Resume from checkpoint (e.g., skip LLM if already refined)
- Dry-run mode to preview actions
- Progress tracking with ETA

---

### Task 5: Confidence-Based Filtering (Priority: MEDIUM)

**Purpose**: Reduce LLM API costs by only refining uncertain weak labels

**Implementation** (`scripts/annotation/filter_confidence.py`):

```python
def filter_by_confidence(records, threshold: float, action: str):
    """Filter spans based on confidence threshold.
    
    Args:
        records: List of JSONL records with spans
        threshold: Confidence cutoff (0-1)
        action: 'above' (keep ≥threshold), 'below' (keep <threshold)
    
    Returns:
        Filtered records (maintains JSONL structure)
    """
    filtered = []
    for rec in records:
        spans = rec.get('spans', [])
        if action == 'below':
            filtered_spans = [s for s in spans if s.get('confidence', 1.0) < threshold]
        else:  # 'above'
            filtered_spans = [s for s in spans if s.get('confidence', 1.0) >= threshold]
        
        if filtered_spans:  # Only include records with remaining spans
            rec_copy = rec.copy()
            rec_copy['spans'] = filtered_spans
            filtered.append(rec_copy)
    
    return filtered
```

**Usage**:
```bash
# Only refine spans with confidence < 0.80
python scripts/annotation/filter_confidence.py \
  --input weak.jsonl \
  --output weak_low_conf.jsonl \
  --threshold 0.80 \
  --action below

# Then refine only low-confidence spans
python scripts/annotation/cli.py refine-llm \
  --weak weak_low_conf.jsonl \
  --output llm_refined.jsonl
```

**Cost Savings Example**:
- Dataset: 1,000 spans
- High-confidence (≥0.80): 600 spans (60%) → skip LLM
- Low-confidence (<0.80): 400 spans (40%) → refine with LLM
- **Cost reduction: 60%** (from $7.20 → $2.88 for GPT-4 per 1K spans)

---

### Task 6: Multi-Annotator Workflow (Priority: LOW - Future)

**Current State**: Single annotator supported in convert_labelstudio.py

**Enhancement**: Support multiple annotators with IAA calculation

**Files to Modify**:
1. `convert_labelstudio.py` - Accept `--annotator` flag or infer from Label Studio user field
2. `adjudicate.py` - Already supports multiple gold JSONL files
3. `quality_report.py` - Already computes pairwise agreement (IOU ≥0.5)

**Workflow**:
```bash
# Annotator A exports
python scripts/annotation/convert_labelstudio.py \
  --input ls_export_A.json --output gold_A.jsonl --annotator alice

# Annotator B exports
python scripts/annotation/convert_labelstudio.py \
  --input ls_export_B.json --output gold_B.jsonl --annotator bob

# Compute IAA
python scripts/annotation/cli.py quality \
  --gold gold_A.jsonl gold_B.jsonl \
  --out reports/iaa_batch_001.json

# Adjudicate conflicts
python scripts/annotation/cli.py adjudicate \
  --inputs gold_A.jsonl gold_B.jsonl \
  --out gold_consensus.jsonl \
  --conflicts conflicts/batch_001.json \
  --min-agree 2
```

**IAA Metrics**:
- Pairwise IOU ≥0.5 agreement rate
- Fleiss' kappa for multi-annotator (3+ annotators)
- Span-level confusion matrix (SYMPTOM vs PRODUCT disagreements)

---

### Task 7: Production Data Preparation (Priority: HIGH)

**Goal**: Prepare first 100 real adverse event reports for annotation

**Data Requirements**:
- **Privacy**: De-identify PII (names, dates, locations) before annotation
- **Format**: Plain text, one complaint per line or JSONL with `{"id": ..., "text": ...}`
- **Selection**: Stratified sample (25% simple, 50% moderate, 25% complex based on weak label confidence distribution)

**Preparation Script** (`scripts/annotation/prepare_production_batch.py`):

```python
def prepare_batch(input_file, output_dir, batch_size=100, stratify=True):
    """
    Prepare annotation batch with stratified sampling.
    
    Steps:
    1. Generate weak labels for all inputs
    2. Stratify by mean confidence per document:
       - Simple: mean confidence ≥0.85 (25%)
       - Moderate: 0.70 ≤ mean confidence < 0.85 (50%)
       - Complex: mean confidence < 0.70 (25%)
    3. Sample batch_size documents (stratified)
    4. Run LLM refinement on sampled batch
    5. Export ready-to-import Label Studio JSON
    
    Outputs:
    - data/annotation/batches/batch_001/weak.jsonl
    - data/annotation/batches/batch_001/llm_refined.jsonl
    - data/annotation/batches/batch_001/tasks.json (Label Studio import)
    - data/annotation/batches/batch_001/manifest.json (metadata)
    """
```

**Manifest Structure**:
```json
{
  "batch_id": "batch_001",
  "created": "2025-11-25T15:30:00Z",
  "n_tasks": 100,
  "stratification": {
    "simple": 25,
    "moderate": 50,
    "complex": 25
  },
  "weak_label_stats": {
    "mean_confidence": 0.78,
    "total_spans": 342,
    "label_distribution": {"SYMPTOM": 298, "PRODUCT": 44}
  },
  "llm_refinement": {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "total_cost_usd": 0.48,
    "modified_spans": 87
  }
}
```

---

## Timeline & Milestones

### Week 1: Foundation (Days 1-5)
- ✅ Day 1: CLI integration complete (evaluate-llm, plot-metrics)
- 🟡 Day 2-3: Create label config + tutorial notebook
- 🟡 Day 4-5: Expand annotation guidelines with examples

### Week 2: Production Preparation (Days 6-10)
- 🟡 Day 6-7: Implement confidence filtering + workflow orchestration
- 🟡 Day 8-9: Prepare first production batch (100 complaints)
- 🟡 Day 10: Pilot annotation session (2-3 annotators, 10 tasks each)

### Week 3: Annotation & Evaluation (Days 11-15)
- 🟡 Day 11-13: Full annotation batch (100 tasks)
- 🟡 Day 14: Run evaluation harness on gold data
- 🟡 Day 15: Analyze results, refine prompts/heuristics

### Week 4: Iteration & Documentation (Days 16-20)
- 🟡 Day 16-17: Second batch with improved prompts
- 🟡 Day 18-19: Multi-annotator IAA analysis (if applicable)
- 🟡 Day 20: Phase 5 completion report + Phase 6 planning

---

## Risk Mitigation

### Risk 1: Low Inter-Annotator Agreement
**Likelihood**: Medium  
**Impact**: High (unreliable gold standard)

**Mitigation**:
- Calibration session: 3 annotators annotate same 20 tasks, discuss disagreements
- Clear examples in annotation guide (correct vs incorrect)
- Weekly sync meetings to clarify edge cases

### Risk 2: LLM Over-Correction
**Likelihood**: Medium  
**Impact**: Medium (introduces errors)

**Mitigation**:
- Monitor "worsened" correction rate in evaluation reports
- If >10%, reduce temperature or add negative examples to prompts
- Confidence filtering: skip high-confidence weak labels (≥0.85)

### Risk 3: Annotation Fatigue
**Likelihood**: High  
**Impact**: Medium (quality degrades)

**Mitigation**:
- Batch size: 50 tasks per annotator per week (max)
- Rotate difficult/easy tasks to maintain engagement
- Gamification: leaderboard for high-quality annotations (optional)

### Risk 4: Label Studio Technical Issues
**Likelihood**: Low  
**Impact**: High (blocks annotation)

**Mitigation**:
- Local installation (no SaaS dependencies)
- Telemetry disabled (privacy + reliability)
- Backup exports every 2 days to prevent data loss

---

## Dependencies & Requirements

### Technical
- **Label Studio**: v1.7.0+ (local installation, telemetry disabled)
- **Python**: 3.9+ with existing SpanForge environment
- **LLM Access**: OpenAI API key or Azure/Anthropic credentials
- **Storage**: ~500 MB for 100-task batches (text + annotations + reports)

### Human Resources
- **Annotators**: 2-3 domain experts (biomedical/clinical background preferred)
- **Annotation Time**: ~2-3 minutes per task (100 tasks = 3-5 hours per annotator)
- **Review Time**: 1 hour per batch for adjudication and quality checks

### Budget (for 100 tasks)
- **LLM Refinement**: $0.48-$7.20 depending on provider (GPT-4o-mini vs GPT-4)
- **Annotator Time**: ~10 hours total @ $30/hr = $300 (if contracted)
- **Infrastructure**: $0 (local Label Studio, free tier LLM limits sufficient for pilot)

**Total Estimated Cost**: $300-$350 per 100-task batch

---

## Success Criteria

### Functional Requirements
- ✅ Label Studio imports LLM-refined weak labels as pre-annotations
- ✅ Human annotators can correct boundaries and labels
- ✅ Export → conversion → gold JSONL pipeline works end-to-end
- ✅ Evaluation harness runs without errors on gold data

### Quality Metrics
- **IOU Improvement**: +10-15% from weak → LLM (measured by evaluation harness)
- **Correction Rate**: <10% worsened, >60% improved
- **Inter-Annotator Agreement**: IOU ≥0.5 agreement >0.75 for calibration set
- **Exact Match Rate**: ≥70% after LLM refinement (vs gold)

### Documentation
- ✅ Tutorial notebook tested with 3+ annotators
- ✅ Annotation guide reviewed and approved
- ✅ Workflow orchestration script documented with examples

---

## Next Steps (Immediate)

### Option A1: Quick Win - Label Config + Tutorial
**Time**: 2-3 hours  
**Tasks**:
1. Create `data/annotation/config/label_config.xml`
2. Draft `scripts/AnnotationWalkthrough.ipynb` (basic version)
3. Test import/export round-trip with 5 synthetic examples

### Option A2: Production Batch Prep
**Time**: 4-6 hours  
**Tasks**:
1. Implement `prepare_production_batch.py` with stratified sampling
2. De-identify 100 real adverse event reports
3. Generate weak labels + LLM refinement for batch

### Option A3: Expand Annotation Guidelines
**Time**: 2-3 hours  
**Tasks**:
1. Add 10+ examples to `docs/annotation_guide.md` (correct vs incorrect)
2. Create glossary of symptom synonyms
3. Document negation policy with examples

---

## References

- **Existing Scripts**: `scripts/annotation/` (10 scripts already functional)
- **Evaluation Harness**: `docs/llm_evaluation.md`
- **Phase 4.5 Summary**: `docs/phase_4.5_summary.md`
- **Copilot Instructions**: `.github/copilot-instructions.md`

---

**Status**: 🟡 PLANNING COMPLETE - Ready for implementation  
**Next Phase**: 6 - Gold Standard Assembly & Token Classification Fine-Tuning  
**Last Updated**: November 25, 2025
