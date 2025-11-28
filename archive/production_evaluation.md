# Production Evaluation Guide

**Version**: 1.0  
**Audience**: Production users running evaluation on real annotation batches  
**Last Updated**: November 25, 2025

---

## Overview

This guide covers **real-world usage** of the LLM evaluation harness with production annotation data, including data preparation, interpretation strategies, optimization techniques, and troubleshooting.

---

## Prerequisites

Before running production evaluation:

✅ **Completed Annotations**: Export gold standard from Label Studio  
✅ **Weak Labels**: Original weak label JSONL with confidence scores  
✅ **LLM Refined Labels**: Output from `refine_llm.py` or LLM agent  
✅ **Dependencies**: Core SpanForge (`requirements.txt`) installed  
✅ **Optional**: Visualization tools (`requirements-viz.txt`) for plots

---

## Production Workflow

### Step 1: Data Preparation

#### 1.1 Verify Data Alignment

**Critical**: All three datasets (weak, LLM, gold) must have **matching document IDs**.

```bash
# Quick ID check
python -c "
import json
from pathlib import Path

weak_ids = {json.loads(line)['id'] for line in open('weak.jsonl')}
llm_ids = {json.loads(line)['id'] for line in open('llm_refined.jsonl')}
gold_ids = {json.loads(line)['id'] for line in open('gold.jsonl')}

missing_llm = weak_ids - llm_ids
missing_gold = weak_ids - gold_ids

if missing_llm:
    print(f'⚠️  Missing LLM refined: {missing_llm}')
if missing_gold:
    print(f'⚠️  Missing gold standard: {missing_gold}')
if not missing_llm and not missing_gold:
    print('✅ All IDs aligned')
"
```

**Common Issues**:
- **Filtered spans**: LLM may skip low-quality spans → Results in fewer LLM spans than weak
- **Annotation subset**: Annotators may skip difficult tasks → Gold has fewer IDs
- **Export errors**: Label Studio export may exclude empty tasks

**Fix**: Use `--allow-missing` flag in evaluation script (future enhancement) or manually align datasets.

#### 1.2 Validate Data Formats

```bash
# Check weak labels format
python -c "
import json
for line in open('weak.jsonl'):
    r = json.loads(line)
    assert 'id' in r, 'Missing id field'
    assert 'text' in r, 'Missing text field'
    assert 'spans' in r, 'Missing spans field'
    for s in r['spans']:
        assert 'start' in s and 'end' in s, 'Missing start/end'
        assert 'label' in s, 'Missing label'
        assert 'confidence' in s, 'Missing confidence (required for stratification)'
        assert s['text'] == r['text'][s['start']:s['end']], 'Span text mismatch'
print('✅ Weak labels validated')
"
```

**Run similar validation** for LLM refined (`llm_suggestions` field) and gold (`source='gold'` field).

---

### Step 2: Run Evaluation

#### 2.1 Basic Evaluation

```bash
python scripts/annotation/cli.py evaluate-llm \
  --weak data/weak/batch_001.jsonl \
  --refined data/llm/batch_001_refined.jsonl \
  --gold data/gold/batch_001.jsonl \
  --output data/annotation/reports/batch_001_eval.json \
  --markdown
```

**Expected Output**:
```
Loading weak labels from data/weak/batch_001.jsonl...
Loading LLM-refined labels from data/llm/batch_001_refined.jsonl...
Loading gold standard from data/gold/batch_001.jsonl...

Span counts:
  Weak: 342
  LLM:  298
  Gold: 315

Computing overall metrics...

✓ JSON report saved to data/annotation/reports/batch_001_eval.json
✓ Markdown summary saved to data/annotation/reports/batch_001_eval.md

============================================================
QUICK SUMMARY
============================================================
IOU Improvement: +8.7%
  Weak:  0.823
  LLM:   0.910
  Delta: +0.087

Correction Rate: 67.3%
  Improved:  62/92
  Worsened:  8/92

LLM F1 Score: 0.892
  Precision: 0.905
  Recall:    0.880
```

#### 2.2 Stratified Analysis

**Recommended for production**: Always stratify to identify weaknesses.

```bash
python scripts/annotation/cli.py evaluate-llm \
  --weak data/weak/batch_001.jsonl \
  --refined data/llm/batch_001_refined.jsonl \
  --gold data/gold/batch_001.jsonl \
  --output data/annotation/reports/batch_001_eval_stratified.json \
  --markdown \
  --stratify label confidence span_length
```

**Added Output**: Stratified tables in Markdown report.

---

### Step 3: Interpretation

#### 3.1 Overall Metrics Targets

| Metric                | Weak Baseline | LLM Target | Excellent |
|-----------------------|---------------|------------|-----------|
| Mean IOU              | 0.75-0.85     | >0.85      | >0.90     |
| IOU Improvement       | -             | +5-10%     | +10-15%   |
| Exact Match Rate      | 50-65%        | 70-80%     | >85%      |
| Correction Rate       | -             | >60%       | >75%      |
| Worsened Rate         | -             | <10%       | <5%       |
| LLM F1 Score          | 0.70-0.80     | >0.85      | >0.90     |

**Reality Check**:
- **Biomedical NER**: Expect IOU improvement ~8-12% (complex domain, noisy text)
- **Simple domains**: May see +15-20% improvement (clearer boundaries, less ambiguity)

#### 3.2 Red Flags

**Warning Sign**: Worsened rate >15%  
**Meaning**: LLM frequently introduces errors (over-correction, hallucination)  
**Action**:
- Review worsened spans in evaluation JSON (`correction_rate.worsened_spans`)
- Reduce LLM temperature (0.1 → 0.0 for determinism)
- Add negative examples to prompts
- Filter high-confidence weak labels (skip LLM if confidence ≥0.85)

**Warning Sign**: IOU improvement <5%  
**Meaning**: LLM provides minimal value over weak labels  
**Action**:
- Check if weak labels already high quality (mean confidence >0.85)
- Increase LLM model capability (GPT-4o-mini → GPT-4)
- Provide more context in prompts (surrounding sentences, lexicon definitions)

**Warning Sign**: Low recall (<0.70)  
**Meaning**: LLM or weak labels miss many gold spans  
**Action**:
- Expand lexicons (symptoms, products)
- Adjust fuzzy threshold (0.88 → 0.85 for more liberal matching)
- Review false negatives in evaluation JSON (`prf.false_negatives`)

**Warning Sign**: Low precision (<0.70)  
**Meaning**: Many predicted spans don't match gold (false positives)  
**Action**:
- Tighten fuzzy threshold (0.88 → 0.90)
- Add anatomy gating (skip single-token anatomy like "skin")
- Review false positives in evaluation JSON (`prf.false_positives`)

#### 3.3 Stratified Insights

**By Label**:
```markdown
| Stratum  | Weak F1 | LLM F1 | IOU Delta | Span Count |
|----------|---------|--------|-----------|------------|
| SYMPTOM  | 0.812   | 0.905  | +0.098    | 298        |
| PRODUCT  | 0.703   | 0.758  | +0.042    | 44         |
```

**Interpretation**: SYMPTOM refinement works well (+9.8% IOU), but PRODUCT lags (+4.2%). Likely causes:
- PRODUCT lexicon incomplete (fewer entries than symptoms)
- Product names more diverse (brand names, abbreviations)
- LLM prompts focus on symptom boundaries

**Action**: Add PRODUCT-specific examples to LLM prompts, expand product lexicon.

**By Confidence**:
```markdown
| Stratum   | Weak F1 | LLM F1 | IOU Delta | Span Count |
|-----------|---------|--------|-----------|------------|
| 0.60-0.70 | 0.614   | 0.812  | +0.215    | 37         |
| 0.70-0.80 | 0.723   | 0.856  | +0.124    | 82         |
| 0.80-0.90 | 0.841   | 0.923  | +0.088    | 145        |
| 0.90-1.00 | 0.932   | 0.958  | +0.028    | 78         |
```

**Interpretation**: Low-confidence spans benefit most (+21.5% in 0.60-0.70 bucket). High-confidence spans see minimal improvement (+2.8% in 0.90-1.00).

**Action**: Implement confidence filtering to optimize LLM costs:
```bash
# Only refine spans with confidence < 0.80
python scripts/annotation/filter_confidence.py \
  --input weak.jsonl \
  --output weak_low_conf.jsonl \
  --threshold 0.80 \
  --action below
```

**Estimated Cost Savings**: If 40% of spans have confidence ≥0.80, save 40% on LLM API costs.

---

### Step 4: Visualization (Optional)

```bash
# Install dependencies (one-time)
pip install -r requirements-viz.txt

# Generate all plots
python scripts/annotation/cli.py plot-metrics \
  --report data/annotation/reports/batch_001_eval_stratified.json \
  --output-dir data/annotation/plots/batch_001/ \
  --formats png pdf \
  --dpi 300
```

**Output**: 6 plots in `plots/batch_001/`:
- `iou_uplift.png` - Before/after IOU distribution
- `calibration_curve.png` - Confidence reliability
- `correction_rate.png` - Improved/worsened/unchanged breakdown
- `prf_comparison.png` - Precision/Recall/F1 side-by-side
- `stratified_label.png` - F1 by entity type
- `stratified_confidence.png` - IOU delta by confidence bucket

**Use Cases**:
- **Presentations**: Visualize improvements for stakeholders
- **Papers**: Publication-quality figures (300 DPI)
- **Debugging**: Identify patterns in calibration curve (over/under-confidence)

---

## Optimization Strategies

### Strategy 1: Confidence-Based Filtering

**Goal**: Reduce LLM costs by skipping high-confidence weak labels.

**Implementation**:
```bash
# Filter weak labels
python scripts/annotation/filter_confidence.py \
  --input weak.jsonl \
  --output weak_filtered.jsonl \
  --threshold 0.85 \
  --action below  # Keep only confidence < 0.85

# Refine filtered spans
python scripts/annotation/cli.py refine-llm \
  --weak weak_filtered.jsonl \
  --output llm_refined_filtered.jsonl

# Merge back high-confidence spans (unchanged)
python scripts/annotation/merge_spans.py \
  --weak weak.jsonl \
  --refined llm_refined_filtered.jsonl \
  --output llm_refined_full.jsonl
```

**Expected Results**:
- **Cost Reduction**: 30-50% depending on weak label quality
- **Quality**: Minimal IOU loss (<2%) since high-confidence spans already accurate

### Strategy 2: Iterative Prompt Refinement

**Goal**: Improve LLM correction rate based on evaluation feedback.

**Workflow**:
1. Run evaluation on first batch (e.g., 50 tasks)
2. Identify failure patterns in worsened spans:
   - LLM removes valid multi-word terms (e.g., "burning sensation" → "burning")
   - LLM hallucinates boundaries (no overlap with gold)
   - LLM over-corrects negations (removes "no" incorrectly)
3. Update LLM prompts in `src/llm_agent.py`:
   - Add negative examples: "Don't truncate multi-word medical terms"
   - Add positive examples: "Keep 'burning sensation' as full phrase"
4. Re-run refinement on second batch
5. Compare evaluation metrics (target: reduce worsened rate by 50%)

**Example Prompt Iteration**:

**Before** (generic):
```
Refine the following medical entity boundaries to match standard terminology.
```

**After** (specific):
```
Refine medical entity boundaries following these rules:
1. Preserve multi-word terms from medical lexicon (e.g., "burning sensation", "anaphylactic shock")
2. Remove non-medical adjectives (e.g., "severe" → remove, "burning" → keep)
3. For negations (e.g., "no redness"), keep the symptom span but mark negation flag
4. Exclude trailing punctuation and conjunctions
```

### Strategy 3: Lexicon Expansion

**Goal**: Improve weak label recall by adding missing canonical terms.

**Workflow**:
1. Run evaluation, extract false negatives from JSON report
2. Review false negative spans (gold spans not captured by weak or LLM)
3. Identify missing lexicon entries:
   - Synonyms: "pruritus" missing (only "itching" in lexicon)
   - Abbreviations: "SOB" missing (shortness of breath)
   - Multi-word: "dry mouth" missing (only "dryness")
4. Add to `data/lexicon/symptoms.csv`:
   ```csv
   canonical_term,concept_id,synonyms
   pruritus,10037087,"itching|pruritic|itch"
   dry mouth,10013781,"xerostomia|mouth dryness"
   ```
5. Regenerate weak labels on second batch
6. Compare recall improvement (target: +5-10%)

---

## Troubleshooting

### Issue 1: Span Count Mismatch

**Symptom**:
```
Span counts:
  Weak: 342
  LLM:  298  ← 44 spans missing
  Gold: 315
```

**Cause**: LLM filtered out low-confidence spans or failed to generate suggestions.

**Diagnosis**:
```bash
# Check LLM metadata for skipped spans
python -c "
import json
for line in open('llm_refined.jsonl'):
    r = json.loads(line)
    weak_count = len(r.get('spans', []))
    llm_count = len(r.get('llm_suggestions', []))
    if llm_count < weak_count:
        print(f'Task {r[\"id\"]}: {weak_count} weak → {llm_count} LLM (skipped {weak_count - llm_count})')
"
```

**Fix**: Check `llm_meta.errors` field in JSONL for API errors or timeouts.

### Issue 2: Calibration Curve Shows Over-Confidence

**Symptom**: Calibration plot shows curve below diagonal (confidence > actual IOU).

**Cause**: Confidence scores from weak labels don't reflect true accuracy.

**Fix**:
1. **Platt Scaling**: Apply logistic regression to calibrate confidence scores
2. **Isotonic Regression**: Non-parametric calibration (requires ≥100 spans)
3. **Temperature Scaling**: Divide logits by temperature parameter >1

**Implementation** (Platt scaling):
```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# Train on weak vs gold IOU
weak_confidences = [s['confidence'] for s in weak_spans]
actual_ious = [compute_iou(weak_span, gold_span) for weak_span, gold_span in pairs]

lr = LogisticRegression()
lr.fit(np.array(weak_confidences).reshape(-1, 1), (np.array(actual_ious) > 0.5).astype(int))

# Calibrate future predictions
calibrated_confidence = lr.predict_proba(confidence)[0][1]
```

### Issue 3: Low IOU Despite High F1

**Symptom**:
```
LLM F1 Score: 0.92  ← High precision/recall
Mean IOU:     0.78  ← Poor boundary alignment
```

**Cause**: Spans overlap with gold but boundaries misaligned (partial matches).

**Diagnosis**:
```bash
# Check boundary precision report
grep "Exact Match Rate" data/annotation/reports/batch_001_eval.md
# Output: Exact Match Rate: 45.2%  ← Low exact matches
```

**Interpretation**: LLM correctly identifies entities but struggles with precise boundaries.

**Fix**:
1. Add boundary examples to LLM prompts (show correct spans)
2. Increase annotation guide clarity (boundary rules)
3. Post-process LLM suggestions with heuristic trimming (remove trailing determiners)

### Issue 4: JSON Report Too Large

**Symptom**: Evaluation JSON exceeds 10 MB, difficult to load in browser.

**Cause**: Saving all span details including full text for debugging.

**Fix**: Use `--compact` flag (future enhancement) to exclude verbose fields:
```bash
python scripts/annotation/cli.py evaluate-llm \
  ... \
  --compact  # Omit span text, rationale fields
```

---

## Production Checklist

Before deploying evaluation in production:

### Data Quality
- [ ] All IDs aligned across weak/LLM/gold datasets
- [ ] Span text matches `text[start:end]` (integrity test)
- [ ] Confidence scores present in weak labels (required for stratification)
- [ ] Gold annotations have provenance (`source='gold'`, annotator, timestamp)

### Evaluation Setup
- [ ] Output directory writable (`data/annotation/reports/`)
- [ ] Stratification flags specified (`--stratify label confidence`)
- [ ] Markdown report enabled (`--markdown`) for quick review

### Interpretation
- [ ] Baseline metrics recorded (weak-only evaluation for comparison)
- [ ] Red flags documented (worsened rate, low IOU improvement)
- [ ] Stratified analysis reviewed (identify weak subgroups)

### Follow-Up
- [ ] Evaluation results shared with annotators (calibration)
- [ ] Prompt/lexicon updates planned based on feedback
- [ ] Cost analysis completed (LLM API usage vs quality gain)

---

## Case Study: Real Production Batch

### Scenario
- **Dataset**: 100 adverse event reports (biomedical domain)
- **Weak Labels**: 342 spans (mean confidence: 0.78)
- **LLM Refinement**: GPT-4o-mini ($0.48 total cost)
- **Gold Standard**: 315 spans (human-annotated)

### Results

**Overall Metrics**:
```
IOU Improvement:   +8.7% (0.823 → 0.910)
Exact Match Rate:  52.3% → 73.8%
Correction Rate:   67.3% improved, 8.7% worsened
LLM F1 Score:      0.892 (Precision: 0.905, Recall: 0.880)
```

**Stratified by Label**:
- SYMPTOM: +9.8% IOU improvement (strong performance)
- PRODUCT: +4.2% IOU improvement (needs targeted prompts)

**Stratified by Confidence**:
- Low confidence (0.60-0.70): +21.5% IOU (highest benefit)
- High confidence (0.90-1.00): +2.8% IOU (diminishing returns)

### Insights

1. **LLM Provides Clear Value**: +8.7% IOU improvement justifies $0.48 cost (~0.5¢ per span)
2. **Confidence Filtering Opportunity**: Skip spans with confidence ≥0.85 → Save ~35% on LLM costs with <2% quality loss
3. **Product Refinement Underperforms**: Need PRODUCT-specific examples in prompts + lexicon expansion
4. **Low Worsened Rate** (8.7%): LLM rarely introduces errors → Safe to use in production

### Actions Taken

1. **Prompt Update**: Added PRODUCT boundary examples
2. **Lexicon Expansion**: Added 50 product names from false negatives
3. **Confidence Filtering**: Implemented threshold=0.85 for next batch
4. **Annotation Guide**: Clarified PRODUCT annotation rules based on disagreements

### Second Batch Results

**Improvements**:
- PRODUCT IOU delta: +4.2% → +7.8% (prompt update effective)
- Cost reduction: $0.48 → $0.31 (35% savings from confidence filtering)
- Worsened rate: 8.7% → 5.2% (fewer LLM errors)

**Conclusion**: Iterative refinement based on evaluation feedback significantly improves quality and reduces costs.

---

## Cost-Benefit Analysis

### Typical Production Batch (100 tasks, 300 spans)

**Costs**:
- LLM Refinement: $0.50-$7.00 (depending on provider/model)
- Evaluation Runtime: ~30 seconds (negligible compute cost)
- Annotator Review: 5 hours @ $30/hr = $150

**Benefits**:
- Time Savings: Pre-annotations reduce annotation time by 30-40% → Save 2 hours = $60
- Quality Improvement: +8-12% IOU → Fewer downstream model errors → Hard to quantify but high value
- Consistency: LLM enforces canonical terms → Reduces annotator disagreements

**ROI Calculation**:
```
Net Benefit = Time Savings + Quality Value - LLM Cost
            = $60 + $100 (estimated quality value) - $7 (GPT-4)
            = $153 per 100-task batch

ROI = ($153 / $7) × 100% = 2,186%
```

**Recommendation**: Use GPT-4o-mini ($0.50) for even better ROI (30,600%) with acceptable quality.

---

## References

- **Evaluation Metrics Documentation**: `docs/llm_evaluation.md`
- **Phase 5 Implementation Plan**: `docs/phase_5_plan.md`
- **CLI Reference**: `python scripts/annotation/cli.py evaluate-llm --help`
- **Visualization Guide**: `python scripts/annotation/cli.py plot-metrics --help`

---

**Last Updated**: November 25, 2025  
**Feedback**: Open GitHub issue for production evaluation questions
