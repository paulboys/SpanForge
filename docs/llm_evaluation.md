# LLM Refinement Evaluation Guide

**Version**: 1.0  
**Last Updated**: November 25, 2025  
**Phase**: 4.5 - LLM-Based Refinement

---

## Overview

The **LLM Refinement Evaluation Harness** provides comprehensive metrics for measuring the quality improvement from weak labels → LLM-refined labels → gold standard annotations. This system helps you:

- **Quantify improvement**: Track IOU uplift, boundary precision gains, and correction rates
- **Identify failure modes**: Find where LLM over-corrects or misses edge cases
- **Optimize prompts**: A/B test different LLM strategies and measure impact
- **Build confidence**: Validate that LLM refinement improves downstream annotation quality

---

## Quick Start

### 1. Generate Evaluation Report

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
- `evaluation.json` - Full metrics in JSON format
- `evaluation.md` - Human-readable summary tables

### 2. Visualize Results (Optional)

```bash
# Install visualization dependencies first
pip install -r requirements-viz.txt

# Generate plots
python scripts/annotation/plot_llm_metrics.py \
  --report data/annotation/reports/evaluation.json \
  --output-dir data/annotation/plots/ \
  --formats png pdf \
  --dpi 300
```

**Output**: 6 publication-quality plots showing IOU uplift, calibration curves, correction breakdown, P/R/F1 comparison, and stratified analysis.

---

## Input Data Formats

### Weak Labels (`weak_labels.jsonl`)

```json
{
  "id": "complaint_001",
  "text": "Patient reports severe burning sensation after using Product X.",
  "spans": [
    {
      "start": 15,
      "end": 43,
      "text": "severe burning sensation",
      "label": "SYMPTOM",
      "confidence": 0.85,
      "source": "weak"
    }
  ]
}
```

**Required Fields**:
- `id` (str): Unique document identifier
- `text` (str): Raw input text
- `spans` (list): Array of span objects
  - `start` (int): Character offset (0-indexed)
  - `end` (int): Character offset (exclusive)
  - `text` (str): Extracted substring (must match `text[start:end]`)
  - `label` (str): Entity type (SYMPTOM, PRODUCT, etc.)
  - `confidence` (float): Weak labeling confidence score (0-1)
  - `source` (str): Must be `"weak"`

### LLM Refined Labels (`llm_refined.jsonl`)

```json
{
  "id": "complaint_001",
  "text": "Patient reports severe burning sensation after using Product X.",
  "spans": [
    {
      "start": 15,
      "end": 43,
      "text": "severe burning sensation",
      "label": "SYMPTOM",
      "confidence": 0.85,
      "source": "weak"
    }
  ],
  "llm_suggestions": [
    {
      "start": 22,
      "end": 40,
      "text": "burning sensation",
      "label": "SYMPTOM",
      "confidence": 0.95,
      "canonical": "burning sensation",
      "rationale": "Removed superfluous adjective 'severe' - medical lexicon uses canonical term"
    }
  ],
  "llm_meta": {
    "provider": "openai",
    "model": "gpt-4",
    "timestamp": "2025-11-25T10:30:00Z",
    "token_usage": {"input": 85, "output": 42}
  }
}
```

**Additional Fields**:
- `llm_suggestions` (list): LLM-proposed span corrections
  - Same structure as `spans` but with added `canonical` and `rationale`
- `llm_meta` (dict): Provenance metadata (provider, model, timestamp, token usage)

### Gold Standard Labels (`gold_standard.jsonl`)

```json
{
  "id": "complaint_001",
  "text": "Patient reports severe burning sensation after using Product X.",
  "spans": [
    {
      "start": 22,
      "end": 40,
      "text": "burning sensation",
      "label": "SYMPTOM",
      "source": "gold",
      "annotator": "annotator_01",
      "timestamp": "2025-11-25T14:00:00Z"
    },
    {
      "start": 57,
      "end": 66,
      "text": "Product X",
      "label": "PRODUCT",
      "source": "gold",
      "annotator": "annotator_01",
      "timestamp": "2025-11-25T14:00:00Z"
    }
  ]
}
```

**Gold-Specific Fields**:
- `source` (str): Must be `"gold"`
- `annotator` (str): Human annotator identifier (for IAA tracking)
- `timestamp` (str): ISO 8601 annotation time

---

## Evaluation Metrics

### 1. IOU (Intersection over Union)

**Formula**: `IOU = overlap / union`

**Interpretation**:
- `1.0` - Perfect match (exact boundaries)
- `0.8-0.99` - Partial overlap (minor boundary differences)
- `0.5-0.79` - Moderate overlap (significant boundary mismatch)
- `<0.5` - Poor overlap (different spans entirely)

**Use Case**: Measure boundary precision improvements from weak → LLM.

### 2. IOU Delta

**Formula**: `Δ = mean(IOU_llm) - mean(IOU_weak)`

**Interpretation**:
- `+0.10` or higher - Strong improvement (LLM significantly corrects boundaries)
- `+0.05 to +0.09` - Moderate improvement
- `-0.05 to +0.04` - Negligible change
- `<-0.05` - Regression (LLM worsens boundaries)

**Use Case**: Headline metric for LLM refinement quality. Track across iterations.

### 3. Boundary Precision

**Metrics**:
- **Exact Match Rate**: % of predictions with IOU = 1.0
- **Mean IOU**: Average IOU across all predictions
- **Median IOU**: Robust central tendency (less sensitive to outliers)

**Use Case**: Understand distribution of boundary quality. High mean + low exact match = consistent partial overlaps.

### 4. Correction Rate

**Categories**:
- **Improved**: LLM span has higher IOU than weak span (desired outcome)
- **Worsened**: LLM span has lower IOU than weak span (failure mode)
- **Unchanged**: LLM kept weak span as-is (confidence in original)

**Formula**: `Correction Rate = Improved / (Improved + Worsened + Unchanged)`

**Use Case**: Track LLM decision quality. High "worsened" % indicates over-correction or hallucination.

### 5. Calibration Curve

**Definition**: Plots predicted confidence vs. observed accuracy (IOU).

**Perfect Calibration**: Diagonal line where `confidence = IOU`.

**Under-Confident**: Curve above diagonal (confidence < actual IOU).

**Over-Confident**: Curve below diagonal (confidence > actual IOU).

**Use Case**: Validate confidence scores for active learning or filtering low-confidence spans.

### 6. Precision / Recall / F1

**Standard NER Metrics**:
- **Precision**: % of predicted spans that match gold (exact or IOU ≥ threshold)
- **Recall**: % of gold spans captured by predictions
- **F1**: Harmonic mean of precision and recall

**Use Case**: Compare LLM refinement to baseline weak labels using standard benchmarks.

---

## Stratified Analysis

### By Label Type

**Purpose**: Identify if LLM improves certain entity types more than others.

**Example**:
- SYMPTOM: +15% IOU delta (strong improvement)
- PRODUCT: +2% IOU delta (minimal improvement)

**Action**: Investigate why PRODUCT refinement underperforms. Adjust prompts or lexicons.

### By Confidence Bucket

**Purpose**: Validate that low-confidence weak labels benefit more from LLM refinement.

**Example**:
- 0.60-0.70: +20% IOU delta (high uncertainty → high benefit)
- 0.90-1.00: +3% IOU delta (already accurate)

**Action**: Prioritize LLM refinement for confidence < 0.80 to optimize cost.

### By Span Length

**Purpose**: Detect if LLM struggles with long multi-word spans.

**Example**:
- 1-10 chars: +18% IOU delta (single-word corrections work well)
- 20-40 chars: +5% IOU delta (complex phrases harder to refine)

**Action**: Add context windows or examples for long spans in LLM prompts.

---

## Interpreting Reports

### Example Markdown Output

```markdown
# LLM Refinement Evaluation Report

## Overall Performance

### IOU Improvement
- **Weak Labels Mean IOU**: 0.882
- **LLM Refined Mean IOU**: 1.000
- **Delta**: +0.118
- **Improvement**: +13.4%

### Correction Statistics
- **Total Spans**: 6
- **Modified by LLM**: 2 (33.3%)
- **Improved**: 2 (100.0% of modified)
- **Worsened**: 0 (0.0% of modified)
- **Unchanged**: 4

## Stratified Analysis

### By Label
| Stratum  | Weak F1 | LLM F1 | IOU Delta | Span Count |
|----------|---------|--------|-----------|------------|
| SYMPTOM  | 1.000   | 1.000  | +0.142    | 5          |
| PRODUCT  | 1.000   | 1.000  | +0.000    | 1          |
```

**Key Takeaways**:
1. **Strong Overall Improvement**: +13.4% IOU gain indicates LLM effectively corrects boundaries.
2. **No Regressions**: 0 worsened spans shows LLM doesn't introduce errors.
3. **Selective Refinement**: Only 33.3% modified suggests LLM preserves high-quality weak labels.
4. **Label Imbalance**: SYMPTOM benefits more than PRODUCT (potential for specialized prompts).

---

## Common Patterns & Actions

### Pattern 1: High IOU Delta, Low Exact Match Rate
**Meaning**: LLM improves boundaries but doesn't achieve perfect alignment.

**Action**: Review gold annotation guidelines. May indicate valid alternative boundaries (e.g., "burning sensation" vs. "severe burning sensation" both medically correct).

### Pattern 2: High "Worsened" Correction Rate
**Meaning**: LLM over-corrects or hallucinates spans.

**Action**: 
- Reduce LLM temperature (increase determinism)
- Add negative examples to prompts
- Filter LLM suggestions by confidence threshold before annotation

### Pattern 3: Calibration Curve Below Diagonal
**Meaning**: LLM overestimates confidence (over-confident predictions).

**Action**: Apply Platt scaling or isotonic regression to calibrate confidence scores.

### Pattern 4: Low IOU Delta in Low-Confidence Buckets
**Meaning**: LLM doesn't improve uncertain weak labels (expected high benefit).

**Action**: Provide more context in LLM prompts (e.g., surrounding sentences, lexicon definitions).

---

## Advanced Usage

### Custom Stratification

```python
from src.evaluation.metrics import stratify_by_span_length

# Custom span length buckets
buckets = [(0, 10), (10, 20), (20, 50), (50, 100)]
stratified = stratify_by_span_length(spans, buckets)
```

### Programmatic Access

```python
import json
from src.evaluation.metrics import compute_iou_delta, compute_correction_rate

# Load data
with open('weak_labels.jsonl') as f:
    weak_data = [json.loads(line) for line in f]

# Extract spans
weak_spans = [s for d in weak_data for s in d['spans']]
llm_spans = [s for d in llm_data for s in d.get('llm_suggestions', [])]
gold_spans = [s for d in gold_data for s in d['spans']]

# Compute metrics
iou_delta = compute_iou_delta(weak_spans, llm_spans, gold_spans)
correction_rate = compute_correction_rate(weak_spans, llm_spans, gold_spans)

print(f"IOU Improvement: {iou_delta['improvement_pct']:.1f}%")
print(f"Correction Rate: {correction_rate['improved_pct']:.1f}%")
```

---

## Visualization Gallery

### 1. IOU Uplift Histogram
Shows distribution of IOU scores before (weak) and after (LLM) refinement. Shift toward 1.0 indicates improvement.

### 2. Calibration Curve
Diagonal = perfect calibration. Points above diagonal = under-confident. Points below = over-confident.

### 3. Correction Rate Breakdown
Pie chart showing % improved/worsened/unchanged. Healthy distribution: >80% improved, <5% worsened.

### 4. P/R/F1 Comparison
Side-by-side bars for weak vs LLM. Delta annotations show improvement magnitude.

### 5. Stratified Label Analysis
Grouped bars showing F1 by entity type. Identifies labels needing targeted refinement.

### 6. Stratified Confidence Analysis
Bar chart of IOU delta across confidence buckets. Validates that low-confidence spans benefit most.

---

## Troubleshooting

### Issue: "No module named 'matplotlib'"
**Solution**: Install visualization dependencies:
```bash
pip install -r requirements-viz.txt
```

### Issue: Empty calibration curve
**Cause**: Fewer than 10 spans with confidence scores.

**Solution**: Increase dataset size or reduce `n_bins` parameter in `calibration_curve()`.

### Issue: All spans show "unchanged" in correction rate
**Cause**: LLM suggestions not properly loaded or `llm_suggestions` field missing.

**Solution**: Verify `llm_refined.jsonl` contains `llm_suggestions` array (see format above).

### Issue: Stratified analysis shows single bucket
**Cause**: All spans have similar confidence/length/label.

**Solution**: Increase data diversity or adjust bucket boundaries for finer granularity.

---

## Integration with Annotation Workflow

### Recommended Sequence

1. **Generate Weak Labels**  
   ```bash
   python -m src.pipeline --input raw.txt --output weak.jsonl
   ```

2. **Refine with LLM**  
   ```bash
   python -m src.llm_agent --weak weak.jsonl --output llm_refined.jsonl
   ```

3. **Import to Label Studio** (when implemented)  
   ```bash
   python scripts/annotation/import_weak_to_labelstudio.py llm_refined.jsonl
   ```

4. **Human Annotation**  
   Annotators correct LLM suggestions in Label Studio.

5. **Export Gold Standard**  
   ```bash
   python scripts/annotation/export_from_labelstudio.py --output gold.jsonl
   ```

6. **Evaluate**  
   ```bash
   python scripts/annotation/evaluate_llm_refinement.py \
     --weak weak.jsonl --refined llm_refined.jsonl --gold gold.jsonl \
     --output reports/eval.json --markdown --stratify label confidence
   ```

7. **Visualize**  
   ```bash
   python scripts/annotation/plot_llm_metrics.py \
     --report reports/eval.json --output-dir plots/ --formats png pdf
   ```

8. **Iterate**  
   Adjust LLM prompts, confidence thresholds, or weak labeling heuristics based on evaluation results.

---

## Performance Benchmarks

### Test Fixtures (Synthetic Data)
- **Dataset**: 3 complaints, 6 spans (5 SYMPTOM, 1 PRODUCT)
- **IOU Improvement**: +13.4% (0.882 → 1.000)
- **Exact Match Rate**: 66.7% → 100.0%
- **Correction Rate**: 100% improved (2/2 modified spans)
- **F1 Score**: 1.000 (perfect precision/recall)

### Production Expectations (Real Data)
- **IOU Improvement**: +8-15% typical for biomedical NER
- **Exact Match Rate**: 70-85% after LLM refinement
- **Correction Rate**: 60-80% improved, <10% worsened
- **F1 Score**: 0.85-0.95 (depending on domain complexity)

---

## Cost Estimation

### LLM API Costs (per 1,000 spans)

| Provider       | Model                  | Input Cost | Output Cost | Total Est. |
|----------------|------------------------|------------|-------------|------------|
| OpenAI         | GPT-4                  | $2.40      | $4.80       | **$7.20**  |
| OpenAI         | GPT-4o-mini            | $0.12      | $0.36       | **$0.48**  |
| Anthropic      | Claude 3.5 Sonnet      | $0.24      | $1.20       | **$1.44**  |
| Azure OpenAI   | GPT-4 (deployment)     | $2.40      | $4.80       | **$7.20**  |

**Assumptions**: 80 input tokens/span (context + weak label), 40 output tokens/span (refinement + rationale).

**Optimization Tips**:
- Use GPT-4o-mini for initial experiments (10x cheaper)
- Batch requests to reduce overhead
- Filter spans by confidence < 0.8 before LLM refinement (skip high-confidence labels)
- Cache LLM responses for repeated spans

---

## References

- **Evaluation Metrics**: `src/evaluation/metrics.py`
- **Evaluation Script**: `scripts/annotation/evaluate_llm_refinement.py`
- **Visualization Script**: `scripts/annotation/plot_llm_metrics.py`
- **Test Suite**: `tests/test_evaluate_llm.py` (27 tests)
- **LLM Agent**: `src/llm_agent.py`
- **Provider Docs**: `docs/llm_providers.md`

---

**Questions?** See `docs/annotation_guide.md` for label definitions or open an issue on GitHub.
