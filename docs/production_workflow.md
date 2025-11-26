# Production Evaluation Workflow Guide

**Version**: 1.0  
**Purpose**: Step-by-step instructions for running evaluation on real annotation batches  
**Audience**: Annotators, project managers, NER practitioners

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Workflow Steps](#workflow-steps)
4. [Data Validation](#data-validation)
5. [CLI Execution Examples](#cli-execution-examples)
6. [Result Interpretation](#result-interpretation)
7. [Iteration Strategy](#iteration-strategy)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This guide walks through the complete production evaluation workflow for measuring annotation quality on real biomedical complaints data.

### Workflow Phases

```
Phase 1: Batch Preparation
   ↓
Phase 2: Weak Label Generation
   ↓
Phase 3: LLM Refinement (Optional)
   ↓
Phase 4: Human Annotation (Label Studio)
   ↓
Phase 5: Export & Conversion
   ↓
Phase 6: Evaluation & Analysis
   ↓
Phase 7: Iteration & Improvement
```

### Key Metrics

- **IOU Improvement**: +8-15% target (weak → LLM vs gold)
- **Exact Match Rate**: 70-85% target (LLM boundaries align with gold)
- **Correction Rate**: >60% improved, <10% worsened
- **F1 Score**: >0.85 target (LLM precision/recall vs gold)

---

## Prerequisites

### Environment Setup

1. **Python Environment**:
   ```bash
   conda activate NER
   pip install -r requirements.txt
   pip install -r requirements-llm.txt
   pip install -r requirements-viz.txt  # Optional for plots
   ```

2. **Label Studio**:
   ```bash
   pip install label-studio
   
   # Disable telemetry (PowerShell)
   $env:LABEL_STUDIO_DISABLE_TELEMETRY = "1"
   ```

3. **LLM Provider** (if using refinement):
   ```bash
   # OpenAI
   $env:OPENAI_API_KEY = "sk-..."
   
   # Or Azure OpenAI
   $env:AZURE_OPENAI_API_KEY = "..."
   $env:AZURE_OPENAI_ENDPOINT = "https://..."
   $env:AZURE_OPENAI_DEPLOYMENT = "gpt-4"
   
   # Or Anthropic
   $env:ANTHROPIC_API_KEY = "sk-ant-..."
   ```

### Data Requirements

- **Raw Complaints**: De-identified text files (UTF-8 encoding)
- **Lexicons**: `data/lexicon/symptoms.csv`, `data/lexicon/products.csv`
- **Config**: `data/annotation/config/label_config.xml`

---

## Workflow Steps

### Step 1: Prepare Production Batch

**Goal**: Select 100 representative complaints for annotation

**Command**:
```bash
python scripts/annotation/prepare_production_batch.py \
  --input data/raw/complaints_pool.txt \
  --output data/annotation/batches/batch_001/ \
  --batch-size 100 \
  --stratify confidence \
  --confidence-bins 0.5,0.7,0.85 \
  --deidentify
```

**Expected Output**:
```
data/annotation/batches/batch_001/
├── manifest.json          # Batch metadata (source, timestamp, counts)
├── tasks.json             # Label Studio import format
├── weak_labels.jsonl      # Weak labels only (baseline)
├── llm_refined.jsonl      # LLM-refined labels (optional)
└── texts.txt              # De-identified complaint texts
```

**Manifest Example**:
```json
{
  "batch_id": "batch_001",
  "created_at": "2025-11-25T10:30:00Z",
  "source": "complaints_pool.txt",
  "total_tasks": 100,
  "stratification": {
    "low_confidence": 30,    // confidence < 0.7
    "medium_confidence": 50, // 0.7 ≤ confidence < 0.85
    "high_confidence": 20    // confidence ≥ 0.85
  },
  "llm_refined": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "estimated_cost": "$3.25"
}
```

**Tips**:
- Use `--stratify confidence` to ensure diverse difficulty levels
- `--deidentify` removes PII (names, dates, locations) automatically
- `--llm-refine` flag runs LLM refinement during batch prep (saves time)

---

### Step 2: Import to Label Studio

**Manual Steps**:

1. **Launch Label Studio**:
   ```bash
   label-studio start
   ```
   Opens at http://localhost:8080

2. **Create Project**:
   - Name: "Batch 001 - Production NER"
   - Description: "100 complaints from complaints_pool.txt"

3. **Import Config**:
   - Settings → Labeling Interface → Code
   - Copy from `data/annotation/config/label_config.xml`
   - Save

4. **Import Tasks**:
   - Click "Import" button
   - Upload `data/annotation/batches/batch_001/tasks.json`
   - Verify pre-annotations appear (LLM suggestions)

**Verification**:
- Check task count: Should show 100 tasks
- Open first task: Verify SYMPTOM/PRODUCT spans visible
- Test hotkeys: `s` for SYMPTOM, `p` for PRODUCT

---

### Step 3: Annotate

**Guidelines**:
- Follow `docs/annotation_guide.md` for boundary rules
- Use `scripts/AnnotationWalkthrough.ipynb` for practice examples
- Target: 2-3 hours per 100 tasks (experienced annotator)

**Quality Checks** (every 25 tasks):
1. Export current progress
2. Run quick evaluation (see Step 5)
3. Adjust annotation strategy if metrics off-target

**Common Issues**:
- **Boundary Errors**: Review Section 7 of tutorial notebook
- **Negation Confusion**: Annotate symptom only, exclude "no"/"without"
- **Anatomy Tokens**: Skip single anatomy words unless part of symptom phrase

---

### Step 4: Export from Label Studio

**Manual Steps**:

1. **Complete All Tasks**:
   - Verify all 100 tasks submitted (status: "completed")

2. **Export JSON**:
   - Click "Export" button
   - Select "JSON" format
   - Download file (e.g., `project-1-at-2025-11-25-export.json`)

3. **Save to Data Directory**:
   ```powershell
   Move-Item project-1-at-2025-11-25-export.json `
     data/annotation/raw/batch_001_export.json
   ```

---

### Step 5: Convert to Gold Standard

**Command**:
```bash
python scripts/annotation/convert_labelstudio.py \
  --input data/annotation/raw/batch_001_export.json \
  --output data/gold/batch_001.jsonl \
  --source "batch_001" \
  --annotator "your_name" \
  --symptom-lexicon data/lexicon/symptoms.csv \
  --product-lexicon data/lexicon/products.csv
```

**Expected Output**:
```
✅ Converted 100 tasks to gold standard
✅ Total spans: 287 (203 SYMPTOM, 84 PRODUCT)
✅ Output: data/gold/batch_001.jsonl

Quality Report:
  - Average spans per task: 2.87
  - Canonical coverage: 94.2% (191/203 symptoms in lexicon)
  - Boundary integrity: 100.0% (all spans valid)
```

**Validation Checks**:
- Canonical coverage ≥90%: Most symptoms map to lexicon entries
- Boundary integrity 100%: All spans have valid start/end positions
- Average spans 2-4: Reasonable density for complaints

---

### Step 6: Run Evaluation

**Command**:
```bash
python scripts/annotation/cli.py evaluate-llm \
  --weak data/annotation/batches/batch_001/weak_labels.jsonl \
  --refined data/annotation/batches/batch_001/llm_refined.jsonl \
  --gold data/gold/batch_001.jsonl \
  --output data/annotation/reports/batch_001_eval.json \
  --markdown \
  --stratify label confidence span_length
```

**Expected Output**:
```
=== Evaluation Summary ===

IOU Improvement:
  Weak Mean IOU:  0.823
  LLM Mean IOU:   0.901
  Delta:          +0.078 (+9.5% improvement)

Boundary Precision:
  Weak Exact Match:  58.7% (169/288 spans)
  LLM Exact Match:   73.6% (212/288 spans)
  LLM Mean IOU:      0.901

Correction Rate:
  Total Modified:  119 spans
  Improved:        78 spans (65.5%)
  Worsened:        11 spans (9.2%)
  Unchanged:       30 spans (25.2%)

LLM Performance vs Gold:
  Precision:  0.883 (25 FP)
  Recall:     0.912 (22 FN)
  F1 Score:   0.897

Stratified Analysis:
  By Label:
    SYMPTOM: F1=0.902 (N=203)
    PRODUCT: F1=0.881 (N=84)
  
  By Confidence:
    Low (0.0-0.7):    IOU delta=+12.3% (N=87)
    Medium (0.7-0.85): IOU delta=+8.1% (N=144)
    High (0.85-1.0):   IOU delta=+3.2% (N=57)

✅ Report saved: data/annotation/reports/batch_001_eval.json
✅ Markdown saved: data/annotation/reports/batch_001_eval.md
```

**Files Generated**:
- `batch_001_eval.json`: Full report with all metrics
- `batch_001_eval.md`: Human-readable summary with tables

---

### Step 7: Generate Visualizations

**Command**:
```bash
python scripts/annotation/cli.py plot-metrics \
  --report data/annotation/reports/batch_001_eval.json \
  --output-dir data/annotation/plots/batch_001/ \
  --formats png pdf \
  --dpi 300 \
  --plots all
```

**Expected Output**:
```
Generated 6 plots in data/annotation/plots/batch_001/:
  ✅ iou_uplift.png (Weak vs LLM IOU distribution)
  ✅ calibration_curve.png (Confidence reliability)
  ✅ correction_rate.png (Improved/worsened breakdown)
  ✅ prf_comparison.png (Precision/Recall/F1 comparison)
  ✅ stratified_label.png (F1 by entity type)
  ✅ stratified_confidence.png (IOU delta by confidence bucket)
```

**Use Cases**:
- **Presentations**: High-DPI PNG for slides
- **Reports**: PDF for professional documentation
- **Analysis**: Identify patterns (e.g., LLM over-corrects low-confidence spans)

---

## Data Validation

### Pre-Evaluation Checks

Run these checks before evaluation to avoid errors:

**1. File Existence**:
```powershell
Test-Path data/annotation/batches/batch_001/weak_labels.jsonl
Test-Path data/annotation/batches/batch_001/llm_refined.jsonl
Test-Path data/gold/batch_001.jsonl
```

**2. JSONL Format Validation**:
```python
import json
from pathlib import Path

def validate_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                rec = json.loads(line)
                assert 'id' in rec, f"Line {i}: Missing 'id' field"
                assert 'text' in rec, f"Line {i}: Missing 'text' field"
                assert 'spans' in rec, f"Line {i}: Missing 'spans' field"
                for span in rec['spans']:
                    assert 'start' in span and 'end' in span, f"Line {i}: Invalid span"
            except Exception as e:
                print(f"❌ Error at line {i}: {e}")
                return False
    print(f"✅ {path} is valid JSONL")
    return True

validate_jsonl('data/gold/batch_001.jsonl')
```

**3. Span Integrity Test**:
```python
# Verify all spans have valid start/end positions
with open('data/gold/batch_001.jsonl', 'r') as f:
    for line in f:
        rec = json.loads(line)
        text = rec['text']
        for span in rec['spans']:
            start, end = span['start'], span['end']
            extracted = text[start:end]
            assert extracted == span['text'], \
                f"Span mismatch: '{span['text']}' != '{extracted}'"
print("✅ All spans have valid boundaries")
```

---

## CLI Execution Examples

### Example 1: Quick Evaluation (No Stratification)

```bash
python scripts/annotation/cli.py evaluate-llm \
  --weak data/annotation/batches/batch_001/weak_labels.jsonl \
  --refined data/annotation/batches/batch_001/llm_refined.jsonl \
  --gold data/gold/batch_001.jsonl \
  --output data/annotation/reports/batch_001_quick.json
```

**Use Case**: Fast quality check during annotation (no stratified analysis)

---

### Example 2: Full Evaluation with All Stratifications

```bash
python scripts/annotation/cli.py evaluate-llm \
  --weak data/annotation/batches/batch_001/weak_labels.jsonl \
  --refined data/annotation/batches/batch_001/llm_refined.jsonl \
  --gold data/gold/batch_001.jsonl \
  --output data/annotation/reports/batch_001_full.json \
  --markdown \
  --stratify label confidence span_length
```

**Use Case**: Comprehensive analysis for final batch report

---

### Example 3: Confidence-Only Stratification

```bash
python scripts/annotation/cli.py evaluate-llm \
  --weak data/annotation/batches/batch_001/weak_labels.jsonl \
  --refined data/annotation/batches/batch_001/llm_refined.jsonl \
  --gold data/gold/batch_001.jsonl \
  --output data/annotation/reports/batch_001_conf.json \
  --stratify confidence
```

**Use Case**: Focus on LLM performance by confidence bucket (optimize refinement threshold)

---

### Example 4: Visualization Only

```bash
# Assumes evaluation already run
python scripts/annotation/cli.py plot-metrics \
  --report data/annotation/reports/batch_001_eval.json \
  --output-dir data/annotation/plots/batch_001/ \
  --formats png \
  --plots iou calibration
```

**Use Case**: Generate specific plots for presentation (skip full suite)

---

## Result Interpretation

### Target Metrics (Healthy Annotation Quality)

| Metric | Target | Red Flag (<) | Interpretation |
|--------|--------|-------------|----------------|
| **IOU Improvement** | +8-15% | +5% | LLM refinement provides meaningful boundary correction |
| **Exact Match Rate** | 70-85% | 60% | LLM boundaries align well with gold standard |
| **Correction Rate (Improved)** | >60% | 50% | Majority of LLM changes improve weak labels |
| **Correction Rate (Worsened)** | <10% | >15% | LLM rarely introduces errors |
| **F1 Score (LLM vs Gold)** | >0.85 | <0.75 | High precision and recall |
| **Canonical Coverage** | >90% | <80% | Most symptoms map to lexicon entries |

### Interpretation Guide

#### 1. IOU Improvement Below Target (+5%)

**Possible Causes**:
- Weak labels already high quality (lexicon well-tuned)
- LLM over-correcting (removing valid multi-word terms)
- Gold standard boundaries inconsistent

**Actions**:
- Review worsened spans (correction rate breakdown)
- Check if LLM removing important context (e.g., "burning sensation" → "burning")
- Calibrate lexicon thresholds (increase fuzzy threshold to 0.90?)

---

#### 2. High Worsened Rate (>10%)

**Possible Causes**:
- LLM prompt too aggressive (removing necessary words)
- LLM hallucinating boundaries (rare but possible)
- Annotator inconsistency (gold standard variability)

**Actions**:
- Inspect worsened spans manually:
  ```python
  # Load evaluation report
  import json
  with open('data/annotation/reports/batch_001_eval.json', 'r') as f:
      report = json.load(f)
  
  # Find worsened spans
  corrections = report['overall']['correction_details']
  worsened = [c for c in corrections if c['category'] == 'worsened']
  
  for span in worsened[:10]:
      print(f"Weak:  '{span['weak_text']}'")
      print(f"LLM:   '{span['llm_text']}'")
      print(f"Gold:  '{span['gold_text']}'")
      print(f"IOU:   {span['weak_iou']:.3f} → {span['llm_iou']:.3f} (Δ={span['iou_delta']:.3f})")
      print()
  ```
- Adjust LLM prompt (reduce aggressiveness)
- Re-annotate sample tasks for consistency check

---

#### 3. Low Canonical Coverage (<80%)

**Possible Causes**:
- Lexicon incomplete (missing colloquial terms)
- Annotators using non-canonical synonyms
- Domain drift (new symptoms not in lexicon)

**Actions**:
- Extract missing terms:
  ```python
  # Find spans not in lexicon
  import pandas as pd
  
  symptoms = pd.read_csv('data/lexicon/symptoms.csv')
  canonical_set = set(symptoms['Canonical Term'].str.lower())
  
  with open('data/gold/batch_001.jsonl', 'r') as f:
      for line in f:
          rec = json.loads(line)
          for span in rec['spans']:
              if span['label'] == 'SYMPTOM':
                  term = span['text'].lower()
                  if term not in canonical_set:
                      print(f"Missing: '{term}'")
  ```
- Update lexicon with new entries
- Re-run weak labeling with expanded lexicon

---

#### 4. Low F1 Score (<0.75)

**Possible Causes**:
- High false positive rate (LLM over-generating spans)
- High false negative rate (LLM missing valid spans)
- Inconsistent annotation guidelines

**Actions**:
- Check precision vs recall:
  - **Low precision (many FP)**: LLM too liberal; increase confidence threshold
  - **Low recall (many FN)**: LLM too conservative; review skipped spans
- Stratify by confidence:
  ```bash
  python scripts/annotation/cli.py evaluate-llm ... --stratify confidence
  ```
  If low-confidence spans drag down F1, filter them out during import

---

## Iteration Strategy

### After First Batch (100 tasks)

**Goals**:
1. Identify systematic errors (boundary, negation, anatomy)
2. Refine prompts/lexicons based on worsened spans
3. Calibrate confidence thresholds for next batch

**Workflow**:

```python
# 1. Analyze worsened spans
import json
from collections import Counter

with open('data/annotation/reports/batch_001_eval.json', 'r') as f:
    report = json.load(f)

corrections = report['overall']['correction_details']
worsened = [c for c in corrections if c['category'] == 'worsened']

# Common patterns
llm_changes = [(c['weak_text'], c['llm_text']) for c in worsened]
change_patterns = Counter([
    'removed_adjective' if 'severe' in weak or 'mild' in weak else
    'truncated_compound' if len(llm.split()) < len(weak.split()) else
    'other'
    for weak, llm in llm_changes
])

print("Worsened Patterns:")
for pattern, count in change_patterns.most_common():
    print(f"  {pattern}: {count}")
```

**Example Output**:
```
Worsened Patterns:
  truncated_compound: 7  ← LLM removing important words
  removed_adjective: 3   ← Expected (but check if gold includes adjectives)
  other: 1
```

**Action**: If `truncated_compound` high, adjust LLM prompt:
```python
# In src/llm_agent.py, update prompt:
"Preserve multi-word medical terms even if they include anatomical references. 
Only remove intensity adjectives (severe, mild, slight)."
```

---

### After Third Batch (300 tasks total)

**Goals**:
1. Measure inter-batch consistency
2. Estimate final model performance (extrapolate F1)
3. Decide on batch size for remaining data

**Workflow**:

```python
# Compare metrics across batches
import pandas as pd

batches = ['batch_001', 'batch_002', 'batch_003']
metrics = []

for batch in batches:
    with open(f'data/annotation/reports/{batch}_eval.json', 'r') as f:
        report = json.load(f)
        overall = report['overall']
        metrics.append({
            'batch': batch,
            'iou_improvement': overall['iou_improvement_pct'],
            'f1': overall['llm_prf']['f1'],
            'worsened_pct': overall['correction_rate']['worsened_pct']
        })

df = pd.DataFrame(metrics)
print(df)
print(f"\nMean F1: {df['f1'].mean():.3f} (±{df['f1'].std():.3f})")
```

**Example Output**:
```
       batch  iou_improvement    f1  worsened_pct
0  batch_001            9.5  0.897           9.2
1  batch_002           11.2  0.903           7.8
2  batch_003            8.7  0.891          10.1

Mean F1: 0.897 (±0.006)  ← Stable performance, ready to scale
```

**Decision**:
- If F1 stable (std <0.01): Scale to 500-task batches
- If F1 volatile (std >0.02): Continue 100-task batches, investigate variability

---

## Troubleshooting

### Issue 1: Evaluation Script Fails with "Mismatched IDs"

**Error Message**:
```
ValueError: IDs in weak labels do not match gold labels
```

**Cause**: Batch preparation and conversion used different ID schemes

**Solution**:
```python
# Ensure IDs consistent across files
# In prepare_production_batch.py:
task_id = f"batch_{batch_num:03d}_task_{idx:03d}"

# In convert_labelstudio.py:
# Preserve original task IDs from Label Studio export
gold_id = original_task['data']['id']  # Not generated fresh
```

---

### Issue 2: Calibration Curve Shows Poor Confidence Reliability

**Symptom**: Calibration curve plot shows large gap between expected and observed IOU

**Cause**: Weak label confidence scores not calibrated (fuzzy + jaccard heuristic inaccurate)

**Solution**:
```python
# Recalibrate confidence formula in src/weak_label.py:
# Current: 0.8*fuzzy + 0.2*jaccard
# Tune weights based on evaluation data:

from sklearn.linear_model import LinearRegression

# Collect (confidence, IOU) pairs from batch evaluations
X = np.array([span['confidence'] for span in all_spans]).reshape(-1, 1)
y = np.array([span['iou_vs_gold'] for span in all_spans])

model = LinearRegression().fit(X, y)
print(f"Calibrated weights: {model.coef_}, intercept: {model.intercept_}")

# Update confidence formula with learned weights
```

---

### Issue 3: LLM Refinement Hangs on API Calls

**Symptom**: Script timeout after 60 seconds, no LLM response

**Cause**: API key invalid, rate limit hit, or network issue

**Solution**:
```bash
# Test API connectivity
python -c "
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model='gpt-4',
    messages=[{'role': 'user', 'content': 'test'}],
    max_tokens=10
)
print(response.choices[0].message.content)
"

# If fails, check:
# 1. API key valid: echo $env:OPENAI_API_KEY
# 2. Rate limit: Reduce batch size or add delay
# 3. Network: Test curl https://api.openai.com/v1/models
```

---

### Issue 4: Markdown Report Missing Stratification Tables

**Symptom**: `batch_001_eval.md` shows overall metrics but no stratified breakdowns

**Cause**: `--stratify` flag not used during evaluation

**Solution**:
```bash
# Re-run evaluation with stratification
python scripts/annotation/cli.py evaluate-llm \
  --weak ... \
  --refined ... \
  --gold ... \
  --output batch_001_eval.json \
  --markdown \
  --stratify label confidence span_length  # ← Add this
```

---

## Summary

### Checklist: First Production Batch

- [ ] **Batch Preparation**: 100 tasks stratified by confidence
- [ ] **Label Studio Import**: Config loaded, tasks imported with pre-annotations
- [ ] **Annotation**: All 100 tasks completed (2-3 hours)
- [ ] **Export & Convert**: Gold JSONL with canonical coverage >90%
- [ ] **Evaluation**: IOU improvement >8%, F1 >0.85, worsened rate <10%
- [ ] **Visualization**: 6 plots generated (IOU, calibration, correction, P/R/F1, stratified)
- [ ] **Iteration**: Worsened spans analyzed, prompts/lexicons refined if needed

### Next Steps

1. **Annotate Batches 2-3** (200 more tasks):
   - Validate consistency (compare F1 across batches)
   - Refine guidelines based on common errors

2. **Scale to 500 Tasks**:
   - Once metrics stable, increase batch size
   - Target: 1,000 gold annotations for fine-tuning

3. **Fine-Tune BioBERT**:
   - Train token classification head on gold data
   - Evaluate on held-out test set (20% of gold annotations)
   - Compare to weak/LLM baselines

---

**Questions?** See `docs/annotation_guide.md` for boundary rules, `docs/llm_evaluation.md` for metric definitions, or open a GitHub issue.
