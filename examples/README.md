# SpanForge Examples

> **Comprehensive tutorials and practical examples for biomedical NER with SpanForge**

This directory contains hands-on examples demonstrating SpanForge's capabilities, from basic entity extraction to advanced LLM refinement and production workflows.

---

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Learning Path](#learning-path)
- [Example Categories](#example-categories)
- [Prerequisites](#prerequisites)
- [Running Examples](#running-examples)
- [Dataset Information](#dataset-information)

---

## 🚀 Quick Start

**Install SpanForge:**
```bash
pip install -e .
```

**Run your first example:**
```bash
cd examples/basic
python simple_ner.py
```

**Expected output:**
```
Detected 2 entities:
  • "burning sensation" [SYMPTOM] at position 15-32 (confidence: 1.00)
  • "redness" [SYMPTOM] at position 37-44 (confidence: 0.92)
```

---

## 🎓 Learning Path

### **Level 1: Fundamentals** (30 minutes)
Start here if you're new to SpanForge or biomedical NER.

1. **`basic/simple_ner.py`** - Extract symptoms and products from text
2. **`basic/weak_labeling.py`** - Understand lexicon-based span detection
3. **`basic/negation_detection.py`** - Handle negated entities

**Key Concepts:** Lexicons, fuzzy matching, negation windows, confidence scores

---

### **Level 2: Advanced Features** (1 hour)
Dive deeper into customization and optimization.

1. **`advanced/llm_refinement.py`** - Use GPT-4/Claude to improve weak labels
2. **`advanced/custom_lexicons.py`** - Build domain-specific lexicons
3. **`advanced/batch_processing.py`** - Process thousands of documents efficiently
4. **`advanced/caers_integration.py`** - Work with FDA adverse event reports

**Key Concepts:** LLM agents, boundary correction, caching, real-world data

---

### **Level 3: Evaluation & Quality** (45 minutes)
Measure and improve model performance.

1. **`evaluation/compute_metrics.py`** - Calculate precision, recall, F1, IOU
2. **`evaluation/stratified_analysis.py`** - Analyze by confidence, label, span length
3. **`evaluation/visualization.py`** - Generate publication-quality plots

**Key Concepts:** Gold standards, calibration curves, correction rates, P/R/F1

---

### **Level 4: Production Workflows** (1 hour)
Build annotation pipelines and quality assurance systems.

1. **`annotation/prepare_batch.py`** - Stratified sampling and de-identification
2. **`annotation/label_studio_setup.py`** - Configure annotation environment
3. **`annotation/quality_report.py`** - Compute inter-annotator agreement

**Key Concepts:** Label Studio, IAA, conflict resolution, gold standard creation

---

## 📁 Example Categories

### **`basic/`** - Foundational Usage
Simple, self-contained examples for learning core functionality.

| File | Description | Lines | Runtime |
|------|-------------|-------|---------|
| `simple_ner.py` | Hello World for entity extraction | ~60 | <1s |
| `weak_labeling.py` | Lexicon-based span detection explained | ~90 | <1s |
| `negation_detection.py` | Handle "no pain", "without rash" | ~75 | <1s |
| `config_examples.py` | Customize thresholds and models | ~85 | <1s |

**What you'll learn:**
- Load lexicons from CSV
- Extract spans with confidence scores
- Configure fuzzy matching thresholds
- Handle negation and boundary cases

---

### **`advanced/`** - Real-World Applications
Production-ready examples with best practices.

| File | Description | Lines | Runtime |
|------|-------------|-------|---------|
| `llm_refinement.py` | GPT-4/Claude boundary correction | ~120 | 5-10s |
| `custom_lexicons.py` | Build lexicons from MedDRA/RxNorm | ~140 | 2-3s |
| `batch_processing.py` | Process 10K+ documents efficiently | ~110 | 1-2min |
| `caers_integration.py` | FDA adverse event data pipeline | ~95 | 30s |

**What you'll learn:**
- Multi-provider LLM integration (OpenAI, Azure, Anthropic)
- Lexicon engineering from medical ontologies
- Memory-efficient batch processing
- Real-world data acquisition and weak labeling

---

### **`evaluation/`** - Performance Analysis
Quantify accuracy and calibrate confidence scores.

| File | Description | Lines | Runtime |
|------|-------------|-------|---------|
| `compute_metrics.py` | IOU, boundary precision, P/R/F1 | ~100 | <1s |
| `stratified_analysis.py` | Break down metrics by subgroups | ~130 | 2-3s |
| `visualization.py` | Generate calibration curves & plots | ~115 | 5s |
| `compare_baselines.py` | Weak vs LLM vs gold comparison | ~105 | 3s |

**What you'll learn:**
- Implement evaluation harness
- Compute 10 standard NER metrics
- Stratify by confidence, label type, span length
- Generate publication-quality visualizations

---

### **`annotation/`** - Human-in-the-Loop
Build annotation workflows for gold standard creation.

| File | Description | Lines | Runtime |
|------|-------------|-------|---------|
| `prepare_batch.py` | Stratified sampling + de-ID | ~125 | 5s |
| `label_studio_setup.py` | Configure annotation project | ~90 | 10s |
| `export_gold_standard.py` | Convert annotations to JSONL | ~80 | 2s |
| `quality_report.py` | IAA, disagreement rate, drift | ~140 | 3s |

**What you'll learn:**
- Design annotation batches for model improvement
- Integrate with Label Studio
- Compute inter-annotator agreement (IAA)
- Detect annotation drift and quality issues

---

## ⚙️ Prerequisites

### **Required Dependencies**
```bash
# Core installation
pip install -e .
```

**Included by default:**
- `transformers` (BioBERT model)
- `rapidfuzz` (fuzzy matching)
- `pydantic` (configuration)
- `pandas` (lexicon handling)

### **Optional Dependencies**

**For LLM Refinement** (`advanced/llm_refinement.py`):
```bash
pip install -e ".[llm]"  # openai, anthropic, tenacity
export OPENAI_API_KEY="sk-..."
```

**For Visualization** (`evaluation/visualization.py`):
```bash
pip install -e ".[viz]"  # matplotlib, seaborn, numpy
```

**For Annotation** (`annotation/label_studio_setup.py`):
```bash
pip install label-studio
export LABEL_STUDIO_DISABLE_TELEMETRY=1
```

---

## 🏃 Running Examples

### **Individual Example**
```bash
cd examples/basic
python simple_ner.py
```

### **All Examples in Category**
```bash
cd examples/basic
for file in *.py; do
    echo "Running $file..."
    python "$file"
    echo "---"
done
```

### **With Custom Data**
Most examples accept command-line arguments:
```bash
python basic/weak_labeling.py \
    --input my_text.txt \
    --symptom-lexicon data/lexicon/symptoms.csv \
    --output results.jsonl
```

### **Interactive Mode**
Some examples support interactive exploration:
```bash
python advanced/llm_refinement.py --interactive
# Enter text at prompt, see real-time refinement
```

---

## 📊 Dataset Information

### **Sample Data**
Each example includes embedded sample text for immediate testing:
- Consumer complaint narratives (cosmetics, supplements)
- Clinical notes (de-identified)
- Adverse event reports (FDA CAERS)

### **Full CAERS Dataset**
Download 666K+ FDA consumer adverse event reports:
```bash
python scripts/caers/download_caers.py \
    --output data/caers/all_cosmetics.jsonl \
    --categories cosmetics \
    --limit 10000
```

See [`scripts/caers/README.md`](../scripts/caers/README.md) for details.

### **Custom Data**
Prepare your own data:
1. **Plain text files** (`.txt`) - One document per file
2. **JSONL format** - `{"text": "...", "metadata": {...}}`
3. **CSV with text column** - Batch processing examples auto-detect

---

## 🎯 Use Cases by Example

### **I want to...**

**Extract entities from clinical notes**
→ Start with `basic/simple_ner.py`, then `advanced/batch_processing.py`

**Improve weak labels with LLM**
→ Use `advanced/llm_refinement.py` with OpenAI or Anthropic

**Build a custom symptom lexicon**
→ Follow `advanced/custom_lexicons.py` using MedDRA or SNOMED CT

**Evaluate model performance**
→ Run `evaluation/compute_metrics.py` after creating gold standard

**Set up annotation workflow**
→ Complete sequence: `annotation/prepare_batch.py` → `label_studio_setup.py` → `quality_report.py`

**Reproduce paper results**
→ See [`docs/benchmarks.md`](../docs/benchmarks.md) for dataset links and exact commands

---

## 📖 Additional Resources

- **API Documentation**: [`docs/api/`](../docs/api/)
- **Architecture Overview**: [`docs/overview.md`](../docs/overview.md)
- **Annotation Guide**: [`docs/annotation_guide.md`](../docs/annotation_guide.md)
- **Heuristic Details**: [`docs/heuristic.md`](../docs/heuristic.md)
- **Production Workflow**: [`docs/production_workflow.md`](../docs/production_workflow.md)

---

## 🤝 Contributing Examples

Have a useful example? Submit a PR!

**Guidelines:**
1. **Self-contained**: Include sample data or generate it programmatically
2. **Well-documented**: Inline comments + docstring explaining purpose
3. **Tested**: Verify example runs without errors
4. **Timely**: Should complete in <2 minutes (or use `--limit` flag)
5. **Categorized**: Place in appropriate subdirectory

**Example template:**
```python
"""
Brief one-line description.

Demonstrates: [key concept 1], [key concept 2]
Prerequisites: [optional dependencies]
Runtime: [expected duration]
"""

def main():
    # Setup with embedded sample data
    sample_text = "..."
    
    # Core logic with explanatory comments
    # ...
    
    # Print results clearly
    print("Results:")
    # ...

if __name__ == "__main__":
    main()
```

---

## 💬 Getting Help

- **Questions**: Open GitHub Discussion
- **Bugs**: File GitHub Issue with example script
- **Feature Requests**: Tag with `enhancement` and `examples`

---

**Last Updated**: November 27, 2025  
**SpanForge Version**: 0.5.0  
**Python Compatibility**: 3.9, 3.10, 3.11
