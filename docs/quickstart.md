# Quick Start

Get up and running with SpanForge in 5 minutes.

## Basic Entity Detection

### 1. Load Lexicons

```python
from pathlib import Path
from src.weak_label import load_symptom_lexicon, load_product_lexicon

# Load symptom and product lexicons
symptom_lex = load_symptom_lexicon(Path("data/lexicon/symptoms.csv"))
product_lex = load_product_lexicon(Path("data/lexicon/products.csv"))

print(f"Loaded {len(symptom_lex)} symptoms, {len(product_lex)} products")
```

### 2. Detect Entities

```python
from src.weak_label import weak_label

text = "Patient developed severe rash and itching after using the moisturizer"
spans = weak_label(text, symptom_lex, product_lex)

for span in spans:
    print(f"{span.text:20} | {span.label:10} | conf={span.confidence:.2f} | negated={span.negated}")
```

Output:
```
severe rash          | SYMPTOM    | conf=1.00 | negated=False
itching              | SYMPTOM    | conf=1.00 | negated=False
moisturizer          | PRODUCT    | conf=1.00 | negated=False
```

### 3. Negation Detection

```python
text = "No irritation from the face wash"
spans = weak_label(text, symptom_lex, product_lex)

for span in spans:
    print(f"{span.text}: negated={span.negated}")
```

Output:
```
irritation: negated=True
face wash: negated=False
```

## Batch Processing

### Process Multiple Documents

```python
from src.weak_label import weak_label_batch

texts = [
    "Severe headache after using the serum",
    "No side effects, works great",
    "Mild dryness but no redness"
]

# Batch process
all_spans = weak_label_batch(texts, symptom_lex, product_lex)

for i, (text, spans) in enumerate(zip(texts, all_spans), 1):
    print(f"\nDocument {i}: {text}")
    print(f"Found {len(spans)} entities:")
    for span in spans:
        print(f"  - {span.text} ({span.label})")
```

### Save to JSONL

```python
from src.weak_label import persist_weak_labels_jsonl
from pathlib import Path

# Persist results
output_path = Path("data/output/results.jsonl")
persist_weak_labels_jsonl(texts, all_spans, output_path)

print(f"Saved to {output_path}")
```

## Pipeline Inference

### Full BioBERT + Weak Labeling

```python
from src.pipeline import simple_inference

texts = [
    "Itching and redness from the facial cream",
    "No adverse effects noted"
]

# Run full pipeline
results = simple_inference(texts, persist_path="output.jsonl")

for result in results:
    print(f"Token count: {result['token_count']}")
    print(f"Detected spans: {len(result['weak_spans'])}")
    
    for span in result['weak_spans']:
        print(f"  {span['text']} ({span['label']})")
```

## Configuration

### Custom Parameters

```python
from src.config import AppConfig

# Create custom config
config = AppConfig(
    negation_window=7,  # Extend negation window
    fuzzy_scorer="jaccard",  # Use Jaccard instead of WRatio
    device="cpu"  # Force CPU
)

# Use in weak labeling
from src.weak_label import match_symptoms

spans = match_symptoms(
    text="Patient has severe itching",
    lexicon=symptom_lex,
    negation_window=config.negation_window,
    scorer=config.fuzzy_scorer
)
```

## Working with Spans

### Filter by Confidence

```python
# Get high-confidence spans only
high_conf_spans = [s for s in spans if s.confidence >= 0.9]

print(f"High confidence: {len(high_conf_spans)}/{len(spans)}")
```

### Group by Label

```python
from collections import defaultdict

by_label = defaultdict(list)
for span in spans:
    by_label[span.label].append(span)

print(f"Symptoms: {len(by_label['SYMPTOM'])}")
print(f"Products: {len(by_label['PRODUCT'])}")
```

### Check for Negated Entities

```python
negated = [s for s in spans if s.negated]
affirmed = [s for s in spans if not s.negated]

print(f"Affirmed: {len(affirmed)}")
print(f"Negated: {len(negated)}")
```

## Common Patterns

### Entity Co-occurrence

```python
def find_symptom_product_pairs(text, symptom_lex, product_lex):
    """Find symptoms mentioned with products."""
    spans = weak_label(text, symptom_lex, product_lex)
    
    symptoms = [s for s in spans if s.label == "SYMPTOM" and not s.negated]
    products = [s for s in spans if s.label == "PRODUCT"]
    
    if symptoms and products:
        return [(s.text, p.text) for s in symptoms for p in products]
    return []

# Example
text = "Developed rash and itching from the hydra cream"
pairs = find_symptom_product_pairs(text, symptom_lex, product_lex)
print(f"Symptom-Product pairs: {pairs}")
# Output: [('rash', 'hydra cream'), ('itching', 'hydra cream')]
```

### Confidence Threshold

```python
def filter_confident_spans(spans, threshold=0.85):
    """Keep only high-confidence, non-negated spans."""
    return [
        s for s in spans 
        if s.confidence >= threshold and not s.negated
    ]

confident = filter_confident_spans(spans, threshold=0.9)
```

## Next Steps

- [Configuration Guide](configuration.md) - Detailed parameter tuning
- [User Guide: Weak Labeling](user-guide/weak-labeling.md) - Advanced techniques
- [User Guide: Negation](user-guide/negation.md) - Negation patterns
- [API Reference](api/weak_label.md) - Complete API documentation
