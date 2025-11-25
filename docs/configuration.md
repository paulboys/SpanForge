# Configuration

Complete guide to SpanForge configuration options.

## Configuration File

SpanForge uses Pydantic BaseSettings for configuration management with environment variable support.

### AppConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | `"dmis-lab/biobert-base-cased-v1.1"` | HuggingFace model identifier |
| `max_seq_len` | int | `256` | Maximum sequence length for tokenization |
| `device` | str | auto-detect | Computation device (`'cuda'` or `'cpu'`) |
| `seed` | int | `42` | Random seed for reproducibility |
| `negation_window` | int | `5` | Tokens after negation cue to mark as negated |
| `fuzzy_scorer` | str | `"wratio"` | Fuzzy matching algorithm (`'wratio'` or `'jaccard'`) |
| `llm_enabled` | bool | `False` | Enable experimental LLM refinement |
| `llm_provider` | str | `"stub"` | LLM provider (`'stub'`, `'openai'`, `'azure'`) |
| `llm_model` | str | `"gpt-4"` | LLM model identifier |
| `llm_min_confidence` | float | `0.65` | Minimum confidence for LLM suggestions |
| `llm_cache_path` | str | `"data/annotation/exports/llm_cache.jsonl"` | LLM response cache file |
| `llm_prompt_version` | str | `"v1"` | Prompt template version |

## Usage Examples

### Basic Configuration

```python
from src.config import AppConfig

# Use defaults
config = AppConfig()
print(config.device)  # 'cuda' if available, else 'cpu'
print(config.negation_window)  # 5
```

### Custom Configuration

```python
# Override defaults
config = AppConfig(
    negation_window=7,
    fuzzy_scorer="jaccard",
    max_seq_len=512,
    device="cpu"
)
```

### Environment Variables

Set via environment variables (prefixed with your app name if needed):

```bash
export MODEL_NAME="dmis-lab/biobert-v1.1"
export NEGATION_WINDOW=10
export DEVICE="cuda"
```

```python
# Automatically reads from environment
config = AppConfig()
```

### Seed Management

```python
from src.config import set_seed

# Set for reproducibility
set_seed(42)

# All random operations now deterministic
import random
import numpy as np
print(random.random())  # Same value every run
print(np.random.rand())  # Same value every run
```

## Parameter Details

### Model Configuration

#### model_name

HuggingFace model identifier. Default is BioBERT base cased v1.1.

**Options:**
- `"dmis-lab/biobert-base-cased-v1.1"` (default)
- `"dmis-lab/biobert-large-cased-v1.1"`
- `"microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"`
- Any HuggingFace transformer model

```python
config = AppConfig(model_name="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
```

#### max_seq_len

Maximum sequence length for tokenization. Longer texts are truncated.

**Recommendations:**
- `256`: Default, good balance for complaints (1-3 sentences)
- `512`: Longer documents, increased memory usage
- `128`: Short texts, faster processing

```python
config = AppConfig(max_seq_len=512)  # For longer documents
```

#### device

Computation device. Auto-detects CUDA availability.

```python
# Force CPU
config = AppConfig(device="cpu")

# Use specific GPU
config = AppConfig(device="cuda:1")
```

### Weak Labeling Configuration

#### negation_window

Number of tokens after negation cue to mark as negated.

**Tuning:**
- `3-5`: Short-range negation (default: 5)
- `7-10`: Long-range negation (more false positives)
- `1-2`: Very conservative

```python
# Example: "Patient has no history of itching or redness"
config = AppConfig(negation_window=7)  # Catches "redness" too
```

#### fuzzy_scorer

Fuzzy matching algorithm selection.

**Options:**
- `"wratio"` (default): WRatio scoring, better for misspellings
- `"jaccard"`: Token-set Jaccard, better for synonym matching

```python
# For exact synonym matching
config = AppConfig(fuzzy_scorer="jaccard")
```

### LLM Configuration (Experimental)

#### llm_enabled

Enable LLM-based span refinement pipeline.

```python
config = AppConfig(
    llm_enabled=True,
    llm_provider="openai",  # or "azure", "anthropic"
    llm_model="gpt-4",
    llm_min_confidence=0.7
)
```

#### llm_min_confidence

Minimum confidence threshold for accepting LLM suggestions.

**Recommendations:**
- `0.5-0.6`: Exploratory, more suggestions
- `0.65-0.75`: Balanced (default: 0.65)
- `0.8-0.9`: Conservative, high precision

## Performance Tuning

### CPU Optimization

```python
config = AppConfig(
    device="cpu",
    max_seq_len=128,  # Shorter sequences
    # Use exact matching only when possible
)
```

### GPU Optimization

```python
config = AppConfig(
    device="cuda",
    max_seq_len=512,  # Longer sequences
)

# Enable GPU optimizations
import torch
torch.backends.cudnn.benchmark = True
```

### Batch Processing

```python
from src.pipeline import simple_inference

# Process large batches
texts = load_texts("large_dataset.txt")  # e.g., 10,000 texts

# Batch processing
batch_size = 32
results = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    results.extend(simple_inference(batch))
```

## Configuration Profiles

### Development Profile

```python
dev_config = AppConfig(
    device="cpu",
    max_seq_len=128,
    negation_window=5,
    seed=42,
    llm_enabled=False
)
```

### Production Profile

```python
prod_config = AppConfig(
    device="cuda",
    max_seq_len=256,
    negation_window=7,
    fuzzy_scorer="wratio",
    seed=42,
    llm_enabled=True,
    llm_provider="azure",
    llm_min_confidence=0.75
)
```

### Testing Profile

```python
test_config = AppConfig(
    device="cpu",
    seed=42,  # Deterministic
    llm_enabled=False,  # No external calls
    negation_window=5
)
```

## Best Practices

1. **Always set seed** for reproducible experiments
2. **Profile before tuning** - measure actual performance
3. **Start with defaults** - they work well for most cases
4. **Use environment variables** for deployment secrets
5. **Document custom configs** in code comments

## Troubleshooting

### Issue: CUDA out of memory

**Solutions:**
```python
# Reduce sequence length
config = AppConfig(max_seq_len=128)

# Force CPU
config = AppConfig(device="cpu")

# Clear cache between batches
import torch
torch.cuda.empty_cache()
```

### Issue: Poor negation detection

**Solutions:**
```python
# Extend window for long-range negation
config = AppConfig(negation_window=10)

# Check NEGATION_TOKENS in src/weak_label.py
# Add custom negation cues if needed
```

### Issue: Low fuzzy match recall

**Solutions:**
```python
# Try Jaccard scorer
config = AppConfig(fuzzy_scorer="jaccard")

# Lower fuzzy threshold in match_symptoms()
spans = match_symptoms(text, lexicon, fuzzy_threshold=85.0)
```

## Next Steps

- [Quick Start](quickstart.md) - Get started with examples
- [User Guide: Weak Labeling](user-guide/weak-labeling.md) - Advanced techniques
- [API Reference](api/config.md) - Full API documentation
