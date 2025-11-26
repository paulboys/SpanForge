# LLM Provider Integration Guide

## Overview

SpanForge now supports real LLM provider integration for entity span refinement. The system supports multiple providers with automatic caching, retry logic, and graceful error handling.

## Supported Providers

### 1. Stub Mode (Default)
- **Use Case**: Testing and development without API costs
- **Configuration**: `llm_provider="stub"`
- **Behavior**: Returns empty suggestions without making API calls

### 2. OpenAI
- **Models**: gpt-4, gpt-4-turbo, gpt-3.5-turbo, etc.
- **Installation**: `pip install -r requirements-llm.txt`
- **Configuration**: 
  ```python
  config = AppConfig(
      llm_enabled=True,
      llm_provider="openai",
      llm_model="gpt-4-turbo"
  )
  ```
- **Environment Variables**:
  - `OPENAI_API_KEY`: Your OpenAI API key (required)

### 3. Azure OpenAI
- **Models**: Deployed models in your Azure OpenAI resource
- **Installation**: `pip install -r requirements-llm.txt`
- **Configuration**:
  ```python
  config = AppConfig(
      llm_enabled=True,
      llm_provider="azure",
      llm_model="your-deployment-name"
  )
  ```
- **Environment Variables**:
  - `AZURE_OPENAI_API_KEY`: Your Azure OpenAI key (required)
  - `AZURE_OPENAI_ENDPOINT`: Your Azure endpoint URL (required)
    - Example: `https://your-resource.openai.azure.com/`

### 4. Anthropic Claude
- **Models**: claude-3-opus, claude-3-sonnet, claude-3-haiku
- **Installation**: `pip install -r requirements-llm.txt`
- **Configuration**:
  ```python
  config = AppConfig(
      llm_enabled=True,
      llm_provider="anthropic",
      llm_model="claude-3-sonnet-20240229"
  )
  ```
- **Environment Variables**:
  - `ANTHROPIC_API_KEY`: Your Anthropic API key (required)

## Quick Start

### 1. Install Dependencies

```powershell
# Install optional LLM dependencies
pip install -r requirements-llm.txt
```

This installs:
- `openai>=1.0.0` (for OpenAI and Azure OpenAI)
- `anthropic>=0.18.0` (for Anthropic Claude)
- `tenacity>=8.0.0` (for retry logic with exponential backoff)

### 2. Set Environment Variables

**Windows PowerShell:**
```powershell
# OpenAI
$env:OPENAI_API_KEY = "sk-..."

# Azure OpenAI
$env:AZURE_OPENAI_API_KEY = "your-key"
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"

# Anthropic
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

**Linux/Mac:**
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Use in Code

```python
from src.config import AppConfig
from src.llm_agent import LLMAgent

# Configure for OpenAI
config = AppConfig(
    llm_enabled=True,
    llm_provider="openai",
    llm_model="gpt-4-turbo",
    llm_min_confidence=0.65
)

# Initialize agent (reads from config)
agent = LLMAgent()

# Use for span refinement
template = "Analyze medical text for entities..."
suggestions = agent.suggest(template, text, weak_spans, knowledge)

# Each suggestion includes:
# - start/end positions
# - label (SYMPTOM or PRODUCT)
# - confidence score
# - reasoning (optional)
# - canonical form (optional)
```

## Features

### Response Caching
- All API responses are cached to disk (JSONL format)
- Cache location: `data/annotation/exports/llm_cache.jsonl`
- Responses are keyed by prompt hash for fast lookup
- Prevents redundant API calls and reduces costs

### Retry Logic
- Automatic exponential backoff for transient failures
- Up to 3 retry attempts with increasing delays (2s, 4s, 8s)
- Graceful handling of rate limits and timeouts
- Requires `tenacity` package (installed with requirements-llm.txt)

### Error Handling
- Missing API keys: Clear error messages
- SDK not installed: Helpful installation instructions
- API errors: Returns empty spans with error notes instead of crashing
- Unsupported providers: Validation error with supported list

## Configuration Options

All LLM settings are in `AppConfig`:

```python
config = AppConfig(
    # Enable/disable LLM refinement
    llm_enabled=True,
    
    # Provider selection
    llm_provider="openai",  # "stub", "openai", "azure", "anthropic"
    
    # Model identifier
    llm_model="gpt-4-turbo",
    
    # Confidence threshold (0.0-1.0)
    llm_min_confidence=0.65,
    
    # Cache file location
    llm_cache_path="data/annotation/exports/llm_cache.jsonl",
    
    # Prompt template version
    llm_prompt_version="v1"
)
```

## Cost Management

### Tips for Reducing Costs

1. **Use Caching**: Responses are automatically cached to avoid duplicate calls
2. **Start with Stub Mode**: Develop and test without API costs
3. **Use Smaller Models**: Consider gpt-3.5-turbo or claude-haiku for initial runs
4. **Batch Processing**: Process multiple texts in batches to amortize setup costs
5. **Set Conservative Thresholds**: Higher `llm_min_confidence` reduces false positives

### Approximate Costs (as of 2024)

| Provider | Model | Input Cost | Output Cost |
|----------|-------|------------|-------------|
| OpenAI | gpt-4-turbo | $10 / 1M tokens | $30 / 1M tokens |
| OpenAI | gpt-3.5-turbo | $0.50 / 1M tokens | $1.50 / 1M tokens |
| Anthropic | claude-3-sonnet | $3 / 1M tokens | $15 / 1M tokens |
| Anthropic | claude-3-haiku | $0.25 / 1M tokens | $1.25 / 1M tokens |

*Check provider websites for current pricing*

## Testing

Run the test suite to verify setup:

```powershell
# Run all LLM tests
pytest tests/test_llm_agent.py -v

# Run with coverage
pytest tests/test_llm_agent.py --cov=src.llm_agent --cov-report=term-missing
```

Tests will skip provider-specific tests if packages aren't installed.

## Troubleshooting

### "openai package not installed"
```powershell
pip install -r requirements-llm.txt
```

### "OPENAI_API_KEY environment variable not set"
```powershell
$env:OPENAI_API_KEY = "sk-your-key-here"
```

### "API error: 429 Too Many Requests"
- Rate limit exceeded
- Retry logic will handle automatically (exponential backoff)
- Consider reducing request frequency

### Cache file not found
- Directory created automatically on first API call
- Ensure write permissions for `data/annotation/exports/`

### Import errors with tenacity
- Retry logic disabled if tenacity not installed
- Install with `pip install tenacity>=8.0.0`
- System will work without retries (single attempt only)

## Next Steps

1. **Evaluate Performance**: Run evaluation harness to measure IOU uplift
2. **Tune Confidence**: Adjust `llm_min_confidence` based on precision/recall metrics
3. **Scale Up**: Process full dataset once satisfied with sample results
4. **Active Learning**: Use LLM suggestions to guide human annotation priorities

## References

- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Anthropic Claude API](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
