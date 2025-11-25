# Installation

## Prerequisites

- Python 3.9, 3.10, or 3.11
- Git
- 4GB RAM minimum
- Optional: CUDA-capable GPU for faster inference

## Standard Installation

### 1. Clone Repository

```bash
git clone https://github.com/paulboys/SpanForge.git
cd SpanForge
```

### 2. Create Virtual Environment

=== "Conda"
    ```bash
    conda create -n spanforge python=3.10
    conda activate spanforge
    ```

=== "venv"
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    # or
    venv\Scripts\activate  # Windows
    ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python scripts/verify_env.py
```

Expected output:
```
✓ PyTorch installed
✓ Transformers installed
✓ Device: cuda (or cpu)
✓ BioBERT model downloadable
✓ Environment ready
```

## Development Installation

For contributors who want to run tests and linting:

```bash
# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pre-commit ruff black isort mypy

# Setup pre-commit hooks
pre-commit install

# Verify tests pass
pytest tests/ -v
```

## Documentation Build

To build documentation locally:

```bash
pip install -r docs-requirements.txt
mkdocs serve
```

Visit http://127.0.0.1:8000 to view docs.

## Troubleshooting

### Issue: PyTorch CUDA not detected

**Solution**: Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Transformers model download fails

**Solution**: Check internet connection or use offline model:
```bash
# Download model manually
from transformers import AutoModel
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model.save_pretrained("./models/biobert")
```

### Issue: Import errors

**Solution**: Ensure you're in the project root and virtual environment is activated:
```bash
pwd  # Should show SpanForge directory
which python  # Should show venv python
```

## Optional Components

### GPU Acceleration

For CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Label Studio (for annotation)

```bash
pip install label-studio
setx LABEL_STUDIO_DISABLE_TELEMETRY 1  # Windows
# or
export LABEL_STUDIO_DISABLE_TELEMETRY=1  # Linux/Mac
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Configuration Options](configuration.md)
- [API Reference](api/config.md)
