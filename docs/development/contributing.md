# Contributing to SpanForge

Thank you for your interest in contributing to SpanForge! This guide will help you get started.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Contribution Areas](#contribution-areas)
- [Community Guidelines](#community-guidelines)

---

## Getting Started

### Prerequisites

- Python 3.9, 3.10, or 3.11
- Git
- Virtual environment tool (venv or conda)
- Familiarity with PyTorch and transformers

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/spanforge.git
cd spanforge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies (dev mode)
pip install -r requirements.txt
pip install -r docs-requirements.txt
pip install pre-commit pytest pytest-cov black isort

# Verify setup
python scripts/verify_env.py
pytest -v
```

---

## Development Setup

### Install Pre-commit Hooks

Pre-commit hooks ensure code quality before commits:

```bash
pre-commit install
```

Hooks will now run automatically on `git commit`:
- pytest (all tests must pass)
- Code formatting checks

### Configure IDE

#### VS Code

Recommended extensions:
- Python (Microsoft)
- Pylance
- Jupyter
- GitLens

Recommended settings (`.vscode/settings.json`):

```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["-v"],
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

#### PyCharm

- Enable pytest as test runner (Settings → Tools → Python Integrated Tools)
- Configure Black as formatter (Settings → Tools → Black)
- Enable type checking (Settings → Editor → Inspections → Python)

---

## Contribution Workflow

### 1. Create Issue (Optional but Recommended)

Before starting work, create an issue describing:
- Problem or feature
- Proposed solution
- Expected impact

Wait for maintainer feedback before starting large changes.

### 2. Fork and Branch

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/your-username/spanforge.git
cd spanforge

# Add upstream remote
git remote add upstream https://github.com/original-org/spanforge.git

# Create feature branch
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Write code following [Code Standards](#code-standards)
- Add tests for new functionality
- Update documentation
- Commit regularly with clear messages

### 4. Test Locally

```bash
# Run all tests
pytest -v

# Run specific test categories
pytest tests/test_weak_label_core.py -v

# Check coverage
pytest --cov=src --cov-report=html

# Build documentation
cd docs
mkdocs serve
```

### 5. Commit

```bash
# Stage changes
git add .

# Commit (pre-commit hooks will run)
git commit -m "feat: add bidirectional negation detection"

# If hooks fail, fix issues and commit again
```

### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/your-feature-name

# Create pull request on GitHub
```

---

## Code Standards

### Style Guide

**Follow PEP 8** with these specifics:

- **Line length**: 100 characters (not 88 like Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes `"` for strings
- **Imports**: Organize with isort (automatic)

### Formatting

Use **Black** for automatic formatting:

```bash
# Format all files
black src tests

# Check without formatting
black --check src tests
```

### Import Organization

Use **isort** for import sorting:

```bash
# Sort imports
isort src tests

# Check without sorting
isort --check-only src tests
```

**Import order:**
1. Standard library
2. Third-party packages
3. Local modules

```python
# Standard library
import os
import sys
from typing import List, Optional

# Third-party
import torch
from transformers import AutoModel

# Local
from src.config import AppConfig
from src.model import get_model
```

### Type Hints

**Always use type hints** for function signatures:

```python
from typing import List, Dict, Optional

def match_symptoms(
    text: str,
    lexicon: List[str],
    fuzzy_threshold: float = 88.0,
    negation_window: int = 5
) -> List[Dict[str, any]]:
    """
    Match symptoms in text using fuzzy matching.
    
    Args:
        text: Input text to search
        lexicon: List of symptom terms
        fuzzy_threshold: Minimum fuzzy score (0-100)
        negation_window: Negation scope in tokens
    
    Returns:
        List of matched span dictionaries
    """
    pass
```

### Docstrings

Use **Google-style docstrings**:

```python
def example_function(arg1: str, arg2: int = 0) -> bool:
    """
    Short one-line description.
    
    Longer description explaining functionality, edge cases,
    and important details.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2 (default: 0)
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When arg1 is empty
    
    Examples:
        >>> example_function("test", 5)
        True
        >>> example_function("", 0)
        ValueError: arg1 cannot be empty
    """
    if not arg1:
        raise ValueError("arg1 cannot be empty")
    return len(arg1) > arg2
```

### Naming Conventions

- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private/Internal**: `_leading_underscore`

```python
# Good
def match_symptoms(text: str) -> List[Dict]:
    NEGATION_TOKENS = {"no", "not", "never"}
    _internal_cache = {}
    
class WeakLabelMatcher:
    pass
```

---

## Testing Requirements

### Test Categories

All contributions must include tests:

1. **Core Tests**: Basic functionality
2. **Edge Case Tests**: Boundary conditions, unicode, errors
3. **Integration Tests**: End-to-end workflows

### Writing Tests

```python
import pytest
from src.weak_label import match_symptoms

def test_match_symptoms_basic():
    """Test basic symptom matching."""
    text = "Patient has severe itching"
    lexicon = ["itching", "redness"]
    
    spans = match_symptoms(text, lexicon)
    
    assert len(spans) == 1
    assert spans[0]["text"] in ["itching", "severe itching"]
    assert spans[0]["label"] == "SYMPTOM"
    assert 0 <= spans[0]["confidence"] <= 1.0

def test_negation_detection():
    """Test negation detection."""
    text = "No itching reported"
    spans = match_symptoms(text, ["itching"])
    
    assert len(spans) == 1
    assert spans[0].get("negated", False) is True
```

### Test Coverage

- **Aim for ≥90% coverage** on new code
- **All functions must be tested**
- **Test edge cases**: empty inputs, unicode, very long texts
- **Test error handling**: invalid inputs, missing files

```bash
# Check coverage
pytest --cov=src --cov-report=term-missing

# Generate HTML report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

### Running Tests

```bash
# All tests
pytest -v

# Specific file
pytest tests/test_weak_label_core.py -v

# Specific test
pytest tests/test_weak_label_core.py::test_match_symptoms_basic -v

# Pattern matching
pytest -k "negation" -v

# Stop on first failure
pytest -x

# Show print statements
pytest -v -s
```

---

## Documentation

### API Documentation

Add docstrings to all public functions/classes:

```python
def new_function(arg: str) -> int:
    """
    One-line summary.
    
    Detailed explanation of what the function does,
    including edge cases and important notes.
    
    Args:
        arg: Description of argument
    
    Returns:
        Description of return value
    
    Examples:
        >>> new_function("test")
        4
    """
    return len(arg)
```

### User Guides

For new features, add user guide documentation:

- Create markdown file in `docs/user-guide/`
- Include examples and use cases
- Add to navigation in `mkdocs.yml`

**Example structure:**

```markdown
# Feature Name

## Overview
Brief introduction

## Usage
Basic examples

## Advanced Usage
Complex scenarios

## Best Practices
Recommendations

## Troubleshooting
Common issues
```

### Building Documentation

```bash
# Install docs dependencies
pip install -r docs-requirements.txt

# Serve locally
mkdocs serve
# Open http://localhost:8000

# Build static site
mkdocs build
# Output in site/
```

---

## Pull Request Process

### Before Submitting

- [ ] All tests pass locally (`pytest -v`)
- [ ] Code is formatted (`black src tests`)
- [ ] Imports are sorted (`isort src tests`)
- [ ] Documentation is updated
- [ ] Changelog is updated (if applicable)
- [ ] Branch is up-to-date with main

### PR Title Format

Use conventional commits format:

```
<type>(<scope>): <subject>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- test: Test additions/changes
- refactor: Code refactoring
- perf: Performance improvements
- chore: Maintenance tasks

Examples:
feat(weak_label): add bidirectional negation detection
fix(pipeline): handle empty text input gracefully
docs(annotation): add Label Studio tutorial
test(negation): add 12 new negation pattern tests
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
- Test 1 added
- Test 2 updated

## Checklist
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Changelog updated (if applicable)
```

### Review Process

1. **Automated Checks**: CI/CD runs tests on 6 configurations
2. **Code Review**: Maintainer reviews code, tests, docs
3. **Feedback**: Address review comments
4. **Approval**: Maintainer approves and merges

**Response time:** Expect initial review within 3-5 business days.

---

## Contribution Areas

### 1. Annotation

**Skills:** Domain knowledge, attention to detail  
**Tasks:**
- Annotate complaint texts in Label Studio
- Review and correct weak labels
- Participate in consensus annotation

**Impact:** Directly improves model quality

### 2. Documentation

**Skills:** Technical writing, markdown  
**Tasks:**
- Expand user guides and tutorials
- Add code examples
- Improve API documentation
- Write blog posts or walkthroughs

**Impact:** Makes project more accessible

### 3. Testing

**Skills:** Python, pytest  
**Tasks:**
- Add edge case tests
- Improve test coverage
- Write integration tests
- Performance benchmarks

**Impact:** Increases code reliability

### 4. Feature Development

**Skills:** Python, NLP, ML  
**Tasks:**
- Implement roadmap features
- Add new heuristics or filters
- Integrate new models
- Build tooling (CLI, API)

**Impact:** Expands functionality

### 5. Research & Experimentation

**Skills:** NLP, ML, evaluation  
**Tasks:**
- Experiment with new models (RoBERTa, ClinicalBERT)
- Tune hyperparameters
- Evaluate weak labeling approaches
- Write research reports

**Impact:** Advances state-of-the-art

---

## Community Guidelines

### Code of Conduct

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful, actionable feedback
- **Be inclusive**: Welcome diverse perspectives
- **Be patient**: Remember everyone is learning

### Communication

- **GitHub Issues**: Bug reports, feature requests
- **Pull Requests**: Code contributions, reviews
- **Discussions**: General questions, brainstorming

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Documentation acknowledgments

---

## Getting Help

### Stuck? Ask for help!

- **GitHub Discussions**: General questions
- **GitHub Issues**: Specific problems
- **Documentation**: [User guides](../user-guide/weak-labeling.md), [API reference](../api/config.md)

### Useful Resources

- [Roadmap](../about/roadmap.md) - Project plan
- [Testing Guide](testing.md) - Test infrastructure
- [Annotation Guide](../user-guide/annotation.md) - Annotation workflow
- [Weak Labeling Guide](../user-guide/weak-labeling.md) - Core functionality

---

## License

By contributing, you agree that your contributions will be licensed under the project's license (see `LICENSE` file).

---

*Thank you for contributing to SpanForge!* 🎉
