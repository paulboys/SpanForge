# Contributing to SpanForge

Thank you for considering contributing to SpanForge! This document provides guidelines and workflows for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing Requirements](#testing-requirements)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

This project adheres to a code of conduct that promotes:
- Respectful and inclusive communication
- Constructive feedback
- Focus on project goals
- Recognition of contributions

## Getting Started

### Prerequisites

- Python 3.9, 3.10, or 3.11
- Git
- Virtual environment tool (conda, venv, or virtualenv)

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/paulboys/SpanForge.git
cd SpanForge

# Create virtual environment
conda create -n spanforge python=3.10
conda activate spanforge

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-cov pre-commit ruff black isort mypy

# Setup pre-commit hooks
pre-commit install

# Verify setup
python scripts/verify_env.py
pytest tests/ -v
```

## Development Workflow

### Branch Strategy

- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features or enhancements
- `fix/*`: Bug fixes
- `refactor/*`: Code refactoring
- `test/*`: Test improvements

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Commit Guidelines

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `docs`: Documentation updates
- `style`: Code style changes (formatting)
- `perf`: Performance improvements
- `chore`: Build/tool configuration

**Example:**
```
feat(negation): Add bidirectional negation window detection

Enhanced negation detection to handle both forward ("no itching")
and backward ("itching absent") patterns. Added 10 new negation
cues including clinical terms and resolution indicators.

Fixes #42
```

## Testing Requirements

### Running Tests

```bash
# Full suite
pytest tests/ -v

# Specific categories
pytest tests/test_weak_label.py -v
pytest tests/edge_cases/ -v
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Test Categories

1. **Core Tests** (`tests/test_*.py`): Unit tests for main functionality
2. **Edge Cases** (`tests/edge_cases/`): 98 parametrized boundary tests
3. **Integration** (`tests/integration/`): End-to-end pipeline tests

### Writing Tests

**Guidelines:**
- Use descriptive test names: `test_negation_cue_detection_with_absent`
- Follow AAA pattern: Arrange, Act, Assert
- Use composition helpers: `SpanAsserter`, `OverlapChecker`, `IntegrityValidator`
- Parametrize when testing multiple inputs: `@pytest.mark.parametrize`
- Add docstrings explaining test purpose

**Example:**
```python
import pytest
from tests.base import WeakLabelTestBase

class TestNegationPatterns(WeakLabelTestBase):
    """Test negation detection with various patterns."""
    
    @pytest.mark.parametrize("text,symptom,should_negate", [
        ("No itching", "itching", True),
        ("Itching absent", "itching", True),
        ("Reports itching", "itching", False),
    ])
    def test_negation_cue_detection(self, text, symptom, should_negate):
        """Parametrized negation cue detection."""
        spans = self.weak_label_fn(text)
        symptom_spans = [s for s in spans if symptom in s.text.lower()]
        
        if should_negate:
            assert any(s.negated for s in symptom_spans)
        else:
            assert all(not s.negated for s in symptom_spans)
```

### Coverage Requirements

- Minimum: 80% coverage on new code
- Target: 90% overall coverage
- Focus on critical paths: weak labeling, negation detection, span merging

## Code Style

### Formatting

**Tools:**
- **black**: Line length 100, Python 3.10 target
- **isort**: Profile "black", imports sorted
- **ruff**: Linter with auto-fix enabled

**Run formatters:**
```bash
black src/ tests/ scripts/
isort src/ tests/ scripts/
ruff check --fix src/ tests/ scripts/
```

### Type Hints

- Use type hints for function signatures
- Import from `typing` module
- Optional for local variables
- Use `# type: ignore` sparingly with explanation

**Example:**
```python
from typing import List, Tuple, Optional

def detect_negated_regions(
    text: str, 
    window: int = 5
) -> List[Tuple[int, int]]:
    """Detect character ranges following negation cues."""
    ...
```

### Documentation

- **Docstrings**: Use Google style
- **Comments**: Explain WHY, not WHAT
- **README**: Update for user-facing changes
- **CHANGELOG**: Document all changes

**Docstring Example:**
```python
def match_symptoms(
    text: str,
    lexicon: List[LexiconEntry],
    fuzzy_threshold: float = 88.0,
) -> List[Span]:
    """Match symptom terms from lexicon against text.
    
    Performs exact and fuzzy matching with negation detection,
    anatomy gating, and confidence scoring.
    
    Args:
        text: Input text to analyze
        lexicon: List of symptom lexicon entries
        fuzzy_threshold: Minimum WRatio score (0-100)
    
    Returns:
        List of detected symptom spans with metadata
    """
    ...
```

## Pull Request Process

### Before Submitting

1. ✅ Run full test suite: `pytest tests/ -v`
2. ✅ Check formatting: `black --check .` and `isort --check-only .`
3. ✅ Run linter: `ruff check .`
4. ✅ Update documentation if needed
5. ✅ Add tests for new features
6. ✅ Update CHANGELOG.md

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass locally
- [ ] Added new tests for changes
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Commented complex code
- [ ] Documentation updated
- [ ] No new warnings
```

### Review Process

1. **Automated Checks**: CI must pass (all tests, linting)
2. **Code Review**: At least one approval required
3. **Discussion**: Address reviewer comments
4. **Merge**: Squash and merge to maintain clean history

## Issue Reporting

### Bug Reports

**Include:**
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Minimal reproducible example

**Template:**
```markdown
**Environment:**
- OS: Windows 11
- Python: 3.10.14
- SpanForge version: main branch (commit abc123)

**Description:**
Brief description of bug

**Steps to Reproduce:**
1. Load lexicon from...
2. Run weak_label on text...
3. Observe error...

**Expected Behavior:**
Should detect "itching" with confidence 1.0

**Actual Behavior:**
Returns empty span list

**Error Message:**
```
KeyError: 'canonical'
```
```

### Feature Requests

**Include:**
- Use case description
- Proposed solution
- Alternative approaches considered
- Willingness to implement

## Project Structure

```
SpanForge/
├── src/               # Core source code
│   ├── model.py       # BioBERT model loading
│   ├── weak_label.py  # Weak labeling logic
│   ├── pipeline.py    # End-to-end pipeline
│   └── config.py      # Configuration management
├── tests/             # Test suite
│   ├── test_*.py      # Core unit tests
│   ├── edge_cases/    # Edge case tests
│   ├── integration/   # Integration tests
│   ├── base.py        # Test base classes
│   └── assertions.py  # Composition helpers
├── scripts/           # Utility scripts
├── data/              # Lexicons and output
├── docs/              # Documentation
└── .github/           # CI/CD workflows
```

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Search existing issues on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Contact**: Open an issue for maintainer contact

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- GitHub contributors page

Thank you for contributing! 🎉
