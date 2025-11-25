# CI/CD Documentation

## Overview

SpanForge uses GitHub Actions for continuous integration and delivery. The CI/CD pipeline runs automated tests, linting, and quality checks on every push and pull request.

## Workflows

### 1. Test Suite (`test.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual dispatch via GitHub UI

**Jobs:**

#### Test Matrix
- **Operating Systems**: Ubuntu, Windows
- **Python Versions**: 3.9, 3.10, 3.11
- **Total Configurations**: 6 (2 OS × 3 Python versions)

**Test Stages:**
1. **Environment Verification**: Runs `scripts/verify_env.py`
2. **Core Tests**: Unit tests for weak labeling, model loading, pipeline
3. **Edge Case Tests**: 98 parametrized edge case tests
4. **Integration Tests**: End-to-end pipeline and scale tests
5. **Coverage Report**: Generates coverage with pytest-cov

**Coverage Upload:**
- Only on Ubuntu + Python 3.10 configuration
- Uploads to Codecov for tracking over time
- Failure doesn't block CI (informational only)

#### Lint Job
- **Checks**: ruff, black, isort
- Runs on Ubuntu + Python 3.10
- Non-blocking (continue-on-error: true)

#### Build Check
- Validates package can be built with `python -m build`
- Checks distribution with `twine check`
- Non-blocking

### 2. Pre-commit Checks (`pre-commit.yml`)

**Triggers:**
- Pull requests only

**Checks:**
- Trailing whitespace removal
- End-of-file fixer
- YAML/JSON validation
- Large file detection (max 1MB)
- Private key detection
- Code formatting (black, isort, ruff)
- Type checking (mypy, non-blocking)

## Local Development

### Setup Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

Now hooks run automatically on `git commit`. To run manually:

```bash
pre-commit run --all-files
```

### Run Tests Locally

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific category
pytest tests/edge_cases/ -v
pytest tests/integration/ -v

# Parallel execution (faster)
pytest tests/ -n auto
```

### Linting

```bash
# Check only
ruff check src/ tests/ scripts/
black --check src/ tests/ scripts/
isort --check-only src/ tests/ scripts/

# Auto-fix
ruff check --fix src/ tests/ scripts/
black src/ tests/ scripts/
isort src/ tests/ scripts/
```

## Badge Integration

Add to README.md:

```markdown
![Test Suite](https://github.com/paulboys/SpanForge/actions/workflows/test.yml/badge.svg)
![Pre-commit](https://github.com/paulboys/SpanForge/actions/workflows/pre-commit.yml/badge.svg)
[![codecov](https://codecov.io/gh/paulboys/SpanForge/branch/main/graph/badge.svg)](https://codecov.io/gh/paulboys/SpanForge)
```

## Pull Request Requirements

Before merging to `main`, ensure:

1. ✅ All test jobs pass (144/144 tests)
2. ✅ No ruff/black/isort violations
3. ✅ Coverage maintained or improved
4. ✅ Pre-commit checks pass
5. ✅ Documentation updated if needed

## Performance Benchmarks

**Current CI Times (Ubuntu + Python 3.10):**
- Environment setup: ~30s
- Test execution: ~18s
- Coverage generation: ~5s
- **Total**: ~1-2 minutes per job

**Local Performance:**
- Full suite: ~17s (144 tests)
- Edge cases: ~1.3s (98 tests)
- Integration: ~44s (26 tests, includes 1000-doc stress test)

## Troubleshooting

### Tests Pass Locally but Fail in CI

**Check:**
- OS-specific path separators (`os.path.join` vs `pathlib.Path`)
- Line endings (CRLF vs LF)
- Timezone dependencies
- File permissions

### Coverage Drops Unexpectedly

**Causes:**
- New untested code paths
- Removed tests without removing code
- Conditional imports not triggered in test environment

**Fix:**
- Add tests for new features
- Review coverage HTML report: `htmlcov/index.html`
- Mark uncoverable lines: `# pragma: no cover`

### Pre-commit Hook Slow

**Solutions:**
- Skip mypy locally: `SKIP=mypy git commit`
- Run hooks in parallel: `pre-commit run --all-files --show-diff-on-failure`
- Update hooks: `pre-commit autoupdate`

## Future Enhancements

### Planned Additions:
1. **Nightly Builds**: Extended stress tests (10k documents)
2. **Performance Regression**: Track inference time over commits
3. **Security Scanning**: Bandit, safety checks
4. **Documentation**: Auto-generate API docs with Sphinx
5. **Release Automation**: Tag-triggered PyPI publish

### Advanced Coverage Goals:
- Target: 90% coverage on `src/`
- Branch coverage tracking
- Mutation testing with `mutmut`

## Maintenance

### Updating Dependencies

```bash
# Update GitHub Actions
# Edit .github/workflows/*.yml, bump action versions

# Update pre-commit hooks
pre-commit autoupdate

# Update Python dependencies
pip list --outdated
# Update requirements.txt accordingly
```

### Monitoring

- **GitHub Actions Dashboard**: Monitor workflow runs
- **Codecov Dashboard**: Track coverage trends
- **Dependabot**: Enable for automated dependency updates

---

**Last Updated**: Phase 4 (Nov 25, 2025)
**Status**: ✅ All workflows configured and tested
