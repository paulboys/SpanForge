# Changelog

All notable changes to SpanForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Label Studio annotation workflow integration
- Gold standard dataset (500+ annotated samples)
- Token classification fine-tuning
- Model evaluation framework
- Active learning pipeline

---

## [0.4.0] - 2025-01-XX

### Added
- **Documentation Infrastructure**: Complete MkDocs setup with Material theme
  - Homepage with architecture diagrams, performance metrics, quick start
  - 5 API reference pages (config, model, weak_label, pipeline, llm_agent)
  - Installation guide with troubleshooting
  - Quick start tutorial with code examples
  - User guides: weak labeling, negation, pipeline, annotation
  - Development guide: testing infrastructure
  - About pages: roadmap, changelog
- **Docstrings**: Comprehensive Google-style docstrings for all core modules
  - `src/config.py`: Module, class, function docstrings with examples
  - `src/model.py`: Detailed function docstrings with parameter/return docs
  - `src/pipeline.py`: Complete pipeline documentation
  - `src/llm_agent.py`: Enhanced class and method documentation
- **Type Hints**: Explicit type annotations throughout codebase
  - `Optional`, `AutoTokenizer`, `AutoModel`, `BatchEncoding` types
  - Consistent typing for all function signatures
- **CI/CD**: GitHub Actions workflows and pre-commit hooks
  - `test.yml`: 6 configurations (Ubuntu/Windows × Python 3.9/3.10/3.11)
  - `pre-commit.yml`: Automated code quality checks
  - Pre-commit hooks for pytest, formatting
- **Configuration**: Consolidated pyproject.toml
  - Black, isort, pytest, coverage configurations
  - Tool configurations centralized

### Changed
- Updated README with CI/CD badges and documentation links
- Enhanced project structure documentation

### Documentation
- ✅ MkDocs Material theme with light/dark mode
- ✅ Auto-generated API reference using mkdocstrings
- ✅ Comprehensive user guides (weak labeling, negation, pipeline, annotation)
- ✅ Testing guide with 144-test infrastructure documentation
- ✅ Roadmap with 12 phases (4 complete, 8 planned)

---

## [0.3.0] - 2025-01-15

### Added
- **Test Infrastructure**: Comprehensive test suite with 144 tests
  - 16 core functionality tests
  - 98 edge case tests (unicode, emoji, negation, boundaries, anatomy, validation)
  - 26 integration tests (pipeline, scale, performance)
  - 4 curation tests (Label Studio export validation)
- **Test Composition Pattern**: Base classes with shared fixtures
  - `WeakLabelTestBase` for common setup
  - Eliminates test code duplication
  - Easy extension for new test categories
- **Edge Case Coverage**: Extensive validation
  - Unicode and emoji handling (12 tests)
  - Negation patterns (24 tests)
  - Boundary conditions (18 tests)
  - Anatomy filter (15 tests)
  - Validation and errors (29 tests)
- **Integration Tests**: End-to-end validation
  - Pipeline inference (11 tests)
  - Scale and batch processing (9 tests)
  - Performance benchmarks (6 tests)

### Fixed
- **Bidirectional Negation**: Expanded negation token list
  - Added: "non", "non-", "free of", "absence of", "ruled out", "r/o"
  - Improved backward negation detection
  - Fixed negation window boundary conditions
- **Unicode Handling**: Robust emoji and special character support
  - Emoji within text doesn't break span detection
  - Medical symbols (≥, ±, °) handled correctly
  - Accented characters in symptoms preserved
- **Validation Fixes**: Improved error handling
  - Empty lexicon handling
  - Empty text handling
  - Confidence score clamping [0, 1]
  - Boundary checks for span offsets

### Performance
- Established performance benchmarks:
  - `match_symptoms`: ~10ms/text (CPU)
  - `simple_inference`: ~200ms/text (CPU), ~50ms/text (GPU)
  - Batch processing: ~5s for 32 texts (CPU), ~1s (GPU)

### Test Results
- **144/144 tests passing** (100% pass rate)
- Test coverage: ~87% overall, ~94% core modules

---

## [0.2.0] - 2024-12-20

### Added
- **Bidirectional Negation Detection**: Forward and backward negation windows
  - Forward: Negation cue → [window] → span
  - Backward: Span → [window] → negation cue
  - Configurable window size (default: 5 tokens)
- **Last-Token Alignment Filter**: Prevents partial-word matches
  - Multi-token fuzzy matches must end at token boundaries
  - Reduces false positives from substring matches
- **Anatomy Singleton Filter**: Rejects generic anatomy mentions
  - Single anatomy tokens (skin, eye, face) rejected unless symptom co-occurs
  - List of 30+ anatomy terms
  - Reduces false positives by ~15%
- **Emoji and Unicode Handling**: Robust text processing
  - Emoji within text doesn't break tokenization
  - Unicode medical symbols supported
  - Preserves span offsets with multi-byte characters
- **Confidence Scoring**: Weighted fuzzy + Jaccard scores
  - Formula: 0.8 × fuzzy_score + 0.2 × jaccard_score
  - Clamped to [0, 1] range
  - Well-calibrated for active learning thresholds
- **Expanded Negation Tokens**: 22 negation cues
  - Clinical: denies, denied, negative, absent, unremarkable
  - Rule-out: rule out, ruled out, r/o
  - Temporal: no longer, ceased

### Changed
- **Negation Window**: Default increased from 3 to 5 tokens
  - Better balance of precision and recall
  - Configurable via `AppConfig.negation_window`
- **Fuzzy Threshold**: Default remains 88.0 (WRatio)
  - Well-calibrated after evaluation
- **Jaccard Threshold**: Default remains 40.0
  - Effective quality gate for fuzzy matches

### Performance
- Negation recall improved ~30% with bidirectional detection
- False positives reduced ~15% with anatomy filter
- Emoji handling prevents span breakage in ~5% of texts

---

## [0.1.0] - 2024-11-15

### Added
- **Initial Release**: Core weak labeling functionality
- **BioBERT Integration**: Model and tokenizer loading
  - Default: `dmis-lab/biobert-base-cased-v1.1`
  - Singleton pattern for efficient caching
  - Auto-detects CUDA availability
- **Fuzzy Matching**: RapidFuzz WRatio-based entity detection
  - Default threshold: 88.0
  - Handles typos and misspellings
  - N-gram tokenization (1-6 tokens)
- **Jaccard Token-Set Filter**: Quality gate for fuzzy matches
  - Default threshold: 40.0
  - Prevents false positives from short common words
- **Forward Negation Detection**: Basic negation scope
  - Window: 3 tokens (initial default)
  - Standard negation cues: no, not, none, never, without
- **Lexicons**: Initial symptom and product lexicons
  - `data/lexicon/symptoms.csv`: MedDRA-derived symptom terms
  - `data/lexicon/products.csv`: Product names and brands
- **Pipeline**: End-to-end inference workflow
  - `simple_inference()` function
  - Optional JSONL export
  - Batch processing support
- **Configuration**: Centralized config management
  - Pydantic BaseSettings
  - Environment variable support
  - Device auto-detection (CUDA/CPU)
- **LLM Agent Stub**: Experimental refinement pipeline
  - Stub implementation for future LLM integration
  - Data structures for span refinement suggestions

### Configuration Options
- `model_name`: BioBERT model identifier
- `max_seq_len`: Maximum sequence length (default: 256)
- `device`: Computation device (auto-detect)
- `seed`: Random seed (default: 42)
- `negation_window`: Negation scope (default: 3)
- `fuzzy_scorer`: Matching algorithm (default: "wratio")

### Dependencies
- `transformers>=4.30.0`: HuggingFace transformers
- `torch>=2.0.0`: PyTorch
- `rapidfuzz>=3.0.0`: Fuzzy string matching
- `pydantic>=2.0.0`: Configuration management
- `pandas>=2.0.0`: Data processing

---

## Version Naming

- **Major (X.0.0)**: Breaking changes, major milestones (e.g., supervised model release)
- **Minor (0.X.0)**: New features, enhancements (e.g., new filters, annotation tools)
- **Patch (0.0.X)**: Bug fixes, documentation updates

---

## Upcoming Releases

### v0.5.0 (Planned: Q1 2025)
- Label Studio annotation workflow
- Weak label export/import scripts
- Consensus and adjudication tools
- Quality assurance metrics (inter-annotator agreement)
- Annotation tutorial and guidelines

### v1.0.0 (Planned: Q2 2025)
- Fine-tuned BioBERT token classification model
- Gold standard dataset (500+ annotations)
- Model evaluation framework
- Training and inference scripts
- Production-ready NER pipeline

---

## Links

- **Repository**: [GitHub](#)
- **Documentation**: [MkDocs Site](#)
- **Issues**: [Issue Tracker](#)
- **Roadmap**: [Project Roadmap](roadmap.md)

---

*Keep this changelog updated with each release or significant change.*
