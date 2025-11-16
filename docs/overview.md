# Overview

End-to-end Adverse Event NER pipeline integrating weak labeling, BioBERT embeddings, human annotation, and quality/provenance tracking. Designed to iteratively refine span quality before supervised fine-tuning.

## Data Flow
```
Raw Text → Weak Labeler → Weak JSONL → Label Studio Tasks → Human Annotation → Export
	→ Gold Converter (+canonical +provenance) → Gold JSONL → Quality Metrics / Registry
								 → (Planned) BIO Tagging → Token Classifier → Evaluation
```

## Core Modules
- `src/config.py`: Centralized configuration (model, device, heuristic thresholds, negation window, scorer).
- `src/model.py`: BioBERT base encoder loader (future token classification head attachment point).
- `src/weak_label.py`: Lexicon + fuzzy + Jaccard gating, negation detection, confidence scoring.
- `src/pipeline.py`: Light inference utilities; will wrap supervised prediction later.
- `scripts/annotation/*.py`: Operational scripts for project init, import, convert, quality, adjudication, registry, CLI.
- `scripts/Workbook.ipynb`: Educational notebook walking through ingestion → gold.

## Entity Types
- `SYMPTOM`: Physiological or subjective adverse effects reported by consumer.
- `PRODUCT`: Product names, formulations, or product category terms referenced.

## Canonical Mapping
Surface forms mapped to canonical normalized terms (symptoms/products) via curated lexicons (`data/lexicon/`). This stabilizes vocabulary for aggregation and downstream modeling.

## Provenance Fields
Gold output includes: `source`, `annotator`, `revision`, `canonical`, optional `concept_id` for traceability and reproducibility.

## Heuristic Summary
- Fuzzy WRatio ≥ 0.88
- Jaccard token-set ≥ 40
- Confidence = `0.8*fuzzy + 0.2*jaccard` (≤1.0)
- Negation window: 5 tokens (≥50% overlap → negated)
- Single generic anatomy token skip unless symptom co-occurs
- Last-token alignment for multi-token fuzzy spans

Details: `heuristic.md`.

## Quality Metrics
`quality_report.py` computes span density, label distribution, conflicts, annotator counts; planned kappa & drift metrics.

## Design Principles
- Pure functions; explicit dependency passing
- Reproducible thresholds in config (tunable)
- Incremental, test-supported evolution (weak → gold → supervised)
- Clear audit trail via registry and provenance fields

## Privacy & Compliance
No raw proprietary complaints committed. Annotation performed locally with telemetry disabled. Only de-identified / approved text permitted in repository.

## Roadmap (Condensed)
1. Lexicon & Weak Labeling (complete / iterative)
2. Annotation & Curation (in progress)
3. Gold Assembly & Expansion
4. Token Classification Fine-Tune
5. Domain Adaptation (MLM)
6. Baseline & Evaluation (RoBERTa)
7. Active Learning Loop

## References
See `annotation_guide.md`, `tutorial_labeling.md`, and `heuristic.md` for deeper detail.
