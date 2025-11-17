<div align="center">
  <img src="docs/assets/SpanForge.png" alt="SpanForge Logo" width="240" />
  <h1>SpanForge</h1>
  <p><em>From noisy complaints to canonical adverse event spans.</em></p>
  <p>
    <strong>Adverse Event NER Workflow</strong><br/>
    BioBERT + Weak Labeling + Human Annotation + Provenance Quality
  </p>
  <hr style="width:60%;border:0;border-top:1px solid #ccc" />
</div>

End-to-end workflow for extracting symptom-like adverse events and product mentions from consumer complaints using BioBERT plus lexicon/fuzzy heuristics, human curation in Label Studio, and provenance/quality reporting.

<details>
<summary><strong>Table of Contents</strong></summary>

1. Current Capability Snapshot
2. Quick Start
3. High-Level Workflow
4. Architecture Overview
5. Key Components
6. Weak Labeling Heuristics
7. Annotation Workflow
8. Provenance & Registry
9. Quality Metrics
10. Directory Layout
11. Testing
12. Roadmap
13. Contributing
14. Privacy & Compliance
15. Next Steps
16. Reference Docs
</details>

## Current Capability Snapshot
| Stage | Implemented | Notes |
|-------|-------------|-------|
| Config & Model Loading | ✅ | BioBERT `dmis-lab/biobert-base-cased-v1.1` via HF Transformers |
| Weak Labeling Heuristics | ✅ | Lexicon + fuzzy (WRatio≥0.88) + Jaccard≥40 + negation window=5 |
| Notebook Tutorial | ✅ | `scripts/Workbook.ipynb` full ingestion→gold workflow |
| Annotation Scripts | ✅ | Import, convert, quality, adjudicate, registry, CLI wrapper |
| Provenance Fields | ✅ | `source, annotator, revision, canonical, concept_id` |
| Quality Metrics | ✅ | Span density, label distribution, conflicts, (future kappa) |
| Gold Conversion Integrity Tests | ✅ | Overlap & canonical validation (tests dir) |
| Token Classification Head | ⏳ | Pending label inventory & gold expansion |
| Domain Adaptation (MLM) | ⏳ | Planned post initial gold batch |
| Baseline (RoBERTa) | ⏳ | Future comparative evaluation |

## Quick Start (PowerShell / Conda Recommended)
```powershell
conda env create -f environment.yml   # one-time (if present)
conda activate NER
pip install -r requirements.txt       # sync dependencies
python scripts/verify_env.py          # model + device sanity
pytest -q                             # run tests
```

Alternative venv:
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/verify_env.py
```

## High-Level Workflow
1. Ingest complaint texts (Notebook or script).
2. Apply weak labeling (`src/weak_label.py`) to produce spans with confidence + negation flags.
3. Export weak labels to JSONL (`data/output/...`).
4. Convert to Label Studio task format & import (`scripts/annotation/import_weak_to_labelstudio.py`).
5. Human annotation & correction in Label Studio (telemetry disabled).
6. Export annotated tasks → convert to gold (`scripts/annotation/convert_labelstudio.py`).
7. Run quality metrics (`scripts/annotation/quality_report.py`).
8. Register batch in provenance registry (`scripts/annotation/register_batch.py`).
9. (Planned) Fine-tune token classification model on BIO-tagged gold.

## Architecture Overview
```
Raw Text → Weak Labeler ─┬─► Weak JSONL → Label Studio Tasks → Human Annotation
                          │
                          └─► (Optional pre-annotation predictions)

Human Export → Gold Converter (+canonical +provenance) → Gold JSONL → Quality Metrics / Registry
                                                    └─► Future: BIO Tagging & Model Fine-Tune
```

## Key Components
- `src/config.py`: Central `AppConfig` (model name, device, max seq len, seed, negation window, fuzzy scorer).
- `src/model.py`: Loads BioBERT tokenizer/model; foundation for future token classification head.
- `src/weak_label.py`: Lexicon-driven + fuzzy/Jaccard gating, negation detection, confidence scoring.
- `src/pipeline.py`: Simple inference utilities (will host supervised tagging once head added).
- `scripts/Workbook.ipynb`: Hands-on, end-to-end educational walkthrough.
- Annotation scripts under `scripts/annotation/`: project init, import weak, convert gold, quality, adjudication, registry, CLI orchestration.

## Weak Labeling Heuristics (Summary)
- Fuzzy scorer: WRatio ≥ 0.88.
- Jaccard token-set threshold: ≥ 40.
- Confidence: `0.8*fuzzy + 0.2*jaccard` (clamped ≤1.0).
- Negation window: 5 tokens; ≥50% overlap triggers `negated=True`.
- Last-token alignment enforced for multi-token fuzzy matches.
- Anatomy single-token skip unless accompanied by explicit symptom phrase.

Details in `docs/heuristic.md`.

## Annotation Workflow (Label Studio)
1. Start Label Studio locally (disable telemetry).
2. Create project using `data/annotation/config/label_config.xml`.
3. Import tasks generated from weak labels.
4. Annotators label SYMPTOM / PRODUCT spans following boundary & negation rules (see `docs/annotation_guide.md`).
5. Export JSON; convert to gold with canonical + provenance.
6. Run quality & adjudication for conflicts.
7. Register batch for traceability.

Step-by-step guide: `docs/tutorial_labeling.md`.

## Provenance & Registry
Gold records enriched with: `source`, `annotator`, `revision`, `canonical`, optional `concept_id`.
`data/annotation/registry.csv` logs batch metadata (id, annotators, counts, notes) to audit evolution of training corpus.

## Quality Metrics
`quality_report.py` outputs JSON containing:
- Task count, mean spans/task.
- Label distribution & conflict list (overlapping differing labels).
- Annotator span counts; (future) pairwise agreement (IOU ≥0.5) & drift signals.

## Directory Layout (Condensed)
```
README.md
docs/
  overview.md
  annotation_guide.md
  tutorial_labeling.md
  heuristic.md
src/
  config.py
  model.py
  pipeline.py
  weak_label.py
scripts/
  verify_env.py
  Workbook.ipynb
  build_meddra_symptom_lexicon.py
  annotation/
    init_label_studio_project.py
    import_weak_to_labelstudio.py
    convert_labelstudio.py
    quality_report.py
    adjudicate.py
    register_batch.py
    cli.py
data/
  lexicon/ (symptoms.csv, products.csv)
  annotation/ (config/, exports/, raw/, reports/)
tests/ (forward, weak labeling integrity, curation integrity)
.github/copilot-instructions.md
```

## Testing
```powershell
pytest -q
```
Focus: forward pass, weak labeling correctness, gold conversion integrity. Extend with token classification evaluation once supervised labels available.

## Roadmap (Phases)
1. Bootstrap & Lexicon ✅
2. Weak Label Refinement ✅ (iterative)
3. Annotation & Curation (IN PROGRESS)
4. Gold Standard Assembly (next 100+ tasks) ⏳
5. Token Classification Fine-Tune ⏳
6. Domain Adaptation (MLM) ⏳
7. Baseline Comparison (RoBERTa) ⏳
8. Evaluation & Calibration ⏳
9. Educational Docs Expansion ✅ (initial) / ongoing
10. Active Learning Loop ⏳

## Contributing
1. Create env & install deps.
2. Run `scripts/verify_env.py` and tests.
3. Inspect lexicons; propose additions via PR (no licensed MedDRA raw data).
4. Use notebook or scripts to generate weak → tasks.
5. Perform annotation batch locally; convert + quality + register.
6. Submit focused PR referencing roadmap phase.

## Privacy & Compliance
- Do NOT commit raw complaint text containing PII (keep outside `data/` or use redacted versions).
- Telemetry disabled for Label Studio (`LABEL_STUDIO_DISABLE_TELEMETRY=1`).
- Canonical mapping strives for consistent terminology without storing licensed vocabularies.

## Next Steps
- Expand annotated corpus; measure agreement.
- Implement consensus / adjudication enhancements & kappa.
- Add token classification training script + BIO tagging conversion.
- Introduce evaluation harness (precision/recall/F1 on held-out gold).

## Reference Docs
See: `docs/overview.md`, `docs/annotation_guide.md`, `docs/tutorial_labeling.md`, `docs/heuristic.md`.

---
For a hands-on walkthrough open `scripts/Workbook.ipynb`.

## Public Landing Page
GitHub Pages (enable in repository settings → Pages → Source: `docs/` folder):

https://paulboys.github.io/SpanForge/

Features: responsive layout, dark mode, quick start, workflow cards, roadmap highlights.
