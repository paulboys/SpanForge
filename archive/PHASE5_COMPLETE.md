# Phase 5: Label Studio Integration & Annotation Workflow

**Status**: ✅ Complete (Previously Implemented)  
**Date**: November 27, 2025  
**Test Results**: 10/10 CLI tests passing (100%)

## Overview

Phase 5 was **already implemented** in the repository. This phase provides a complete annotation workflow using Label Studio for human curation of weak labels into gold-standard training data.

## What Was Discovered

All Phase 5 components were already in place:

### 1. Scripts (8 total) ✅
Located in `scripts/annotation/`:
- ✅ `init_label_studio_project.py` - Project bootstrap with telemetry disabled
- ✅ `import_weak_to_labelstudio.py` - Import weak labels as pre-annotations
- ✅ `convert_labelstudio.py` - Export + consensus logic
- ✅ `quality_report.py` - Per-annotator metrics + agreement
- ✅ `adjudicate.py` - Conflict resolution
- ✅ `register_batch.py` - Batch provenance tracking
- ✅ `refine_llm.py` - LLM refinement wrapper
- ✅ `cli.py` - Unified CLI interface

### 2. Configuration ✅
Located in `data/annotation/config/`:
- ✅ `label_config.xml` - Label Studio interface config
  - SYMPTOM (red #FF6B6B, hotkey `s`)
  - PRODUCT (teal #4ECDC4, hotkey `p`)
  - Word-level granularity
  - Colorblind-safe palette
  - NEGATED checkbox for negation tracking
  - Inline instructions and examples

### 3. Documentation ✅
Located in `docs/`:
- ✅ `annotation_guide.md` - Entity definitions, boundary rules, examples
- ✅ `tutorial_labeling.md` - Step-by-step Label Studio walkthrough
- ✅ `production_workflow.md` - End-to-end annotation process
- ✅ `production_evaluation.md` - Quality metrics and reporting

### 4. Directory Structure ✅
```
data/annotation/
├── config/
│   ├── label_config.xml       # Label Studio UI config
│   └── project_config.json    # Project metadata (created on bootstrap)
├── exports/                   # Converted gold JSONL
├── conflicts/                 # Adjudication workspace
├── reports/                   # Quality and evaluation reports
└── plots/                     # Visualization outputs
```

## Architecture

### Workflow Overview

```
1. Weak Labeling
   ↓
   weak_labels.jsonl (from src/weak_labeling package)
   
2. Optional LLM Refinement
   ↓
   llm_refined.jsonl (from src/llm_agent.py)
   
3. Import to Label Studio
   ↓
   CLI: bootstrap → import-weak → annotate
   
4. Human Annotation
   ↓
   Label Studio UI (local, telemetry off)
   
5. Export + Consensus
   ↓
   gold_standard.jsonl (majority vote, conflict resolution)
   
6. Quality Report
   ↓
   quality_report.json (agreement, coverage, drift)
   
7. Batch Registration
   ↓
   registry.csv (provenance tracking)
```

### CLI Commands

```powershell
# 1. Bootstrap Label Studio project
python scripts/annotation/cli.py bootstrap --project-name "Batch1"

# 2. Import weak labels
python scripts/annotation/cli.py import-weak \
    --input weak_labels.jsonl \
    --project-id 1

# 3. (Optional) Refine with LLM before import
python scripts/annotation/cli.py refine-llm \
    --weak weak_labels.jsonl \
    --output llm_refined.jsonl

# 4. After annotation, export and convert
python scripts/annotation/convert_labelstudio.py \
    --input labelstudio_export.json \
    --output gold_standard.jsonl \
    --consensus majority

# 5. Generate quality report
python scripts/annotation/cli.py quality \
    --gold gold_standard.jsonl \
    --out quality_report.json

# 6. Resolve conflicts
python scripts/annotation/cli.py adjudicate \
    --conflicts conflicts.json \
    --output adjudicated.jsonl

# 7. Register batch
python scripts/annotation/cli.py register \
    --batch-id batch1 \
    --gold gold_standard.jsonl
```

## Label Studio Configuration

### Entity Types

#### SYMPTOM (Red #FF6B6B, Hotkey `s`)
- Physical symptoms: rash, burning sensation, swelling
- Visual changes: redness, discoloration, blistering
- Sensory symptoms: itching, tingling, numbness
- Systemic reactions: nausea, dizziness, headache
- Medical diagnoses: allergic reaction, dermatitis

#### PRODUCT (Teal #4ECDC4, Hotkey `p`)
- Product names: CeraVe Facial Moisturizer
- Categories: facial cream, shampoo, lipstick
- Generic: cream, lotion, serum
- Supplements: vitamin D, fish oil

#### NEGATED Checkbox
- Checked when symptom is negated (e.g., "no rash")
- Still annotate the symptom span
- Supports separate negation modeling

### UI Features
- Word-level granularity (prevents character-level misalignment)
- Inline instructions visible during annotation
- Examples embedded in interface
- Colorblind-safe palette (WCAG AAA compliant)
- Keyboard shortcuts (s/p) for speed

## Annotation Guidelines

### Boundary Rules

| Rule | Example | Notes |
|------|---------|-------|
| Full clinical phrase | `burning sensation` ✓ | Not just `burning` |
| Exclude punctuation | `redness` ✓ | Not `redness.` |
| Exclude determiners | `rash` ✓ | Not `the rash` |
| Stop at conjunctions | `redness`, `swelling` ✓ | Not `redness and swelling` |
| Include modifiers | `severe itching` ✓ | Severity is informative |

### Negation Handling

```
Text: "no redness occurred"
Annotation: 
  - Span: "redness" (SYMPTOM)
  - NEGATED: ✓ checked

Rationale: Model needs both positive and negative examples
```

### Quality Checklist

Before submission:
- [ ] All symptoms and products annotated?
- [ ] Boundaries correct (no punctuation, determiners)?
- [ ] Full phrases selected?
- [ ] Negated symptoms marked with NEGATED?
- [ ] Generic anatomy skipped unless with symptom?
- [ ] No overlapping spans with different labels?

## Testing

### CLI Tests (10 total)
```
tests/annotation/test_cli.py::TestCLI::test_bootstrap_subcommand        PASSED
tests/annotation/test_cli.py::TestCLI::test_import_weak_subcommand      PASSED
tests/annotation/test_cli.py::test_quality_subcommand                   PASSED
tests/annotation/test_cli.py::test_adjudicate_subcommand                PASSED
tests/annotation/test_cli.py::test_register_subcommand                  PASSED
tests/annotation/test_cli.py::test_refine_llm_subcommand                PASSED
tests/annotation/test_cli.py::test_evaluate_llm_subcommand              PASSED
tests/annotation/test_cli.py::test_plot_metrics_subcommand              PASSED
tests/annotation/test_cli.py::test_cli_help                             PASSED
tests/annotation/test_cli.py::test_invalid_subcommand                   PASSED
```

### Integration Tests
```
tests/annotation/test_evaluate_llm_refinement.py     27 tests passing
tests/annotation/test_plot_llm_metrics.py             14 tests (skipped - viz)
```

## Privacy & Security

### Local-Only Operation
- ✅ `LABEL_STUDIO_DISABLE_TELEMETRY=1` (no analytics)
- ✅ Local data storage (no cloud uploads)
- ✅ De-identified data only (no PII in complaints)
- ✅ `.gitignore` for raw data

### De-identification
Handled upstream in CAERS download:
- Names, addresses, emails already redacted by FDA
- Additional PII removal in `scripts/caers/download_caers.py`
- Raw complaints stored in `data/caers/raw/` (gitignored)

## Consensus & Adjudication

### Majority Vote Strategy
```python
# convert_labelstudio.py --consensus majority
- Identical (start, end, label): Keep if ≥50% agree
- Tie-breaker: Longest span
- Label conflict on overlap: Escalate to adjudication
```

### Conflict Resolution
```python
# adjudicate.py
- Longest span strategy (default)
- Manual escalation for label conflicts
- Conflicts saved to data/annotation/conflicts/
```

### Quality Metrics
```python
# quality_report.py
- Per-annotator span counts
- Label distribution
- Overlap conflicts (different labels)
- Pairwise agreement (IOU ≥0.5)
- Cohen's kappa (if multiple annotators)
```

## Batch Registry

### Provenance Tracking
```csv
# data/annotation/registry.csv
batch_id,n_tasks,annotators,revision,consensus_done,created_at
batch1,100,annotator1|annotator2,v1,true,2025-11-27T10:00:00
batch2,150,annotator1,v1,false,2025-11-27T11:00:00
```

### Fields
- `batch_id`: Unique identifier
- `n_tasks`: Number of annotated texts
- `annotators`: Pipe-separated list
- `revision`: Version number
- `consensus_done`: Boolean flag
- `created_at`: ISO timestamp

## Example Workflow

### Setup (One-Time)
```powershell
# Install Label Studio
pip install label-studio

# Start Label Studio
cd C:\Users\User\Documents\NER
python scripts/annotation/cli.py bootstrap --project-name "SpanForge-Batch1"

# This will:
# 1. Set LABEL_STUDIO_DISABLE_TELEMETRY=1
# 2. Start server on port 8080
# 3. Create project with label_config.xml
# 4. Save project_config.json
```

### Import Tasks
```powershell
# Option A: Import weak labels directly
python scripts/annotation/cli.py import-weak \
    --input data/caers/cosmetics_1000.jsonl \
    --project-id 1

# Option B: Refine with LLM first (recommended)
python scripts/annotation/cli.py refine-llm \
    --weak data/caers/cosmetics_1000.jsonl \
    --output data/caers/cosmetics_1000_refined.jsonl

python scripts/annotation/cli.py import-weak \
    --input data/caers/cosmetics_1000_refined.jsonl \
    --project-id 1
```

### Annotate
```
1. Open browser: http://localhost:8080/projects/1
2. Click "Label" on any task
3. Select spans (drag to select words)
4. Press 's' for SYMPTOM (red) or 'p' for PRODUCT (teal)
5. Check NEGATED if symptom is negated
6. Submit
```

### Export & Process
```powershell
# 1. Export from Label Studio UI: 
#    Settings -> Export -> JSON

# 2. Convert to gold JSONL
python scripts/annotation/convert_labelstudio.py \
    --input labelstudio_export.json \
    --output data/annotation/exports/batch1_gold.jsonl \
    --consensus majority

# 3. Generate quality report
python scripts/annotation/cli.py quality \
    --gold data/annotation/exports/batch1_gold.jsonl \
    --out data/annotation/reports/batch1_quality.json

# 4. Register batch
python scripts/annotation/cli.py register \
    --batch-id batch1 \
    --gold data/annotation/exports/batch1_gold.jsonl
```

## Success Criteria (Achieved)

✅ **All scripts implemented and tested**  
✅ **CLI interface with 8 subcommands**  
✅ **Label Studio config with colorblind-safe palette**  
✅ **Privacy-safe setup (telemetry disabled)**  
✅ **Comprehensive documentation**  
✅ **Consensus and adjudication logic**  
✅ **Quality metrics and reporting**  
✅ **Batch registry for provenance**

## Future Enhancements (Optional)

### Phase 5.5 (Potential)
- [ ] Active learning integration (select most informative samples)
- [ ] Real-time agreement monitoring dashboard
- [ ] Automated boundary correction suggestions
- [ ] Multi-language support (Spanish, French)
- [ ] Integration with external annotation platforms (Prodigy, Doccano)

### Phase 6 (Next)
- [ ] Gold standard assembly (compile 1000+ annotated samples)
- [ ] Token classification fine-tuning (BioBERT + classification head)
- [ ] Model evaluation (precision/recall/F1)
- [ ] Error analysis and iterative improvement

## Related Documentation

- **Annotation Guide**: `docs/annotation_guide.md`
- **Tutorial**: `docs/tutorial_labeling.md`
- **Production Workflow**: `docs/production_workflow.md`
- **Evaluation Guide**: `docs/production_evaluation.md`
- **Phase 4 (Refactoring)**: `docs/development/phase4_refactoring.md`

## Questions?

For annotation workflow questions:
1. Check CLI help: `python scripts/annotation/cli.py --help`
2. Review annotation guide: `docs/annotation_guide.md`
3. Follow tutorial: `docs/tutorial_labeling.md`
4. Run tests: `pytest tests/annotation/ -v`

---

**Summary**: Phase 5 was already complete with 8 scripts, comprehensive documentation, privacy-safe Label Studio integration, and full CLI workflow. All 10 CLI tests passing. Ready for production annotation batches.
