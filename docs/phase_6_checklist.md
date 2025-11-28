# Phase 6 Quick Start Checklist

**Goal**: Assemble 500-1000 gold standard annotations  
**Status**: ⏸️ READY TO START  
**Estimated Time**: 20-30 hours annotation + 5-10 hours QA

---

## 🎯 Phase 6 Objectives

- [ ] Batch 001: 100 cosmetics complaints (calibration batch)
- [ ] Batch 002-005: 400 additional complaints (scale to 500 total)
- [ ] Optional Batches 006-010: Scale to 1000 total
- [ ] Inter-annotator agreement >0.75 (IOU ≥ 0.5)
- [ ] Canonical coverage >90%
- [ ] All batches registered with provenance

---

## ✅ Prerequisites (Complete)

✅ Weak labeling infrastructure (531/531 tests passing)  
✅ LLM refinement agent (OpenAI/Anthropic/stub support)  
✅ Evaluation metrics (10 functions implemented)  
✅ Annotation scripts (12 scripts in `scripts/annotation/`)  
✅ Documentation (`docs/annotation_guide.md`, `docs/production_workflow.md`)  
✅ CAERS data pipeline (666K+ complaints available)  
✅ Batch preparation script (NEW: `prepare_production_batch.py`)  
✅ Phase 6 implementation guide (NEW: `docs/phase_6_gold_standard.md`)

---

## 📋 Implementation Checklist

### Week 1: Batch 001 (Calibration)

#### Day 1: Preparation
- [ ] Download CAERS cosmetics data (1000 complaints)
  ```powershell
  python scripts/caers/download_caers.py --output data/caers/cosmetics_1000.jsonl --categories cosmetics --limit 1000 --min-spans 1
  ```
- [ ] Prepare batch 001 (100 tasks, stratified sampling)
  ```powershell
  python scripts/annotation/prepare_production_batch.py --input data/caers/cosmetics_1000.jsonl --output data/annotation/exports/batch_001.jsonl --n-tasks 100 --strategy stratified --min-spans 1 --max-text-len 500 --stats-output data/annotation/reports/batch_001_stats.json
  ```
- [ ] Review batch statistics
  ```powershell
  Get-Content data/annotation/reports/batch_001_stats.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
  ```

#### Day 2: Optional LLM Refinement
- [ ] **Option A**: Use OpenAI GPT-4 (requires API key + budget)
  ```powershell
  $env:OPENAI_API_KEY = "sk-..."
  python scripts/annotation/refine_llm.py --input data/annotation/exports/batch_001.jsonl --output data/annotation/exports/batch_001_refined.jsonl --provider openai --model gpt-4o-mini --temperature 0.1
  ```
- [ ] **Option B**: Use Anthropic Claude (requires API key + budget)
  ```powershell
  $env:ANTHROPIC_API_KEY = "sk-ant-..."
  python scripts/annotation/refine_llm.py --input data/annotation/exports/batch_001.jsonl --output data/annotation/exports/batch_001_refined.jsonl --provider anthropic --model claude-3-5-sonnet-20241022 --temperature 0.1
  ```
- [ ] **Option C**: Skip LLM (use weak labels directly)
  ```powershell
  Copy-Item data/annotation/exports/batch_001.jsonl data/annotation/exports/batch_001_refined.jsonl
  ```

#### Day 3-4: Human Annotation (3-5 hours)
- [ ] Initialize Label Studio project (first time only)
  ```powershell
  python scripts/annotation/init_label_studio_project.py --project-name "SpanForge Gold Standard" --config data/annotation/config/label_config.xml
  ```
- [ ] Import batch to Label Studio
  ```powershell
  python scripts/annotation/import_weak_to_labelstudio.py --input data/annotation/exports/batch_001_refined.jsonl --project-id 1 --confidence-filter 0.6
  ```
- [ ] Launch Label Studio: `label-studio start`
- [ ] Navigate to http://localhost:8080
- [ ] Annotate 100 tasks (~2-3 min/task = 3-5 hours total)
  - ✅ Correct span boundaries
  - ✅ Add missing spans
  - ✅ Remove false positives
  - ✅ Verify negation flags
  - ✅ Confirm canonical mappings

#### Day 5: Export & Validation (1-2 hours)
- [ ] Export from Label Studio UI: Settings → Export → JSON
- [ ] Convert to gold format
  ```powershell
  python scripts/annotation/convert_labelstudio.py --input data/annotation/exports/label_studio_export.json --output data/annotation/exports/batch_001_gold.jsonl --consensus majority --min-agree 1
  ```
- [ ] Generate quality report
  ```powershell
  python scripts/annotation/quality_report.py --gold data/annotation/exports/batch_001_gold.jsonl --output data/annotation/reports/batch_001_quality.md
  ```
- [ ] Evaluate weak → LLM → gold
  ```powershell
  python scripts/annotation/evaluate_llm_refinement.py --weak data/annotation/exports/batch_001.jsonl --refined data/annotation/exports/batch_001_refined.jsonl --gold data/annotation/exports/batch_001_gold.jsonl --output data/annotation/reports/batch_001_evaluation.json --markdown --stratify label confidence
  ```
- [ ] Generate visualizations
  ```powershell
  python scripts/annotation/plot_llm_metrics.py --report data/annotation/reports/batch_001_evaluation.json --output-dir data/annotation/plots/batch_001/ --formats png pdf --plots all
  ```
- [ ] Review metrics (target: IOU +8-15%, F1 >0.85, exact match 70-85%)

#### Day 6: Register & Reflect (30 min)
- [ ] Register batch in provenance log
  ```powershell
  python scripts/annotation/register_batch.py --gold-file data/annotation/exports/batch_001_gold.jsonl --batch-id batch_001 --n-tasks 100 --annotators "your_name" --revision 1 --notes "Initial calibration batch - cosmetics complaints"
  ```
- [ ] Review quality report for edge cases
- [ ] Update `docs/annotation_guide.md` with lessons learned
- [ ] Commit to git
  ```powershell
  git add data/annotation/exports/batch_001_gold.jsonl data/annotation/reports/ data/annotation/plots/
  git commit -m "feat(annotation): complete batch 001 gold standard (100 tasks)"
  git push origin main
  ```

---

### Week 2-3: Batch 002-005 (Scale to 500)

#### Batch 002 (Cosmetics - Refined Guidelines)
- [ ] Prepare batch 002 (100 tasks, exclude batch 001 IDs)
- [ ] Optional LLM refinement
- [ ] Annotate (3-5 hours)
- [ ] Export, validate, evaluate
- [ ] Register batch

#### Batch 003 (Supplements - Category Expansion)
- [ ] Download supplements data
  ```powershell
  python scripts/caers/download_caers.py --output data/caers/supplements_1000.jsonl --categories supplements --limit 1000 --min-spans 1
  ```
- [ ] Prepare batch 003 (100 tasks)
- [ ] Optional LLM refinement
- [ ] Annotate (3-5 hours)
- [ ] Export, validate, evaluate
- [ ] Register batch

#### Batch 004 (Personal Care - Diverse Language)
- [ ] Download personal care data
- [ ] Prepare batch 004 (100 tasks)
- [ ] Optional LLM refinement
- [ ] Annotate (3-5 hours)
- [ ] Export, validate, evaluate
- [ ] Register batch

#### Batch 005 (Mixed - Balanced Sampling)
- [ ] Prepare batch 005 (100 tasks, `--strategy balanced`)
- [ ] Optional LLM refinement
- [ ] Annotate (3-5 hours)
- [ ] Export, validate, evaluate
- [ ] Register batch

---

### Week 4: Quality Assurance & Phase 6 Completion

#### Aggregate Analysis
- [ ] Combine all gold batches
  ```powershell
  Get-Content data/annotation/exports/batch_00*.jsonl | Out-File data/annotation/exports/gold_all_500.jsonl
  ```
- [ ] Generate combined quality report
  ```powershell
  python scripts/annotation/quality_report.py --gold data/annotation/exports/gold_all_500.jsonl --output data/annotation/reports/gold_all_500_quality.md
  ```
- [ ] Calculate inter-annotator agreement (if multiple annotators)
- [ ] Verify canonical coverage >90%
- [ ] Check label distribution (target: 85% SYMPTOM / 15% PRODUCT)

#### Phase 6 Completion Criteria
- [ ] ✅ Total gold tasks ≥500 (1000 ideal)
- [ ] ✅ Inter-annotator agreement >0.75 (IOU ≥ 0.5)
- [ ] ✅ Canonical coverage >90%
- [ ] ✅ All batches registered in `registry.csv`
- [ ] ✅ Quality reports reviewed and approved
- [ ] ✅ Annotation guidelines finalized
- [ ] ✅ All integrity tests passing
  ```powershell
  pytest tests/test_curation_integrity.py -v
  ```

#### Documentation & Handoff
- [ ] Update `README.md` with Phase 6 completion status
- [ ] Commit all gold files, reports, plots
- [ ] Tag release: `git tag v0.6.0 -m "Phase 6 Complete: Gold Standard Assembly"`
- [ ] Push tag: `git push origin v0.6.0`
- [ ] Prepare for Phase 7 (Token Classification Fine-Tuning)

---

## 🚀 Optional: Scale to 1000 (Batches 006-010)

If time and resources allow, continue to 1000 total annotations:

- [ ] Batch 006-010: 500 additional tasks (100 each)
- [ ] Maintain stratification and category balance
- [ ] Monitor annotator throughput and agreement
- [ ] Adjust guidelines based on cumulative quality reports

**Benefits of 1000 annotations**:
- More robust model training
- Better evaluation of rare labels
- Stronger baseline for active learning
- Publication-ready dataset size

---

## 📊 Success Metrics Dashboard

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Batch 001** | 100 tasks | 0 | 🔴 Not started |
| **Batch 002** | 100 tasks | 0 | 🔴 Not started |
| **Batch 003** | 100 tasks | 0 | 🔴 Not started |
| **Batch 004** | 100 tasks | 0 | 🔴 Not started |
| **Batch 005** | 100 tasks | 0 | 🔴 Not started |
| **Total Gold** | 500 | 0 | 🔴 Not started |
| **SYMPTOM Spans** | 400-900 | 0 | 🔴 Not started |
| **PRODUCT Spans** | 100-300 | 0 | 🔴 Not started |
| **Agreement (IOU)** | >0.75 | - | 🟡 Pending |
| **Canonical Coverage** | >90% | - | 🟡 Pending |
| **Avg Time/Task** | <3 min | - | 🟡 Pending |

**Update this dashboard after each batch!**

---

## 🛠️ Troubleshooting Quick Links

- **Low agreement?** → See `docs/phase_6_gold_standard.md` § Troubleshooting
- **Slow annotation?** → Reduce task length, add glossary
- **High false positives?** → Adjust weak labeling thresholds
- **PII concerns?** → Use `--deidentify` flag
- **Script errors?** → Check `scripts/annotation/cli.py --help`

---

## 📚 Documentation References

- **Full Guide**: `docs/phase_6_gold_standard.md`
- **Annotation Rules**: `docs/annotation_guide.md`
- **Production Workflow**: `docs/production_workflow.md`
- **Tutorial Notebook**: `scripts/AnnotationWalkthrough.ipynb`
- **CLI Reference**: `scripts/annotation/cli.py --help`

---

## 🎓 Next Phase Preview

**Phase 7: Token Classification Fine-Tuning**
- Convert gold JSONL to BIO-tagged format
- Add classification head to BioBERT
- Fine-tune 3-5 epochs (learning rate 2e-5)
- Target: F1 >0.87 (SYMPTOM), >0.82 (PRODUCT)
- See: `docs/phase_7_training.md` (to be created after Phase 6 complete)

---

**Document Version**: 1.0  
**Created**: November 28, 2025  
**Next Update**: After Batch 001 complete  
**Owner**: SpanForge Project Team
