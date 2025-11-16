# Labeling Tutorial (Label Studio)

Walk-through for non-technical users to curate SYMPTOM and PRODUCT spans.

## 1. Environment Setup
```powershell
setx LABEL_STUDIO_DISABLE_TELEMETRY 1   # one-time (optional)
conda activate NER
label-studio start --no-browser
```
Open http://localhost:8080 in a browser.

## 2. Obtain API Token
- Click user avatar → Account Settings → Enable legacy API tokens (Org tab if required).
- Generate token; copy for import scripts.

## 3. Create Project
Use included config file:
```powershell
python scripts/annotation/init_label_studio_project.py --name "Adverse Event NER"
```
Alternatively create manually in UI, paste XML from `data/annotation/config/label_config.xml`.

## 4. Generate & Import Weak Tasks
Weak label JSONL (from notebook or pipeline) lives at `data/output/<batch>_weak.jsonl`.
```powershell
python scripts/annotation/cli.py import-weak `
  --weak data/output/workflow_demo_weak.jsonl `
  --out data/annotation/exports/tasks.json `
  --push --project-id <PROJECT_ID>
```
Tasks appear in the project task list.

## 5. Annotating
For each task:
1. Read entire complaint.
2. Highlight SYMPTOM or PRODUCT phrase.
3. Select label from interface.
4. Adjust existing weak suggestions (delete incorrect, expand truncated, add missing).
5. Save/Submit.

### Boundary Checklist
- Include severity modifiers ("severe rash").
- Exclude trailing punctuation.
- Preserve internal spacing/casing.
- Keep negated symptoms (e.g., "no irritation")—negation handled later.

### Keyboard Shortcuts (Default)
| Action | Shortcut |
|--------|----------|
| Select label | Click or numeric index |
| Undo last span | Ctrl+Z |
| Redo | Ctrl+Y |

## 6. Export Annotations
In project: Export → JSON. Save to `data/annotation/raw/<export_name>.json`.

## 7. Convert to Gold
```powershell
python scripts/annotation/convert_labelstudio.py `
  --input data/annotation/raw/<export_name>.json `
  --output data/annotation/exports/<batch>_gold.jsonl `
  --source <batch_id> `
  --annotator <your_name> `
  --symptom-lexicon data/lexicon/symptoms.csv `
  --product-lexicon data/lexicon/products.csv
```

## 8. Quality Report
```powershell
python scripts/annotation/cli.py quality `
  --gold data/annotation/exports/<batch>_gold.jsonl `
  --out data/annotation/reports/<batch>_quality.json
```
Review conflicts and distributions.

## 9. Register Batch
```powershell
python scripts/annotation/cli.py register `
  --batch-id <batch_id> `
  --gold data/annotation/exports/<batch>_gold.jsonl `
  --annotators <comma_names> `
  --revision 1
```

## 10. Adjudication (If Needed)
Conflicts listed in quality report can be processed:
```powershell
python scripts/annotation/adjudicate.py --gold data/annotation/exports/<batch>_gold.jsonl --out data/annotation/exports/<batch>_gold_resolved.jsonl
```

## Troubleshooting
| Issue | Resolution |
|-------|------------|
| 400 import error | Ensure tasks POST body is raw array, not wrapped object. |
| Empty task list | Verify project ID & API token validity. |
| Token rejected | Legacy tokens may need enabling in Org settings. |
| Wrong character indices | Confirm spans copy exact text; use notebook integrity cell. |

## Best Practices
- Annotate daily small batches (≤50) for consistency.
- Log notes in registry for threshold changes.
- Periodically re-run quality reports to monitor drift.

## Next Steps
After several batches: perform consensus, finalize label inventory, generate BIO tags, and begin supervised fine-tuning.
