# Label Studio Configuration

This directory contains Label Studio project configuration files.

## Files

### label_config.xml

**Purpose**: Defines the annotation interface for adverse event NER.

**Entity Types**:
- **SYMPTOM** (green, hotkey: `s`) - Medical symptoms, adverse reactions, clinical observations
- **PRODUCT** (blue, hotkey: `p`) - Product names, medications, devices

**Features**:
- Word-level granularity for precise boundary selection
- Color-coded labels for visual distinction
- Keyboard shortcuts for faster annotation (s/p keys)

## Usage

### 1. Import to Label Studio

**Option A - Via Web UI**:
1. Create new project in Label Studio
2. Go to Settings → Labeling Interface
3. Click "Code" tab
4. Copy contents of `label_config.xml`
5. Click "Save"

**Option B - Via API** (using `init_label_studio_project.py`):
```bash
python scripts/annotation/init_label_studio_project.py \
  --config data/annotation/config/label_config.xml \
  --name "Adverse Event NER" \
  --description "Symptom and product annotation for biomedical complaints"
```

### 2. Test Configuration

**Import sample tasks**:
```bash
python scripts/annotation/cli.py import-weak \
  --weak tests/fixtures/annotation/weak_baseline.jsonl \
  --out data/annotation/imports/test_tasks.json \
  --include-preannotated
```

**Verify**:
1. Select text in Label Studio interface
2. Confirm SYMPTOM (green) and PRODUCT (blue) labels appear
3. Test keyboard shortcuts: `s` for SYMPTOM, `p` for PRODUCT
4. Export sample annotation and validate JSON structure

### 3. Customize (Optional)

**Add negation flag** (requires Label Studio ≥1.7.0):
```xml
<Labels name="label" toName="text">
  <Label value="SYMPTOM" background="#2ca02c" hotkey="s"/>
  <Label value="PRODUCT" background="#1f77b4" hotkey="p"/>
</Labels>

<Choices name="attributes" toName="text" perRegion="true">
  <Choice value="negated" hint="Symptom is negated (e.g., 'no redness')"/>
</Choices>
```

**Add severity scale** (for advanced annotation):
```xml
<Rating name="severity" toName="text" maxRating="5" 
        icon="star" perRegion="true" required="false"/>
```

## Annotation Guidelines

See `docs/annotation_guide.md` for detailed rules:
- Boundary definitions (include full medical terms, exclude adjectives)
- Negation policy (annotate negated symptoms with flag)
- Anatomy gating (skip single tokens like "skin" unless part of symptom phrase)
- Overlapping spans (choose most semantically complete)

## Troubleshooting

**Issue**: Labels don't appear when selecting text  
**Fix**: Verify `granularity="word"` is set in `<Text>` tag

**Issue**: Hotkeys don't work  
**Fix**: Click inside annotation area first, then use shortcuts

**Issue**: Export JSON structure mismatch  
**Fix**: Run `convert_labelstudio.py` to normalize to SpanForge format

## Integration Points

- **Import**: `import_weak_to_labelstudio.py` converts weak JSONL → Label Studio tasks
- **Export**: Label Studio → JSON export (via API or manual download)
- **Conversion**: `convert_labelstudio.py` converts Label Studio JSON → gold JSONL
- **Validation**: `quality_report.py` checks annotation quality metrics

## References

- Label Studio Documentation: https://labelstud.io/guide/setup.html
- Annotation Guide: `docs/annotation_guide.md`
- Tutorial Notebook: `scripts/AnnotationWalkthrough.ipynb` (when complete)
