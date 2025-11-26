# FDA CAERS Data Integration

This directory contains scripts for downloading and processing FDA CAERS (Consumer Adverse Event Reporting System) data for SpanForge NER testing.

## Overview

The FDA CAERS database contains adverse event reports for:
- Cosmetics and personal care products
- Dietary supplements
- Foods and beverages
- Baby products

This makes it an excellent dataset for testing SpanForge's biomedical NER capabilities with consumer complaint text.

## Quick Start

### 1. Download and Process First 1,000 Cosmetics Complaints

```powershell
python scripts\caers\download_caers.py `
  --output data\caers\cosmetics_test.jsonl `
  --categories cosmetics `
  --limit 1000
```

### 2. Process All Personal Care and Baby Products

```powershell
python scripts\caers\download_caers.py `
  --output data\caers\full_dataset.jsonl `
  --categories personal_care baby `
  --min-spans 2
```

### 3. Quick Validation Check (100 Samples)

```powershell
python scripts\caers\download_caers.py `
  --output data\caers\validation_sample.jsonl `
  --limit 100 `
  --validate
```

## Script Features

### `download_caers.py`

**Capabilities:**
- ✅ Automatic download from FDA (updated quarterly)
- ✅ Category filtering (cosmetics, supplements, foods, personal_care, baby)
- ✅ Text extraction from multiple CAERS columns
- ✅ Weak labeling with SpanForge lexicons
- ✅ JSONL export for annotation workflow
- ✅ Validation pipeline (spans, boundaries, text alignment)
- ✅ Statistics reporting

**CLI Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | Path | `data/caers/caers_labeled.jsonl` | Output JSONL file path |
| `--download-dir` | Path | `data/caers/raw` | Directory for downloaded CSV |
| `--categories` | List | None | Filter by categories (cosmetics, supplements, etc.) |
| `--limit` | Int | None | Maximum complaints to process |
| `--min-spans` | Int | 1 | Minimum spans required per complaint |
| `--force-download` | Flag | False | Force re-download even if CSV exists |
| `--validate` | Flag | True | Enable validation checks |
| `--no-validate` | Flag | - | Disable validation checks |
| `--symptom-lexicon` | Path | `data/lexicon/symptoms.csv` | Path to symptom lexicon |
| `--product-lexicon` | Path | `data/lexicon/products.csv` | Path to product lexicon |

## Output Format

### JSONL Record Structure

```json
{
  "text": "After using this facial moisturizer, I developed severe burning sensation and redness on my cheeks.",
  "source": "FDA_CAERS",
  "metadata": {
    "report_id": "12345678",
    "date_created": "2024-03-15",
    "product_type": "Cosmetics",
    "product_name": "Premium Face Cream",
    "age": "34",
    "gender": "Female",
    "outcomes": "Recovered"
  },
  "date_processed": "2025-11-25T10:17:15.123456",
  "spans": [
    {
      "text": "burning sensation",
      "start": 57,
      "end": 74,
      "label": "SYMPTOM",
      "canonical": "Burning Sensation",
      "source": "sample",
      "concept_id": null,
      "sku": null,
      "category": null,
      "confidence": 1.0,
      "negated": false
    },
    {
      "text": "redness",
      "start": 79,
      "end": 86,
      "label": "SYMPTOM",
      "canonical": "Redness",
      "source": "sample",
      "concept_id": null,
      "sku": null,
      "category": null,
      "confidence": 0.95,
      "negated": false
    }
  ]
}
```

### Statistics Report

The script generates a statistics JSON file alongside the output:

```json
{
  "total_rows": 1000,
  "processed": 1000,
  "skipped_no_text": 89,
  "skipped_no_spans": 52,
  "validation_failed": 12,
  "successful": 847,
  "total_spans": 1923,
  "symptom_spans": 1645,
  "product_spans": 278
}
```

## Integration with SpanForge Workflow

### Step 1: Download and Label

```powershell
# Get 1000 cosmetics complaints with weak labels
python scripts\caers\download_caers.py `
  --output data\caers\batch1.jsonl `
  --categories cosmetics `
  --limit 1000
```

### Step 2: LLM Refinement (Optional)

```powershell
# Refine weak labels with LLM
python -m src.llm_agent `
  --weak data\caers\batch1.jsonl `
  --output data\caers\batch1_refined.jsonl
```

### Step 3: Import to Label Studio

```powershell
# Import for human annotation (when script ready)
python scripts\annotation\import_weak_to_labelstudio.py `
  data\caers\batch1_refined.jsonl
```

### Step 4: Evaluation

```powershell
# After annotation, evaluate performance
python scripts\annotation\evaluate_llm_refinement.py `
  --weak data\caers\batch1.jsonl `
  --refined data\caers\batch1_refined.jsonl `
  --gold data\caers\batch1_gold.jsonl `
  --output data\annotation\reports\caers_eval.json `
  --markdown
```

## Data Source

**FDA CAERS Database:**
- URL: https://www.fda.gov/food/compliance-enforcement-food/human-foods-complaint-system-hfcs
- Direct Download: https://www.fda.gov/media/180475/download
- Update Frequency: Quarterly
- Total Records: 666,000+ (as of Q4 2024)
- License: Public Domain (CC0)

**Product Categories in CAERS:**
- Cosmetics: ~45,000 reports
- Dietary Supplements: ~250,000 reports
- Foods: ~300,000 reports
- Personal Care: Subset of cosmetics
- Baby Products: Subset tagged with infant/pediatric

## Expected Performance

**Processing Speed:**
- Small batch (100 records): ~5 seconds
- Medium batch (1,000 records): ~30 seconds
- Large batch (10,000 records): ~5 minutes

**Span Statistics (Typical):**
- Average spans per complaint: 2-3
- Symptom spans: ~85% of total
- Product spans: ~15% of total
- Negated spans: ~10-15% of symptoms

**Success Rate:**
- Records with text: ~90%
- Records with spans: ~85%
- Validation pass rate: ~95%

## Troubleshooting

### Download Failures

If download fails, try:
1. Check internet connection
2. Verify FDA URL is still active
3. Use `--force-download` to retry
4. Download manually and place in `data/caers/raw/`

### Low Span Detection

If too few spans detected:
1. Expand symptom lexicon (`data/lexicon/symptoms.csv`)
2. Add product brand names to product lexicon
3. Reduce `--min-spans` threshold
4. Check validation logs for skipped records

### Memory Issues

For large batches (>50,000 records):
1. Use `--limit` to process in chunks
2. Process by category separately
3. Increase system RAM or use 64-bit Python

## Future Enhancements

- [ ] Add MedDRA concept mapping for symptoms
- [ ] Extract product brand names from CAERS data
- [ ] Implement deduplication across reports
- [ ] Add temporal analysis (trends over time)
- [ ] Support for other FDA databases (FAERS, MAUDE)

## License

Script: MIT License (SpanForge project)  
CAERS Data: Public Domain (CC0) - U.S. Government work
