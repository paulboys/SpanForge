"""
Quick Start Guide for FDA CAERS Integration
============================================

This guide shows how to use the CAERS integration script to download and process
FDA consumer complaint data for SpanForge NER testing.

IMPORTANT: Pandas is required. Install with:
    pip install pandas

Examples
--------

1. Test with 100 complaints (cosmetics only):

    python scripts/caers/download_caers.py \
        --output data/caers/test_100.jsonl \
        --categories cosmetics \
        --limit 100

2. Get 1000 complaints from multiple categories:

    python scripts/caers/download_caers.py \
        --output data/caers/mixed_1000.jsonl \
        --categories cosmetics personal_care baby \
        --limit 1000 \
        --min-spans 2

3. Process all cosmetics (no limit):

    python scripts/caers/download_caers.py \
        --output data/caers/all_cosmetics.jsonl \
        --categories cosmetics

4. Quick validation check:

    python scripts/caers/download_caers.py \
        --output data/caers/validation.jsonl \
        --limit 50 \
        --validate

Expected Output
---------------

Console logs will show:
- Download progress (125+ MB file)
- Number of records loaded (666,000+)
- Filtering results
- Processing progress (every 100 complaints)
- Final statistics (successful, skipped, spans detected)

Output files:
- data/caers/[output_name].jsonl - Weak labeled complaints
- data/caers/[output_name]_stats.json - Processing statistics

Troubleshooting
---------------

If script fails with import errors, ensure you're in the NER conda environment:
    conda activate NER

If pandas is missing:
    pip install pandas

If download is slow or fails:
- Check internet connection
- The CAERS CSV is ~125 MB and may take 5-10 minutes to download
- File is cached in data/caers/raw/ for future runs

Next Steps
----------

After generating JSONL data:
1. Inspect output: cat data/caers/test_100.jsonl | head -n 1
2. Check statistics: cat data/caers/test_100_stats.json
3. Use for weak label testing
4. Import to Label Studio for annotation
5. Run evaluation pipeline

For more details, see scripts/caers/README.md
"""

if __name__ == "__main__":
    print(__doc__)
