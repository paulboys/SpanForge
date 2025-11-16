# Weak Labeling Heuristics

Defines rule-based span proposal logic prior to supervised training.

## Inputs
- Complaint texts
- Symptom lexicon: `data/lexicon/symptoms.csv`
- Product lexicon: `data/lexicon/products.csv`

## Processing Steps
1. **Candidate Windows**: Iterate token windows up to max phrase length of lexicon entries.
2. **Fuzzy Scoring**: Compute WRatio (RapidFuzz) between window text and lexicon term.
3. **Token-Set Jaccard**: Lowercased token set overlap percentage.
4. **Gates**: Accept if `fuzzy ≥ 0.88` AND `jaccard ≥ 40`.
5. **Alignment**: Multi-token candidate must align on last token boundary with lexicon term (mitigates mid-span inflation).
6. **Anatomy Filtering**: Skip single generic anatomy tokens unless symptom keyword co-occurs.
7. **Negation Marking**: Build window of 5 tokens around negation cues; ≥50% token overlap → mark `negated=True`.
8. **Confidence**: Weighted combination (see below); duplicates resolved by highest confidence.

## Thresholds
| Parameter | Default | Purpose |
|-----------|---------|---------|
| Fuzzy WRatio | 0.88 | Balances lexical variant recall vs noise |
| Jaccard % | 40 | Ensures partial but meaningful token overlap |
| Negation window | 5 tokens | Captures local negation context |
| Overlap for negation | ≥50% | Avoids spurious negation marking |
| Multi-token alignment | Enforced | Reduces partial window drift |

## Confidence Formula
```
confidence = clamp(0.8 * (fuzzy/100) + 0.2 * (jaccard/100), 0.0, 1.0)
```
Fuzzy & Jaccard are raw percentages (0–100) before weighting.

## Negation Cues (Examples)
`no`, `not`, `without`, `never`, `none`, `free of`, `lack of`.
Token normalization handles casing; multi-word cues expanded via phrase tokenization.

## Canonical Mapping
If lexicon entry matched, `canonical` set to curated term; otherwise fallback canonical = surface span (enables later normalization decisions & lexicon expansion).

## Duplicate / Overlap Policy
- Exact duplicates (same start,end,label) → keep highest confidence only.
- Overlapping distinct spans retained; conflicts surfaced later for human review.

## Exclusions
- Pure stopword spans.
- Isolated anatomy token without symptom context.
- Zero-length or punctuation-only windows.

## Tuning Guidance
| Symptom | Adjust | Effect |
|---------|--------|--------|
| High false positives | Increase fuzzy (0.90) or Jaccard (50) | Precision ↑, Recall ↓ |
| Low recall variants | Lower Jaccard (35) first | Recall ↑ moderate |
| Many partial matches | Enforce stricter alignment | Noise ↓ |

Tune one parameter at a time; evaluate on gold comparison script.

## Planned Enhancements
- Embedding similarity (BioBERT cosine) secondary gate.
- Contextual disambiguation around product proximity.
- Label-specific thresholds (PRODUCT vs SYMPTOM).
- Active learning: mine high-uncertainty spans for prioritization.

## Drift Monitoring
Track PRODUCT:SYMPTOM ratio per batch; sudden deviation triggers threshold audit.

## Safety Considerations
Avoid aggressive threshold lowering early—annotation burden & noise escalate quickly.

## Reference Implementation
See `src/weak_label.py` for authoritative logic and configuration hooks.
