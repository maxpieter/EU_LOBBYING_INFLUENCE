# Gold-standard evaluation of EU lobbying matcher

Methodology: stratified random sample (50% matched / 50% no_match) with a two-pass annotation process. In the first pass, Claude Opus 4.7 proposed a label given the meeting text and the top-20 fuzzy candidates; the researcher accepted, corrected, or rejected each proposal. In the second pass, each row was enriched with the matcher's production signals (MEP-declared `related_procedure`, predicted procedure ID, match method, and matched alias) and Opus 4.7 re-evaluated the label with this additional context. The 38 rows where the second-pass proposal disagreed with the first-pass label were re-reviewed by the researcher, resulting in 35 corrected labels. Final labels from the second pass are used throughout.

Reported metrics use Wilson 95% confidence intervals for proportions (accuracy, precision, recall) and bootstrap (B=2000) percentile 95% confidence intervals for F1, which is non-linear in the underlying counts. Rows the researcher marked `uncertain` or `outside_candidates` are excluded from the denominator. A wrong-procedure-but-matched row counts as both a false positive (matched the wrong file) and a false negative (missed the correct one).

## Meeting → Procedure

- Sample size: 200
- Excluded (uncertain / outside_candidates / unlabeled): 2
- Evaluated: 198

**Confusion matrix:**

|             | gold = match | gold = no_match |
|-------------|--------------|-----------------|
| pred match  | TP=64            | FP=37             |
| pred no_m   | FN=10            | TN=89             |

_Note: 2 rows had matcher predicting a DIFFERENT match than gold. Each is counted as both FP and FN._

**Headline metrics:**
- Accuracy: 77.3% (95% CI [70.9%, 82.6%])
- Precision: 63.4% (95% CI [53.6%, 72.1%])
- Recall: 86.5% (95% CI [76.9%, 92.5%])
- F1: 0.731 (bootstrap 95% CI [0.651, 0.802])

**Per-source breakdown:**

| source | n | accuracy | precision | recall | F1 (95% CI) |
|---|---|---|---|---|---|
| lobbying | 107 | 83.2% (95% CI [75.0%, 89.1%]) | 75.0% (95% CI [62.3%, 84.5%]) | 87.5% (95% CI [75.3%, 94.1%]) | 0.808 [0.708, 0.887] |
| commission | 91 | 70.3% (95% CI [60.3%, 78.7%]) | 48.9% (95% CI [35.0%, 63.0%]) | 84.6% (95% CI [66.5%, 93.9%]) | 0.620 [0.471, 0.744] |

**Per-match-method precision:**

| match method | n | correct | wrong | precision (95% CI) |
|---|---|---|---|---|
| ai_high | 40 | 15 | 25 | 37.5% (95% CI [24.2%, 53.0%]) |
| aia_keyword | 1 | 0 | 1 | 0.0% (95% CI [0.0%, 79.3%]) |
| alias_exact | 19 | 13 | 6 | 68.4% (95% CI [46.0%, 84.6%]) |
| exact_id | 41 | 36 | 5 | 87.8% (95% CI [74.5%, 94.7%]) |
| no_match | 97 | 89 | 8 | 91.8% (95% CI [84.6%, 95.8%]) |

## Methodology paragraph (paste into thesis)

To evaluate the accuracy of the procedure-matching pipeline, a gold-standard evaluation set was constructed through stratified random sampling of 200 meeting–procedure inputs, drawn equally from rows the matcher had classified as matched (n=100) and as no_match (n=100). Annotation followed a two-pass design to mitigate labelling bias. In the first pass, Claude Opus 4.7 (Anthropic, 2025) proposed a ground-truth label for each input given the meeting text and the top-20 fuzzy candidate procedures; the researcher then accepted, corrected, or rejected each proposal. In the second pass, each row was enriched with the matcher's production signals — the MEP-declared related procedure, the predicted procedure identifier, the match method, and the matched alias — after which Opus 4.7 re-evaluated its proposal with this additional context. The 38 rows where the second-pass proposal disagreed with the first-pass label were re-reviewed by the researcher, resulting in 35 corrected labels. Two inputs that the researcher could not resolve confidently were excluded, yielding 198 evaluated rows. The matcher's persisted production decision for each input was then compared to the final gold label, producing a confusion matrix from which precision, recall, and accuracy were computed with Wilson 95% confidence intervals; F1 was reported with a non-parametric bootstrap percentile interval (B=2,000 resamples).
