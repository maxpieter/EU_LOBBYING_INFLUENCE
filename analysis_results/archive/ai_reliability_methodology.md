# AI Classifier Reliability: Methodology

## Sample Design

Reliability was evaluated separately for two AI-assisted classification components embedded in the EU lobbying data pipeline. For each component, a stratified random sample of N=400 AI-positive decisions was drawn from the production database.

For the **procedure-matching component**, rows were sampled from `meeting_procedure_links` filtered to `match_method = 'ai_high'` (approximately 1,176 rows at time of evaluation). Stratification used the stored `match_confidence` float, divided into three bins: high (≥0.9), mid (0.8–0.9), and low (<0.8), with proportional allocation. For the **organisation-resolution component**, rows were sampled from `organizations` filtered to `dedup_status = 'high'` — the value written by `resolve_stubs()` in `pipeline/assets/organizations/fuzzy.py` when Claude confirmed a match with high confidence. Stratification used `organization_type` as a proxy for entity category. Both samples used a fixed random seed (42) and are committed to the repository as `analysis/labels_procedure.csv` and `analysis/labels_org.csv` for full reproducibility.

## Labeling Protocol

All rows were labeled by a single rater (the thesis author) using an interactive CLI (`scripts/label_ai_matches.py`). Each decision was classified as correct, wrong, or uncertain; uncertain labels are excluded from the precision denominator but reported separately. For each wrong label, a short error-category string was recorded to support qualitative error typology analysis. To enable a self-agreement check, 10% of each sample (N=40) was designated at sampling time as `relabel_round = 1`. These rows are re-labeled blind by the same rater at least one week after the initial round, and Cohen's kappa is computed from the two rounds using `scripts/compute_ai_reliability.py`.

## Primary Metric and Justification for Excluding Recall

The primary reliability metric is **precision**: of all meetings/organisations the AI classifier accepted, what fraction are genuinely correct matches. Precision was estimated as a point estimate with a 95% Wilson score confidence interval (not the normal approximation), which produces asymmetric bounds appropriate for samples with high precision rates. Recall — the fraction of true matches that the AI accepted — is explicitly outside scope for this evaluation: estimating recall would require exhaustive labeling of AI-negative decisions (meetings the AI rejected), which would roughly double the labeling burden and is practically infeasible within the thesis timeline. Precision is the operationally relevant quantity because it determines the data cleanliness for downstream network and influence analysis; inflated recall would silently introduce false positives, whereas missed matches (recall failure) reduce dataset coverage but do not corrupt the matches that are present.

## Reproducibility

Label files, sampling script, and metrics script are version-controlled in the thesis repository. The Wilson CI formula is hand-rolled in `scripts/compute_ai_reliability.py` (no external statistics library dependency), and Cohen's kappa is computed from first principles. Any researcher with access to the Supabase instance can reproduce the exact sample by running `label_ai_matches.py --seed 42` and rerunning the metrics script on the committed CSV.
