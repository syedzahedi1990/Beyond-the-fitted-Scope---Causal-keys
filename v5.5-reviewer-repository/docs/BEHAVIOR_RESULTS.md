# Behavioral results from saved case-level evidence

Run from the repository root:

```sh
python -m reproduce.behavior --output outputs/behavior
python -m unittest tests.test_behavior -v
```

The output directory must be new. The command verifies every compressed and decompressed evidence-file hash, validates complete panels and semantic labels, recomputes the results, and compares the installed reference. It writes `behavior.json`, `behavior_tables.md` and `RECEIPT.json`. NumPy and the Python standard library are the only dependencies. No model is loaded.

`--data-dir DIR` accepts the same evidence schema and an accompanying `SOURCE_PROVENANCE.json`. Integrity and panel validation still run. A replacement evidence directory does not automatically claim agreement with the installed historical results: supply `--expected FILE` to compare an independently supplied reference. The receipt states whether that comparison ran.

## What is recomputed

| Evidence | Cases retained | Results |
|---|---:|---|
| Original matched Qwen72/Mistral24 | 1,500 pairs per model, two interfaces, all three intended/alternative/PCA fits | Main Table 2 candidate and global correctness, candidate mass, objective gap; main Table 3 five-shift learned–PCA contrasts; clean/shared-opportunity populations |
| Fixed consequences | 120 discovery and 120 checking stories per model; seven/eight views; all nine bases and three natural controls | Main Table 3 consequence contrasts; every per-fit/per-view and joint count; output-recoded PCA; direct-anchored consistency, errors and account-discrimination opportunity |
| Broader Qwen72 function family | Four objectives × three 300-update fits, each 500 IID and 300 shifted pairs | Every fit's own-target and source-target rates; nontrivial and clean-endpoint subsets |
| Historical 7B rank motivation | Both 200-pair rank-16/64 DAS pilots; Qwen prefix-PCA and both event-PCA panels | Every saved rank/group cell, source transfer, and clean-endpoint-conditioned counts |
| Failed selective-belief applicability | Qwen three-agent and Mistral two-agent screens, 60 stories each × eight views × four controls | All global/candidate correctness and mass cells, whole-story controls, original natural-prerequisite decisions |
| Failed route feasibility | All 1,080 original natural-control rows and all 252 wording-development rows | All cells, original 54/60 and 11/12 prerequisites, all three variants; no learned stage invented |
| Failed final-answer mediation | All 20 candidates × 60 stories × eight views, seed 101, with three matched controls | Target–clean log-score total/restoration/transplant effects, control comparisons, unaffected preservation costs and original practical nomination decisions |

The main Table 2/3 numbers are independently asserted against the published values in `tests/test_behavior.py`, rather than testing only equality to the generated reference. Those tests also check all twelve learned checking rows, direct-success/joint-failure recoding, the 9.762-point historical Qwen discovery near-miss, consistency counts, negative controls and mediation failures.

## Estimands and interpretation

The main tables distinguish candidate ranking from the actual saved global next-token label. A `null` global label means an original noncandidate output; it is never replaced with the candidate argmax. Candidate masses remain absolute full-vocabulary masses. Historical breadth/rank files retain their original candidate-label estimand and are not relabelled as global-token evidence.

The main paired intervals use the original 2,000-draw NumPy procedure, seed 2026091301. Primary pairs are resampled within five equally weighted 300-pair families. Consequences resample whole stories. Each draw keeps the three fixed-fit columns paired; there is no fit selection or claim of independent model/optimizer populations. These are pointwise intervals, not multiplicity-adjusted intervals. The empty original-boundary global clean-opportunity population is reported as non-estimable. It is not assigned zero accuracy.

Output recoding applies the fixed pair-swap to each saved PCA answer, including preserved and overwritten questions. It produces 120/120 correct direct targets but 0/120 fully correct eight-answer stories in every fit of both models. This is an external answer-processing comparator, not a newly executed neural intervention. Direct-anchored consistency instead uses the model's actual direct answer to forecast the other seven answers and can be coherent even when the target is wrong.

Symbolic account comparisons retain every candidate account, disagreement denominator and fit. Panel-equivalent accounts are labelled explicitly; common opportunity is the intersection of stories capable of distinguishing event rewrite from every nonequivalent account. The retrospective diagnostics do not revise the historical discovery decision. Qwen alternative seed 101 has 82 net event-versus-belief hits over 840 discovery answers: 9.7619%, below its original ten-point criterion.

Qwen's original story PCA comparator is a matched-procedure reconstruction, not an authenticated copy of its missing original initializer. Mistral's matched story comparator is the exact saved initializer. These identities are retained in the computed JSON. The broader 300-update fits and preliminary 7B panels remain separate from these original matched cohorts.

Both selective-belief controls fail. The Qwen nonobserver cell is 45/60; the Mistral cell is 1/60. Neither failure becomes a claim of learned selective-belief success. Both route designs fail their original prerequisites. All 20 final-answer-site mediation candidates fail the half-effect restoration and transplant criteria; 16 and 14 also fail their respective matched-control comparisons. No held-out mediation checking stage was run.

## Evidence and verification boundaries

Three gzip-compressed JSON files contain selected original scalar evidence, not saved aggregate results. `SOURCE_PROVENANCE.json` lists original source-file hashes, dataset hashes, archive/member chains where necessary, and hashes of each compact file. The archive is not included or required for this CPU command. Original source data were read only; no failed condition was deleted.

`main.json.gz` preserves the normalized per-case candidate/global decisions and candidate masses, ordered pair metadata, all consequence cores, all methods and both panels. It does not include full vocabulary logits or global token IDs for noncandidate answers, which the original normalized records did not retain. `supporting.json.gz` preserves per-case predictions and labels for the stated panels. `mediation.json.gz` preserves each original target and clean log probability and global decision for seven conditions and three matched controls, together with the hash of the complete original row.

This replay authenticates the included evidence and recomputes behavioral statistics. It does not independently replay GPU forwards, optimizer updates, PCA/QR reconstruction, tensor captures, native journals, or omitted identity-sham payloads. The mediation result is a final-answer-site screen, not a proof that no other mediator exists. No arbitrary domain-general behavior, extra sites, extra seeds, or arithmetic experiments are added here.

The installed full reference comparison requires exact fields, integer counts and labels. Floating values allow absolute difference at most 1e-12 for portable reduction-order roundoff; CI endpoint signs must also agree exactly. Every unrounded recomputed number is saved. The tolerance is a CPU reproducibility check, not a scientific success gate.
