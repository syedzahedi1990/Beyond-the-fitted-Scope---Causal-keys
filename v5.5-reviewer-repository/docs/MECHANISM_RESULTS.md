# Mechanism evidence and CPU reproduction

Run from the repository root:

```sh
python -m reproduce.mechanism --output results/mechanism
```

The output directory must be new. This command reads the included per-return evidence, independently reconstructs symbolic answers and endpoint comparisons, computes paired confidence intervals, and checks published validation targets. It does not call a model, import the original analyzers, or use reported aggregate estimates as computational inputs.

## Included scientific panels

| Panel | Models | Cores per model | Fixed fits | Saved returns per model | Included conditions |
|---|---|---:|---:|---:|---|
| Native keys, no output restoration | Mistral24B, Qwen72B | 120 | 101, 102, 103 | 63,000 | Full path, nominated band, exact complement, deterministic equal-size band; both recipient frames; self, other, signed-null, natural-B |
| Learned keys with fixed natural target values | Mistral24B, Qwen72B | 72, including the 48-core distinct primary population | 101, 102, 103 | 16,200 | Keys from P, M, natural S, natural T, signed-null, and natural B/B; both frames |
| Six-cycle and pair-swap mapping replication | Mistral24B, Qwen72B | 108 | 101, 102, 103 | 35,640 | Both mappings, both frames; full-path self, other, signed-null, natural-B |

There are **229,680 saved scientific returns**. Each dataset includes all its natural and fitted endpoints, conditions, consumers, errors, and invalid vocabulary IDs. Technical qualifications, fitting trajectories, development-screen returns, and raw activations are not pooled into these confirmation estimates. The nominated native masks are retained in each dataset's metadata; their selection occurred during development, not during this reproduction.

The five consumers are direct location, two observer color-table queries, and two nonobserver color-table queries. The first three are affected consumers; the last two test preservation. Native paths cover blocks 6–40 for Mistral and 6–80 for Qwen, all eight KV groups, with keys exchanged and no value exchange or output-prefix restoration. The fixed-value panel instead uses the historical broad K/V operation with natural target V in **both** recipient frames and its declared prefix restoration. Its same-key hybrid is not a bare native endpoint.

The native 120-core panels have 96 distinct, 12 base=source, and 12 base=target cores. Each mapping's 108-core panel has 72/18/18 strata; mapping-specific memberships can differ. The common 54-core mapping-disagreement population is also retained. The 72-core panel has 48/12/12 strata. These populations must not be substituted for one another.

Mistral uses exact saved effective PCA initializers. Qwen uses its documented matched-procedure reconstructed PCA references, without an exact historical-initializer claim. The mapping replication compares separate 300-update fit cohorts: newly fitted Mistral and archived Qwen. The 108 semantic cores were Qwen-exposed before their Mistral evaluation. This is not a new independent population for each model, nor a controlled comparison of model scale or PCA provenance.

## Evidence, authentication, and limits

`data/mechanism/MANIFEST.json` binds six metadata files. Each metadata file binds one deterministic gzip JSONL, the original source-row SHA-256 and byte size, closure/configuration/input identities where available, token alphabets, ordered cores, masks, and fit labels. The 12 compact fields are:

```
call_id, core_index, mapping, fit_seed, endpoint, mask, frame,
condition, view_index, global_token_id, six_candidate_log_scores, candidate_mass
```

View and candidate-label orders are explicit in the metadata and module. A null endpoint denotes an exchange. Natural endpoints have null fit seeds and occur once; they are not relabeled as three independent observations. Location/color labels and validity are reconstructed from the actual full-vocabulary winning token and the disjoint saved candidate alphabets. Every candidate score is preserved; missing out-of-alphabet endpoint scores stay unavailable. The Qwen fixed-value portable evidence did not retain candidate mass, so those fields remain unavailable instead of being invented.

The compact projection was extracted from complete authenticated original row ledgers. Source row bytes were streamed and rehashed during extraction; Qwen fixed-value rows came from its authenticated per-example portable package. The extraction utility's SHA is recorded. Large native journals, top-token displays, model arrays, and operational receipts are omitted from this CPU evidence format; they are unnecessary for the listed statistics. This repository does **not** reauthenticate native tensor arithmetic or regenerate raw model outputs. Original closures and independent raw audits remain the provenance boundary, not an audit secretly replaced by these compact files.

Qwen fixed-value evidence has a disclosed interruption: all native calls survived, an independently checked CPU recovery produced a derived closure, and no additional model calls occurred. The original producer COMPLETE marker was absent; final live guards were not rerun. This exception is carried in that dataset's metadata.

## Independent statistics

The module constructs the complete core × fit × condition × view grid and rejects duplicate or missing cells. For native studies, self-token, candidate-score and candidate-mass identities are checked against the corresponding endpoints.

Fidelity compares **actual opposite P/M global token IDs**, including wrong or invalid IDs. Correctness uses the executable location permutation and the two color tables. Natural-B recovery uses the B oracle separately. Preservation reports both actual-recipient agreement and base-oracle correctness. Full patterns combine affected fidelity or correctness with **base-oracle** nonobserver preservation; these are not interchangeable. Error partitions distinguish retained correct answers, newly damaged correct answers, copied donor errors, repaired donor errors, and other errors. Pair partitions are computed jointly.

Scores include candidate log probabilities, aligned symbolic margins, and actual destination-minus-recipient margins. If distinct endpoint IDs lack saved scores, the population/fit margin is unavailable; no case is discarded. Coincident endpoint IDs have exactly zero margin, including coincident invalid IDs.

Each analysis uses 20,000 shared whole-core draws and linear pointwise 95% percentile intervals. Fixed fits are averaged **within each core before resampling**; all three fit estimates are also shown. Native panels use seed 2026092201 and unstratified all120 draws. Mapping panels use seed 2026092301 and unstratified all108 draws; paired m1−m3 effects use the identical all108 or disagreement core memberships. Fixed-value panels use seed 2026091635 and separate 48/12/12 draws; full72 estimates combine these with their original population weights. The same draws are reused across conditions, and their hashes are output. Intervals condition on this fixed fit ensemble, not a population of training seeds. There is no efficacy gate or outcome filtering.

Both forward and actual inverse permutation output recoders are computed from saved endpoint predictions and score distributions. In particular, removal for the non-involutive six-cycle does not apply the forward cycle a second time. The old 50% color ceiling does not apply to the commuting six-cycle. For the fixed pair-swap 48-core panel, the program separately enumerates all 6^6 table-independent six-color functions. It reports the source-oracle and actual-P maximum correct observer-pair counts, source balance, and whether P satisfies the source premise. Both published panels yield 24/48 for all three fits. This function-family ceiling does not bound arbitrary table-aware programs.

## Outputs and checks

- `mechanism_summary.json`: panel counts, main pooled effects, intervals, all fit points, validation status.
- `mechanism_primary.csv`: four main effects per model/mapping, in percentage points.
- `mechanism_conditions.csv`: condition, endpoint, output-recoder, and paired-control statistics for every population and fit. Counts, denominators, fidelity, correctness, preservation, error copying and scores remain distinct.
- Six study JSON files: complete computed metric tables, draw identities, provenance, and paired mapping differences. Confidence intervals are emitted for the paired contrasts on the selected fidelity, correctness, preservation, error-transfer and margin metrics. Absolute condition and endpoint tables are descriptive counts/means; their null interval fields mean no interval was calculated. This does not reproduce every secondary interval in the much larger historical reports.

`expected/mechanism.json` contains separately extracted original-report validation targets and report hashes. It is used only **after** calculation: 11,712 comparisons cover all main pooled/per-fit points and intervals plus selected condition/count fields across every retained population. A 2×10^-12 absolute numerical tolerance accommodates summation/BLAS roundoff; this is not a bitwise historical-report reproduction claim. Tests also cover error copying with invalid IDs, missing-margin retention, joint preservation, true inverse recoding, fixed-fit averaging, complete grids, and the exhaustive color bound.

The main native other-minus-null gains are approximately 55.28/75.37 percentage points for Mistral/Qwen addition and 76.57/83.70 for removal. Mapping results remain distinctly less correct than faithful in addition; the computation retains all imperfect preservation and B-control patterns. Positive actual-endpoint transfer is not a claim of minimality, native necessity, universal correctness, or uniform hybrid superiority.

## Fresh GPU results

A completed new GPU run can emit the same compact metadata/JSONL schema. Analyze it with:

```sh
python -m reproduce.mechanism --data-dir NEW_RUN/data --new-evidence --output NEW_RUN/statistics
```

This explicit mode retains all integrity/grid checks and statistical definitions but does not require new predictions to equal published outcomes. It accepts a separately labeled full-only native mask configuration; such a run does not reproduce the published localization comparisons. The default command always requires agreement with the published evidence targets.
