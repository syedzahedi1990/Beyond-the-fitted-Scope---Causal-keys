# Paper-to-result map

This repository targets the v5.5 main text. It keeps the data and computations supporting the main claims, plus compact evidence for studies explicitly referenced there. It does not require any historical archive.

| Manuscript location | Evidence and CPU command | Model rerun |
|---|---|---|
| Section 5.1, matched response-interface table | `data/behavior/main.json.gz`; `python -m reproduce.behavior --output results-behavior` | `python -m gpu.behavior --model MODEL --study five_shifts --output runs/five-shifts` |
| Sections 5.2–5.3, eight-question consequences, PCA, output recoding and consistency | Same behavioral command; both panels and all fits remain in `behavior.json` | `python -m gpu.behavior --model MODEL --study consequences --output runs/consequences` |
| Section 6.2, native reciprocal exchange, absolute outcomes, localisation controls, Figure 2 | `data/mechanism/native_MODEL.*`; `python -m reproduce.mechanism --study native_MODEL --output results-native` | `python -m gpu.components --help` lists the fixed full/localised studies |
| Section 6.2, six-cycle/pair-swap table | `data/mechanism/mapping_MODEL.*`; `python -m reproduce.mechanism --study mapping_MODEL --output results-mapping` | Fixed mapping study, both mappings and all three fits |
| Section 6.3, fixed-natural-value colour contrast and output-recoding bound | `data/mechanism/fixed_value_MODEL.*`; `python -m reproduce.mechanism --study fixed_value_MODEL --output results-fixed-value` | Fixed-value study, all 72 cores, with 48-primary analysis |
| Rank motivation, broader Qwen fitted-function family, selective-belief/route failures and final-answer mediation reference | `data/behavior/supporting.json.gz` and `mediation.json.gz`; behavioral command | Supporting historical GPU pipelines are not part of this consolidated runner |

Replace `MODEL` with `mistral` or `qwen`. Standalone output directories must not already exist. The top-level `python -m reproduce` runs every CPU result family and constructs the display tables and figure.

## Distinct settings

| Setting | Population | Operation | What it tests |
|---|---|---|---|
| Original consequence audit | 120 discovery and 120 checking stories; seven/eight questions | Fixed event-span residual patch from learned or PCA basis | Consequence scope, preservation, consistency and output-recoding dissociation |
| Fixed-value colour experiment | 72 cores; 48 distinct-location primary | Key exchange with natural target values and recipient-prefix attention-output restoration | Conditional key contribution and a specified fixed colour-recoding explanation |
| Native key-only confirmation | 120 primary cores; 96 distinct and two 12-core collision strata | Key exchange, values/queries evolving naturally, no output restoration | Reciprocal learned–PCA fidelity and distributed contributions |
| Mapping replication | 108 cores, plus a separate 12-core pilot | Full-path key exchange for two separately fitted 300-update mappings | Recurrence across mapping objectives and both models |

The primary CPU replication retains all fixed confirmation conditions, including spatial controls and natural-base donors. Pilot/development outcomes are not pooled with confirmation. Selected masks are fixed inputs here; this repository does not repeat the exposed pilot search to nominate a different mask.

## Reproduction levels

1. **Saved-prediction recomputation:** CPU analysis computes estimands, paired bootstrap intervals and figures from compact per-example outputs. Expected files are used to check the answer, not to calculate it.
2. **Saved-basis model evaluation:** GPU commands load the published bases and exact synthetic inputs, execute new model calls, and retain resulting predictions. Numerical identity is not assumed from the source refactor alone.
3. **Optional refitting:** GPU fitting uses the published training pool, three fixed seeds, rank, site, optimizer and sampling schedules. Freshly trained tensors and their exact initializers have new identities. Historical Qwen initializers remain unavailable, even when a new run saves its own initializers exactly.

Native activation caches and the original full execution journals are not needed for level 1 and are not included. Supporting historical studies have compact data/analysis coverage, not newly ported GPU launchers. Earlier unrun reserved panels are never represented as evaluated outcomes.
