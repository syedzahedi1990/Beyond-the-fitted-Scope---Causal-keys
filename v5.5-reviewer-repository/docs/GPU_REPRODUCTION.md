# Optional GPU reruns

The repository provides a derived, shared implementation for new model runs. The published saved-prediction replay requires no GPU. Numerical helpers, native attention hooks, prompts, datasets, fitted arrays and fixed conditions are retained from the experiments, while the execution and packaging code is consolidated. `RELEASE.json` authenticates these derived files; historical source seals do not authenticate transformed copies.

The new entrypoints have been checked with CPU fixtures and source review. **This consolidated implementation has not been run end to end on a GPU.** It does not promise byte-identical predictions on a new machine. There is no behavioral success threshold, case filtering, fit selection or automatic follow-up. A technical scope, identity, nonfinite-value or inventory failure terminates the run and retains its evidence.

## Environment and models

Use a separate environment from the NumPy 2.3.5 CPU replay environment. The native model wrapper requires Python **3.12.14**, NumPy **1.26.4**, Torch **2.11.0+cu128** and the exact packages in `gpu/requirements.txt`.

```bash
python3.12 -m venv .venv-gpu
.venv-gpu/bin/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv-gpu/bin/python -m pip install -r gpu/requirements.txt
.venv-gpu/bin/python -m reproduce.integrity
```

`python3.12` must resolve to patch version 3.12.14. Model loading checks package versions and model identity; component adapters additionally check the pinned Transformers model/SDPA source hashes. One visible, unquantized BF16 CUDA device is required. Mistral requires at least 75 GiB physical GPU memory; the inherited Qwen profile requires one B200 with at least 178 GiB. Neither model parallelism nor quantization is supported by this implementation. Set `CUDA_VISIBLE_DEVICES` before invocation if the machine exposes multiple GPUs. Provide ample CPU memory and disk for the weights and returned evidence; full model weights and historical raw activation outputs are not included in this repository.

| Profile | Model | Pinned revision |
|---|---|---|
| Mistral | `mistralai/Mistral-Small-24B-Instruct-2501` | `9527884be6e5616bdd54de542f9ae13384489724` |
| Qwen | `Qwen/Qwen2.5-72B-Instruct` | `495f39366efef23836d0cfae4fbe635880d2be31` |

Omitting `--model-path` loads the pinned Hugging Face revision and may download weights. To use a local complete snapshot, pass `--model-path models/mistral` or `models/qwen`. The bundled asset receipts verify the snapshot's relative filenames, hashes, sizes and published weight digests. No credentials are included. Each process loads one model; run the models sequentially when sharing a device.

## Fitting and the original behavioral controls

These commands use all three fixed seeds, 101–103. They save new outputs only.

```bash
.venv-gpu/bin/python -m gpu.train --model mistral --cohort original_1000 --output runs/mistral_original_fit
.venv-gpu/bin/python -m gpu.train --model qwen --cohort original_1000 --output runs/qwen_original_fit
.venv-gpu/bin/python -m gpu.train --model mistral --cohort mapping_300 --output runs/mistral_mapping_fit
.venv-gpu/bin/python -m gpu.train --model qwen --cohort mapping_300 --output runs/qwen_mapping_fit
```

`original_1000` fits intended and pair-swap objectives for one shuffled 1,000-update epoch. `mapping_300` fits six-cycle and pair-swap objectives for 300 replacement-sampled updates. Within each seed, paired objectives share the effective PCA initializer and sampling order. Training saves raw PCA, exact effective pre-step-zero FP32 bases, and fitted bases. The backbone remains frozen. The source/base pair supplies the intervention; the remapped target is a loss label, never a target-donor activation.

Original behavioral reruns are available separately:

```bash
.venv-gpu/bin/python -m gpu.behavior --model mistral --study five_shifts --output runs/mistral_five_shifts
.venv-gpu/bin/python -m gpu.behavior --model qwen --study five_shifts --output runs/qwen_five_shifts
.venv-gpu/bin/python -m gpu.behavior --model mistral --study consequences --split checking --output runs/mistral_consequences
.venv-gpu/bin/python -m gpu.behavior --model qwen --study consequences --split checking --output runs/qwen_consequences
```

These retain both original and answer-prefill interfaces, the five held-out prompt-shift families and the prespecified consequence views. The `five_shifts` study defaults to `lockbox` and also accepts `calibration`; the `consequences` study accepts `discovery` or `checking`. See [the behavioral GPU instructions](BEHAVIOR_GPU.md) for their exact populations and the analysis command for fresh behavioral outputs. Inspect the chosen dataset provenance and resulting run specification when comparing a rerun with a particular paper table.

## Key interventions and mappings

The component commands below use bundled fitted bases by default. Plans can be checked without loading a model:

```bash
.venv-gpu/bin/python -m gpu.components --model mistral --study native-full --plan-only
.venv-gpu/bin/python -m gpu.components --model qwen --study mapping --plan-only
```

The same four study choices work for either model:

| Study | Fixed panel | Native forwards per model | Operation |
|---|---|---:|---|
| `native-full` | 120 cores, 3 fits, 5 consumers | 19,800 | Full native key-only path, reciprocal self/other/signed-null/base arms |
| `native-localized` | Same 120 cores and 3 fits | 63,000 | Full path, previously nominated band, its complement and equal-size control; both frames |
| `fixed-value` | 72 cores, 3 fits, 5 consumers | 16,200 | Six K/V source combinations in each frame, with recipient prefix restoration |
| `mapping` | Shared 108 cores, m1 and m3, 3 fits | 35,640 | Both mappings, full native key-only path and all four arms |

```bash
.venv-gpu/bin/python -m gpu.components --model mistral --study native-full --output runs/mistral_native
.venv-gpu/bin/python -m gpu.components --model qwen --study native-localized --output runs/qwen_localized
.venv-gpu/bin/python -m gpu.components --model mistral --study fixed-value --output runs/mistral_fixed_value
.venv-gpu/bin/python -m gpu.components --model qwen --study mapping --output runs/qwen_mapping
```

Each invocation authenticates the full repository, fixed configuration, complete dataset, basis files and frozen groupwise signed coordinate permutations before forwarding. The default 48-hour deadline is a technical budget limit, adjustable with `--deadline-seconds`; the finite call count is fixed. New output paths are mandatory. No automatic resume or retry is performed after a failed run.

All five consumers are retained: direct location, both observer color tables and both nonobserver color tables. Natural B/S/T anchors are shared across the three fits within a mapping. Fitted P/M interventions reuse a single BF16 complete-event patch made from B/S after decoder block 4 across every consumer. Natural T is a clean calibration anchor and an explicitly declared key/value source in the fixed-value study; it never supplies the fitted layer-4 patch.

Native exchange changes selected critical-token key projections before native RoPE. Queries and values remain endogenous; attention outputs and downstream residuals are not restored. Full paths use all eight KV groups in blocks 6–40 for Mistral or 6–80 for Qwen. Localized masks are copied from the original development nomination and are not selected again. Their complements are complete alternative interventions, not measurements of native necessity.

The fixed-value study is a different estimand. Both frames hold absolute natural T values while varying keys from P/M/S/T or a recipient-specific signed null; B/B is the semantic control. It restores recipient attention-output prefixes through the critical token. It must not be described as native key-only exchange or pooled with that intervention. The full six conditions are retained.

## Bundled controls versus newly trained controls

`gpu/component_data/BASES.json` provides every bundled original intended, PCA and pair-swap basis, plus both 300-update mapping bases. All are rank 16. Mistral's bundled PCA controls are saved exact effective initializers. Qwen's historical story and mapping PCA controls were reconstructed from the recorded initialization procedure; the registry labels that distinction. They are not silently relabelled as exact saved step-zero tensors.

To evaluate newly trained bases, use the entire completed training directory:

```bash
.venv-gpu/bin/python -m gpu.components --model mistral --study mapping --basis-dir runs/mistral_mapping_fit --output runs/mistral_mapping_refit
.venv-gpu/bin/python -m gpu.behavior --model qwen --study consequences --split checking --basis-dir runs/qwen_original_fit --output runs/qwen_consequences_refit
```

The loader verifies the full training inventory, matching model/cohort, all three seeds and all nine effective bases. A new Qwen fit has its own exact saved initializer; it does not recreate the historical reconstructed-control provenance. Evaluating new fits is a new experiment, not replay of the published predictions.

## Outputs and CPU analysis

Successful component outputs contain untouched native result rows and call journals, complete encoded inputs, source/configuration snapshots, actual BF16 K/V and layer-4 payloads, FP32 null plans, scope receipts, and a hash-bound `COMPLETE.json`. All global-token errors and invalid answers are retained. Native self controls must match the original endpoint's entire scientific signature. Actual endpoint fidelity and symbolic correctness are separate CPU metrics.

To reduce storage, recipient attention-output prefix arrays are checked in memory and recorded by exact hashes but are not serialized. This output is sufficient for saved-prediction statistics and retained K/V payload inspection; it is **not** the historical full raw O-prefix archive and cannot support a standalone reconstruction of those omitted arrays. `FAILED.json` preserves a technical failure and available returned evidence. It does not create a success closure.

After a complete run, use the CPU environment for the compact data export:

```bash
python -m reproduce.mechanism --data-dir runs/mistral_native/data --new-evidence --output runs/mistral_native_statistics
```

The CPU module retains all fits, populations, controls and errors and computes the declared paired whole-core intervals. `--new-evidence` explicitly distinguishes new predictions from validation against published counts. It never certifies native tensor execution. Exact prediction identity, recreated training outcomes and performance of this consolidated GPU pipeline remain untested until an actual run completes.

## CPU checks for the GPU code

```bash
python -m unittest gpu.tests.test_components gpu.tests.test_key_scope -v
```

The fixtures check every full-panel finite ledger, all bundled basis hashes, source/target mappings, single-core all-condition execution with deliberately invalid answers, self identity rejection, journal corruption, exact groupwise FP32/BF16 null arithmetic, failed-return evidence, and export compatibility with the CPU reader. Native-style GQA/RoPE doubles check partial-group scope, endogenous downstream Q/V responses, prefix restoration and hook cleanup. Tensor fixtures require CPU Torch; they skip explicitly if Torch is absent. These are implementation checks, not a model replication.
