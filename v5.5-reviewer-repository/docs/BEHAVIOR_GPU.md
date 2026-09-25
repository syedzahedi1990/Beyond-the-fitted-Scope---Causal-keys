# Behavioral evaluation and optional refitting

Use the separate pinned GPU environment in [GPU_REPRODUCTION.md](GPU_REPRODUCTION.md). These commands execute models and consume GPU time. They are not run by the CPU reproduction command or GitHub workflow. A new output directory is required for every run; exceptions leave a `FAILED.json` record and completed rows intact.

## Evaluate the published bases

The default is the complete 1,500-pair held-out population, both response interfaces, all three intended/alternative fits, paired PCA controls, clean/source endpoints, full-event transfer and execution identity controls:

```bash
python -m gpu.behavior --model mistral --study five_shifts --output runs/mistral-five-shifts
python -m gpu.behavior --model qwen --study five_shifts --output runs/qwen-five-shifts
```

The complete 120-core checking panel has eight questions. Every materialized event patch is held fixed across questions and interfaces:

```bash
python -m gpu.behavior --model mistral --study consequences --output runs/mistral-consequences
python -m gpu.behavior --model qwen --study consequences --output runs/qwen-consequences
```

Use `--split calibration` for the five-shift 300-pair calibration population or `--split discovery` for the seven-question 120-core discovery panel. No cases are screened out using fitted outcomes. All six account forecasts are saved; the new runner does not change the historical discovery decision or nominate a replacement account.

Outputs contain each case's candidate log probabilities, candidate ranking, candidate mass, actual global token ID, semantic global answer when valid, patch hashes and identity-control differences. These fields deliberately distinguish candidate success from global-next-token success. Noncandidate outputs remain invalid semantic answers, not guessed labels.

Analyze a completed model rerun in the CPU environment:

```bash
python -m reproduce.behavior_rerun --run runs/mistral-five-shifts --output rerun-statistics
```

The adapter authenticates the completed rows, checks the exact story grid, and reuses the published behavioral estimators. It accepts a complete lockbox or either complete consequence panel, reports new results without enforcing historical success values, and preserves both response interfaces. Calibration rows remain available as diagnostics without being presented as the 1,500-pair main table.

The exact original training and evaluation records are packaged in `gpu/behavior_data/datasets.json.gz`. Provenance maps each part to its saved source digest. The training pool used for the 300-update mapping cohorts is the same ordered 1,000 records; replacement versus shuffled-epoch sampling is separate.

## Refit the interventions

Refitting is optional because the published bases are supplied. Each command fits both declared objectives at all three fixed seeds. It freezes all model weights, uses rank 16 at block four over the complete event, and saves each effective paired PCA initializer before optimization.

```bash
python -m gpu.train --model mistral --cohort original_1000 --output runs/mistral-original-fits
python -m gpu.train --model qwen --cohort original_1000 --output runs/qwen-original-fits
python -m gpu.train --model mistral --cohort mapping_300 --output runs/mistral-mapping-fits
python -m gpu.train --model qwen --cohort mapping_300 --output runs/qwen-mapping-fits
```

`original_1000` fits intended transfer and the pair-swap: every pair is visited once in the seed-specific order. `mapping_300` fits the six-cycle and pair-swap: 300 draws with replacement use `random.Random(seed + 16000)`. The fixed seeds are 101, 102 and 103. AdamW uses learning rate 0.001, zero weight decay and gradient clipping at norm 1. Candidate cross-entropy is computed in FP32; projection operands use BF16 as in evaluation. PCA uses a centered rank-32 sketch and two power iterations. A CPU QR initializes the trainable parameter, and the actual device QR basis is saved as the paired initializer.

Every fit records its schedule, loss, gradient norms and finite connected-gradient checks. The three technical probes compare differentiable and materialized scores; they do not apply an efficacy threshold. The original fitted tensors are not overwritten. A fresh run's exact saved PCA is not evidence of identity to the unavailable historical Qwen initializer.

To evaluate freshly fitted original-cohort bases:

```bash
python -m gpu.behavior --model mistral --study consequences --basis-dir runs/mistral-original-fits --output runs/mistral-refitted-consequences
```

The basis loader requires a completed run of the matching model/cohort and authenticates every requested basis. GPU component evaluation also accepts the completed fitting directory for a matching cohort, as described in its instructions.

## Execution details and limits

Mistral behavioral runs use the system message “You are a helpful assistant.” and 1,024-token right padding. Qwen uses its original user-only chat-template call, unpadded five-shift/training inputs and 256-token consequence padding. No truncation or event-span alignment fallback is allowed. The assistant `Answer:` suffix is applied after the native chat boundary. All choices must be single leading-space tokens with prefix-preserving appends.

The behavioral forward adapter is newly consolidated from the published execution paths. Its projection, prompt/span logic, sampling, fixed-patch identity and differentiable-versus-materialized paths have CPU tests, including an actual small PyTorch model. Large-model GPU equivalence has not been rerun during this release preparation. The supplied prediction evidence remains the source for exact historical statistics; a fresh GPU result must be evaluated and compared, not presumed identical.

Supporting historical 7B rank pilots, breadth screens, feasibility failures and final-answer head screens have CPU evidence and analysis in this repository. They do not receive new GPU launchers here.
