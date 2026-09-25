# Beyond the Fitted Answer

Code, synthetic datasets, saved intervention bases and prediction evidence for **Beyond the Fitted Answer: Attention Keys and the Consequences of Learned Interventions**, manuscript v5.5.

The main question is how a learned location-remapping intervention produces consequences beyond the answer used to fit it. The experiments compare learned and PCA patches, audit untrained consequences, and exchange their induced attention keys. Endpoint fidelity, symbolic correctness and preservation are reported separately throughout.

## Reproduce the reported results on CPU

Use Python 3.12 and a fresh environment, from this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m reproduce --output reproduced
```

The last command verifies release files, recomputes statistics from saved per-example predictions, checks published reference values, and produces tables plus the native-key effect figure. It performs no model inference, model download or training. It refuses to overwrite an existing output directory.

Start with:

- `reproduced/behavior/behavior_tables.md`: response-interface results, PCA comparisons and consequence counts.
- `reproduced/figures/main_mechanism_tables.md`: native key-only and mapping-replication outcomes, with fidelity, correctness and preservation together.
- `reproduced/figures/native_key_effects.png`: reciprocal effects with intervals and all three fits.
- `reproduced/mechanism/mechanism_conditions.csv`: every retained condition, population and per-fit metric, including score effects.
- `reproduced/COMPLETE.json`: successful end-to-end CPU reproduction receipt.

Full machine-readable results are in the two result subdirectories. This is a recalculation from prediction evidence, not a replay of the language models or an authentication of omitted activation caches. See [the result map](docs/PAPER_RESULTS.md) and [validation record](docs/VALIDATION.md) for the exact scope.

## Rerun model experiments on GPU

The separate [GPU instructions](docs/GPU_REPRODUCTION.md) describe pinned model revisions, environment installation, hardware requirements, the saved-basis evaluation commands and optional fitting. GPU execution uses a different environment from CPU analysis. Model weights are external; the small intervention bases are included.

The released GPU entry points are:

```bash
python -m gpu.behavior --help
python -m gpu.components --help
python -m gpu.train --help
```

These are a consolidated implementation. The validation record distinguishes CPU checks from native GPU reruns; the release does not claim that GPU calculations were repeated during packaging. Fresh fits receive their own saved initializers and provenance, and must not be identified with historical fitted tensors.

## Repository layout

| Path | Purpose |
|---|---|
| `reproduce/` | Independent CPU calculations and figure generation |
| `data/behavior/` | Compact behavioral and supporting-study prediction evidence |
| `data/mechanism/` | Compact endpoint and exchange predictions, scores and core definitions |
| `gpu/` | Shared model execution, exact input stories, frozen experiment configurations and fitted bases |
| `expected/` | Published-result references used only for validation |
| `tests/` | Data, estimand, integrity and intervention tests |
| `docs/` | Result mapping, methods, GPU recipes and validation scope |
| `RELEASE.json` | SHA-256 identities of the released files |

No historical archive, model weights, remote-machine controls or activation caches are included. Code comments and docstrings are omitted; explanations live in these documents. The evidence retains failures, invalid outputs, all fixed fits and the declared populations.

## Interpretation and provenance

The eight-question audit, the 72-core fixed-value experiment and the 120-core native key-only confirmation are separate settings. The 108-core mapping replication is a further, separate cohort. Shared semantic cores across models are not independent additional stories. Mistral controls use exact saved initializers; historical Qwen PCA controls are reconstructed. Their identities are retained in the basis registry and result documentation.

Intervals condition on the three saved fits. Fidelity can copy endpoint errors; a positive fidelity effect is not automatically a correctness gain. The fixed-value colour-recoding ceiling is not applied to the six-cycle mapping. Supporting failed screens are included as results and do not become success gates for the later experiments.

Please cite the accompanying v5.5 manuscript when using these experiments. No DOI, publication venue or acceptance status is asserted by this repository.

## License

A distribution license has not yet been selected. Pretrained models and external dependencies retain their own terms.
