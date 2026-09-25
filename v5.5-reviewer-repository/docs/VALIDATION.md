# Validation and reproduction scope

This release separates recalculating saved predictions from rerunning the language models. The former is verified locally; the latter has implementation and CPU tests but was not executed on a GPU during repository preparation.

## Verified CPU calculations

- The behavioral replay independently calculates the main response-interface table, all eight learned/PCA comparison rows and their paired intervals, checking-panel counts, output recoding, consistency, opportunity subsets and supporting negative screens.
- The mechanism replay reads 229,680 saved returns across six study/model panels. It reconstructs full grids, symbolic oracles and paired statistics. Its 11,712 reference comparisons cover the main effects and pointwise intervals, all fit estimates and selected counts across retained conditions and populations. The largest observed discrepancy is approximately 2.84×10^-14, within the declared 2×10^-12 rounding tolerance.
- Both fixed-value cohorts independently reproduce the 24/48 ceiling by enumerating all 6^6 fixed six-colour functions, checking the actual PCA premise and balanced source population separately.
- The display code uses these recomputed results. Its native and mapping table rows reproduce the manuscript's displayed values; the figure is a fresh rendering of the same estimands.

Reference files contain expected validation values and provenance. Analysis functions do not read them to calculate results. Default published-evidence commands fail on mismatches. Explicit fresh-evidence analysis retains input/grid checks while allowing genuinely different outcomes.

## Implementation checks

The 51-test suite covers published numerical claims; fixed populations and schedules; complete condition grids; exact versus reconstructed basis identities; invalid global outputs; missing-margin propagation; error copying; observer/nonobserver joint outcomes; inverse six-cycle recoding; whole-core bootstrap units; source/data corruption; component hook scope and cleanup; fixed-patch identity; and connected intervention gradients in a small CPU PyTorch model.

PyTorch-dependent CPU tensor tests are optional when only `requirements.txt` is installed. To run all 51 tests locally, also install PyTorch 2.11.0 for the local CPU platform. The GitHub workflow installs that optional dependency before running the tests. These small-model/tensor tests are not full Mistral or Qwen executions.

The released code contains no Python comments or module/class/function docstrings. The public payload is checked for personal absolute paths, credentials and obsolete remote-machine controls. Release files have SHA-256 and byte-size bindings in `RELEASE.json`; prediction and basis metadata also retain their own evidence bindings.

## GPU validation boundary

Native key/relay engine numerical bodies are derived from the original executed implementation. Model revisions, geometry, BF16 operands, SDPA source checks, payload identities, matched-null construction, consumer-prefix checks and finite call plans remain explicit. The derived release source identities replace historical source-file seals; they do not retroactively change historical provenance.

The new consolidated behavioral/fitting entry points preserve the declared prompts, padding, full event-span operator, paired training schedules and optimizer settings. They have CPU functional tests, but their large-model numerical equivalence has not been demonstrated by another GPU run in this preparation pass. The commands produce new evidence and retain technical failures instead of silently retrying or screening outcomes.

Historical native activation caches, optimizer journals and full tensor inventories are not included. The compact evidence authenticates projections of saved outputs against recorded source digests; it is not an independent rerun of those original executions. The documented Qwen fixed-value closure-recovery exception remains in its evidence metadata.

## Environment

The CPU clean-copy workflow uses Python 3.12.14, NumPy 2.3.5 and Matplotlib 3.10.7 on macOS ARM64. It requires no files outside the repository beyond the installed dependencies. The GPU environment is separately specified in `gpu/requirements.txt` and the GPU instructions. Linux GitHub Actions execution can only be observed after upload; configuring the workflow is not a claim that a hosted CI run has already passed.
