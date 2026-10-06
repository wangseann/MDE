# Continuous integration

Two GitHub Actions workflows define the CI contract. Both grant only
`contents: read`, cap native math threads, cancel superseded runs, set explicit
timeouts, and retain JUnit results. Official Actions are pinned to full release
commit hashes, and checkout credentials are not persisted after each step.

## Required workflow: `CI`

`.github/workflows/ci.yml` runs for pushes, pull requests, and manual dispatch.

| Job | Coverage | Python | Limit |
| --- | --- | --- | --- |
| `package` | sdist/wheel build, metadata check, clean wheel install, packaged data, lazy backend import, and optional GraphMDE CLI | 3.11 | 20 min |
| `standard-tests` | 69 collected configuration, API, CLI, reverse-MDE, graph, evaluation, compatibility, and fallback tests; optional Torch cases skip here | 3.11, 3.14 | 20 min |
| `torch-backend` | 21 deployed-contract/parity cases plus non-gating repeated timing artifacts using Torch on CPU | 3.11 | 30 min |
| `minimum-pyedm` | exact declared floor and required pyEDM call signatures | 3.11 / pyEDM 2.5.6 | 20 min |
| `scientific-regression` | all five bundled numerical tests | 3.11 | 90 min |

The repository currently declares Python `>=3.11`. Python 3.11 protects the
minimum contract; Python 3.14 detects compatibility issues on the current
stable interpreter. The scientific job runs once to avoid multiplying the
longest computation across the version matrix.

The Torch job pins the official CPU wheel to Torch 2.13.0 and deliberately
hides CUDA. It proves optional-dependency packaging, deployed-behavior
fixtures, numerical parity, chunk invariance, and MDE integration without
pretending that a hosted runner exercised a GPU. The integration cases
separately verify a directional precomputed CCM slope matrix and fixed-seed
live pyEDM EmbedDimension/CCM. They establish unchanged downstream causality
qualification; Torch still accelerates only the candidate Simplex sweep. The
job uploads three `torch-performance-selected-*.json`
reports for 1-D, 2-D, and 25-D sweep stages (`selected = 0, 1, 24`). Each
contains cold and repeated warm CPU-reference/Torch wall durations, median/IQR,
environment, source hashes, memory controls, and per-repeat correctness.
Timing is diagnostic and never gates CI.

## pyEDM dependency floor

Current MDE passes `kdWorkers` to both `pyEDM.Simplex` and
`pyEDM.EmbedDimension`. Published pyEDM 2.5.0 lacks those parameters and fails
two compatibility-contract tests; the next published release, 2.5.6, provides
them. `pyEDM>=2.5.6` is therefore a runtime requirement rather than a routine
dependency refresh. The exact-minimum job prevents future source changes from
silently raising that floor.

## Scheduled workflow: `Extended validation`

`.github/workflows/validation.yml` runs each Monday at 03:17 UTC, on manual
dispatch, and on pushes to `main` and `integration/upstream-*` branches so
upstream merges exercise the independent reference suite before review and
again on the merged commit. It pins
`pao-unit/EDM_MDE_validation` by full commit:

- `external-pyedm`: all 31 independent pyEDM tests, with shared sample data
  restored between cases so an upstream state leak cannot alter later files;
- `external-mde`: both independent MDE tests, using the documented
  legacy-keyword adapter; both must pass against their pinned goldens.

Both jobs pin pyEDM 2.5.6, the reference runtime used when this validation
snapshot was integrated. pyEDM 2.5.7 changes neighbor tie resolution and
exclusion masking; five external EmbedDimension cases fail their frozen
exact-value assertions with that release. The reference outputs and
assertions remain unchanged. Required package, standard, and bundled
regression jobs continue to resolve the current supported pyEDM release.

The external suite is deliberately separate from pull-request CI because it
duplicates upstream pyEDM coverage and contains expensive numerical cases. A
failure is still actionable: dependency updates and GPU backend changes must
not silently alter these reference results.

## Branch protection

After the local fork is published and the workflows have completed once, mark
these `CI` jobs as required checks on `main`:

- Build and install package;
- Standard tests (Python 3.11);
- Standard tests (Python 3.14);
- Optional Torch backend (CPU parity and timing);
- Minimum pyEDM compatibility;
- Bundled scientific regressions.

Do not require the scheduled workflow for every pull request. Run it manually
before dependency-floor changes, numerical algorithm changes, releases, and
accelerated-backend merges.
