# Testing

MDE uses layered tests so quick API failures are separated from longer
scientific validation.

## Test ownership

| Suite | Owner/source | Unique cases | Role |
| --- | --- | ---: | --- |
| Bundled scientific regressions | Pre-existing `pao-unit/MDE` tests | 5 | Authoritative MDE numerical behavior |
| Fast CI-foundation tests | Added before acceleration work | 40 | Configuration, API, CLI, orchestration, dependency, and external adapter behavior |
| Torch acceleration tests | Added with the opt-in backend | 24 | Configuration, deployed consumer contract, fallback/error handling, numerical parity, CCM-path parity, and MDE integration |
| Independent validation | Pinned `pao-unit/EDM_MDE_validation` | 33 | External pyEDM and MDE conformance |

The five bundled tests retain upstream ownership. Three MDE golden tests were
strengthened to compare selected-variable identity and order, and the existing
CCM-matrix test now compares the three dimensions stated in its original
comment. The external test source and golden files are not modified.

## Reproducible environment

Install the project and its test extra in an isolated Python 3.11+ environment:

```bash
python -m pip install --editable ".[test]"
```

For deterministic, headless execution use:

```bash
export MPLBACKEND=Agg
export MPLCONFIGDIR="$PWD/.ci/matplotlib"
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0
mkdir -p "$MPLCONFIGDIR"
```

The native-thread limits prevent oversubscription when MDE or pyEDM starts
multiple Python processes. They are performance controls, not numerical
tolerances.

## Fast standard tests

```bash
python -m pytest -q \
  tests/test_Config.py \
  tests/test_MDE_Unit.py \
  tests/test_ReverseMDE.py \
  tests/test_CLI_Parser.py \
  tests/test_pyEDM_Compatibility.py \
  tests/test_ExternalValidationAdapter.py \
  tests/test_TorchBackend.py \
  tests/test_TorchCrossMap.py \
  --timeout=120 --durations=20
```

These tests should finish in seconds to roughly one minute and are run on the
minimum supported Python and the newest supported Python.

## Optional Torch parity

Install a CPU Torch build and run the focused backend files on Python 3.11:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --editable ".[test]"
CUDA_VISIBLE_DEVICES="" python -m pytest -q \
  tests/test_TorchBackend.py tests/test_TorchCrossMap.py \
  --timeout=180 --durations=20
```

These 21 cases exercise the deployed direct-file consumer contract, lazy
imports, deployed edge behavior, batch/chunk invariance, `Tp = -1, 0, 1`,
finite-data and split fallback, rho parity with pyEDM, and complete MDE
selections with no CCM, a directional precomputed slope matrix, and fixed-seed
live pyEDM CCM. The matrix case proves that live `EmbedDimension` and CCM are
bypassed; the live case proves that both sweep backends reach identical
embedding dimensions and rounded CCM slopes. It runs genuine pyEDM CCM
serially to avoid testing process scheduling twice. The remaining three
acceleration cases validate invalid backend controls in the standard suite.
Torch does not compute CCM in either mode: it accelerates the candidate Simplex
sweep and leaves causality qualification in existing MDE/pyEDM code. Passing
here proves the Torch algorithm on a CPU device; it does not prove CUDA
execution or speed.

## Diagnostic performance report

Run the same non-gating benchmark used by the Torch CI job:

```bash
for selected in 0 1 24; do
  python ci/benchmark_torch_backend.py \
    --device cpu \
    --selected "$selected" \
    --warmups 2 \
    --repeats 3 \
    --torch-threads 1 \
    --output "torch-performance-selected-${selected}.json"
done
```

The JSON records the extracted and prototype hashes, Python/Torch/device and
GitHub-run metadata, input shape, cold start, warm-up and memory controls,
CPU/Torch correctness for every repeat, individual internal and wall durations,
median/IQR, and median ratio. The harness fails for a candidate or rho mismatch,
but never for being slower than a threshold. CI pins Torch 2.13.0; use the same
version when comparing reports longitudinally.

The ratio is a candidate-sweep diagnostic against the deployed scalar SciPy
reference. It is not a full `MDE.Run()` speedup claim; end-to-end timing must be
measured separately on the same data, split, process settings, and hardware.

Use `--device cuda` only on a machine where the selected Python environment can
see CUDA. Retain the resulting JSON with any speed claim. A standard GitHub
hosted runner has no CUDA device, and its variable CPU load makes its timing
artifact diagnostic rather than a stable regression baseline.

## Bundled scientific regressions

```bash
python -m pytest -q \
  tests/test_MDE.py tests/test_MDE_CCM_Matrix.py \
  --timeout=1800 --durations=20
```

All five repository-owned tests run on pushes and pull requests. Do not loosen
their rounding or equality rules simply to accommodate a platform difference;
first identify whether the difference is deterministic and scientifically
acceptable.

## Independent validation

The scheduled workflow checks out the pinned validation repository rather than
copying it. To reproduce it locally:

```bash
git clone https://github.com/pao-unit/EDM_MDE_validation.git external-validation
git -C external-validation checkout bae270e568dd52830f57ab661380e700097aa58d

python -m pytest -q -p ci.external_validation_adapter \
  external-validation/test_Simplex.py \
  external-validation/test_SMap.py \
  external-validation/test_CCM.py \
  external-validation/test_EDim.py

python -m pytest -q -p ci.external_validation_adapter \
  external-validation/test_MDE.py
```

The first command runs all 31 pyEDM cases without editing their source. The
adapter restores pyEDM's shared Lorenz sample after every case because the
external `test_simplex7` modifies it in place; with test isolation all 31 pass.
The second command runs both MDE cases with the documented legacy-keyword
translation; both pass. Its Lorenz case protects the rule that a leading time
column is removed exactly once, leaving all four candidate variables available.
See [test provenance](UPSTREAM_BASELINE.md).

## Packaging

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

CI additionally installs the built wheel outside the source checkout and
checks `dimx.__version__`, both packaged example data files, and that the
direct-loadable `dimx.TorchBackend` module is present without requiring Torch.
