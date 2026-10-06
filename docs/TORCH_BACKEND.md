# Torch candidate-sweep backend

This backend makes the expensive greedy Simplex candidate sweep available as
an optional Torch operation while leaving ordinary MDE use unchanged.

## Provenance

The production prototype came from:

```text
fmri-edm-ccm/planB_audiojepa/scripts/mde_sweep_backend_probe.py
SHA-256 eb1e0f82baf0474fc4794d94898e9948db174c8acd8ca9c7cbcabc8bc7df63c0
```

`driving-MDE/Scripts/run_driving_gpu_mde.py` requires that file through
`--backend-script`, loads it with `importlib`, calls
`candidate_sweep_torch()` once per greedy dimension, and uses
`simplex_1d_predict()` or `simplex_predict()` for the final prediction. Its
result records the resolved backend path, SHA-256, function, device, and batch
controls. The nested driving runner uses the same candidate-sweep interface.
The fMRI shadow-comparison runner additionally imports `eval_cols()`,
`candidate_sweep_cpu_reference()`, and `compare_rows()` from the same file.

`dimx/TorchBackend.py` extracts that array-level interface without either
project's data loading, split construction, CLI, or repository paths. It keeps
all six consumer/reference callables and their signatures so existing direct-
file loaders can point at the installed module. During the integration audit,
the function bodies for `rho()`, the six exported callables, and the deployed
source were verified as AST-equivalent. Only the module wrapper, formatting,
and provenance constants differ. In particular, this first PR does not change
buffering, synchronization, timing, validation order, tie handling, or the
deployed 1-D exact-match behavior.

Python 3.11 CI locks the combined numerical-function AST fingerprint to
`3de7fd2abc7719f5e93241ad12270a9a6d9acd6bdc6fd93b224aec87f9f4d787`.
The fingerprint test is skipped on newer Python AST schemas; numerical parity
still runs there.

The source artifact hash is also available as
`dimx.TorchBackend.__prototype_sha256__`. The packaged extraction in this PR
has SHA-256
`8f7f4b39daca7bc8418f4f7bcfec944cdfb980da419ca20f2b0546afe7a6b89b`.

## Install

CPU-only MDE needs no new dependency. From a source checkout, install the
optional backend with:

```console
python -m pip install -e ".[torch]"
```

After publication, the corresponding installation is:

```console
python -m pip install "dimx[torch]"
```

The extra declares `torch>=2.1`; it cannot choose the correct CUDA build for
every host. GPU users should use the
[official PyTorch package selector](https://pytorch.org/get-started/locally/)
for the CUDA build matching their driver and platform, then verify that CUDA is
visible in the same Python environment.

## Use from MDE

The default stays on the existing multiprocessing/pyEDM path:

```python
from dimx import MDE

mde = MDE( data, target = "FWD", D = 5 )
mde.Run()
```

Require a CUDA sweep with a minimal configuration change:

```python
mde = MDE(
    data,
    target = "FWD",
    D = 5,
    crossMapBackend = "torch",
)
mde.Run()
```

The defaults select `torchDevice = "cuda"`, 16 candidates per batch, and 128
prediction rows per chunk. Override those controls only for device selection or
memory tuning. Use `torchDevice = "cuda:1"` to select a particular visible GPU,
or `torchDevice = "cpu"` for backend parity tests. With
`crossMapBackend = "auto"`, MDE attempts the requested Torch device and falls
back to its CPU pool when Torch or the device is unavailable, or when the
input/split contract is unsupported. Runtime failures after selection, such as
device out-of-memory, remain visible rather than being silently retried.

The same controls are available on the MDE command line. The minimal opt-in is
`--crossMapBackend torch`; optional tuning uses `--torchDevice`,
`--torchBatchCandidates`, and `--torchPredChunk`.

## CCM and precomputed slope matrices

Torch changes only the candidate Simplex sweep that supplies cross-map rho
values. With a precomputed slope matrix, MDE performs the same directional
`slopeMatrix.loc[candidate, target]` lookups and runs neither `EmbedDimension`
nor CCM. Without a matrix, and with `noCCM = False`, MDE runs pyEDM
`EmbedDimension` when `E = 0`, then runs CCM for candidates that pass the
embedding gate. `MDEOut.rho` remains cross-map rho; it is not a CCM slope.

Focused parity tests run CPU and Torch MDE against the same asymmetric matrix,
including a highest-rho candidate that fails the slope threshold, and against
the same live pyEDM calculation with a fixed CCM seed. They require identical
selected-variable order, embedding dimensions, and CCM slopes, with the
documented tolerance applying only to cross-map rho. These synthetic tests
verify execution-path parity; their small CCM sample is not a scientific
estimate or a GPU performance measurement.

## Standalone consumer compatibility

The installed backend's direct path is available without copying a private
project file:

```python
import dimx.TorchBackend

backend_script = dimx.TorchBackend.__file__
contract_version = dimx.TorchBackend.__driving_backend_contract__
```

For example, the existing driving loader can receive that value as its required
`--backend-script` argument. A fMRI runner can either import
`dimx.TorchBackend` directly or resolve the same `__file__` in its existing
direct-file loader. The packaged module also retains the fMRI shadow runner's
three CPU-reference/comparison helpers. Consumers should record the installed
file's SHA-256 and the dimx commit or release in every scientific result; the
prototype and packaged hashes above distinguish provenance from the exact file
that ran.

## Scientific support boundary

The accelerated operation is multivariate, already-embedded Simplex candidate
evaluation. It preserves candidate input order, uses the classical
`D + 1` neighbors, applies exponential nearest-neighbor weights, aligns targets
by `Tp`, and returns ordinary Python scalars. Candidate batches and prediction
chunks are memory controls and must not change results.

The MDE adapter currently requires:

- real, finite library, prediction, and target arrays;
- valid inclusive, one-offset MDE `lib` and `pred` range pairs;
- enough library and observed prediction rows for the requested dimension;
- library and prediction rows separated by more than `exclusionRadius`.

Missing or non-finite data and lib/pred proximity requiring per-query temporal
exclusion remain on pyEDM. Explicit `crossMapBackend = "torch"` rejects these
cases; `"auto"` falls back before the sweep. The standalone array kernel cannot
prove a scientifically valid split, so driving/fMRI callers remain responsible
for disjoint LIB/PRED/TEST construction and for selecting dimensions without
looking at TEST. MDE's adapter enforces the lib/pred row-separation condition it
can observe.

Torch accelerates no CCM, `EmbedDimension`, time-delay construction, or final
inference stage. Those operations retain their existing project-specific or
pyEDM implementation.

The deployed 1-D final-prediction helper assigns equal weight to two neighbors
when its nearest distance is exactly zero. That behavior is retained here for
compatibility. Changing it may be scientifically reasonable, but it would be a
separate numerical-change PR with its own validation rather than part of this
packaging and CI integration.

## Numerical evidence and CI

The kernel computes distances and predictions in `float32`; pyEDM/SciPy may
use `float64`. Hosted parity tests therefore use absolute and relative rho
tolerances of `2e-5`, verify the selected candidate, preserve candidate input
order, and require invariance across batch/chunk boundaries. At the adapter
boundary, MDE rounds finite backend rho values to six decimal places, matching
the existing cross-map result precision.

Exact equality is not promised. Equal-distance neighbor ties, nearly equal
candidate scores, or a value close to a scientific threshold can resolve
differently across Torch versions and devices. Reproducible reports should
record the selected backend, device, Torch version, dimx version/commit,
installed backend SHA-256, batch sizes, data/split manifest, and candidate
ranking—not only the winning column.

Ordinary hosted CI covers the unchanged CPU default, lazy import and fallback,
deployed compatibility fixtures, and Torch-on-CPU numerical parity. It also
runs a deterministic repeated benchmark and uploads
three `torch-performance-selected-*.json` artifacts spanning 1-D, 2-D, and
25-D sweep stages. Reports include source hashes, environment, cold start,
shape and memory controls, per-repeat candidate/rho parity, internal and wall
durations, median/IQR, and the diagnostic CPU-reference/Torch ratio. CI pins
Torch 2.13.0 and one Torch thread so runs remain interpretable over time.
This ratio compares the candidate kernels, not a complete multiprocessing
`MDE.Run()` wall clock, so it must not be presented as end-to-end user speedup.

The timing artifact has no speed threshold: shared hosted-runner load is too
variable for a scientifically defensible performance gate. The standard
`ubuntu-latest` runner has no CUDA device, so its report does not establish GPU
speed. Run the same harness on labelled hardware with:

```console
python ci/benchmark_torch_backend.py \
  --device cuda \
  --selected 24 \
  --warmups 2 \
  --repeats 5 \
  --torch-threads 1 \
  --output torch-performance-cuda.json
```

The CUDA report must be retained with the PR and interpreted using its recorded
device/software metadata. A
[GitHub-managed GPU larger runner](https://docs.github.com/en/actions/reference/runners/larger-runners#specifications-for-gpu-larger-runners)
requires an eligible organization plan and runner configuration; a self-hosted
runner can instead be targeted by explicit labels. CI does not declare such a
runner automatically, because an unavailable label would leave the workflow
queued indefinitely.
