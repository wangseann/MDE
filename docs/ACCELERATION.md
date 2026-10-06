# Acceleration integration contract

The Torch backend is an experimental, opt-in acceleration of MDE's greedy
candidate cross-map sweep. The established pyEDM implementation remains the
default and the scientific reference.

## Stable default

Existing code keeps its behavior and does not import Torch:

```python
mde = MDE( data, target = "FWD", D = 5 )
mde.Run()
```

Set `crossMapBackend = "torch"` to require Torch, or `"auto"` to use it only
when the dependency, requested device, data, and split semantics are supported.
An explicit request fails clearly when it cannot be honored; `auto` falls back
to the CPU pool during backend resolution. The default is deliberately
`crossMapBackend = "cpu"` while parity is being established.

Torch replaces only the repeated Simplex candidate sweep in `Run()`. Embedding
dimension selection, CCM qualification, slope-matrix handling, thresholds, and
output construction continue through the same pyEDM/MDE code.

## Implementation choice

The numerical kernel was ported from the backend actually shared by the
`fmri-edm-ccm` and `driving-MDE` projects, not from the older kEDM experiment.
Its source artifact is
`fmri-edm-ccm/planB_audiojepa/scripts/mde_sweep_backend_probe.py`, SHA-256
`eb1e0f82baf0474fc4794d94898e9948db174c8acd8ca9c7cbcabc8bc7df63c0`.
The packaged module is a near-verbatim extraction: all seven numerical and
reference function bodies are AST-equivalent to that artifact. Project-specific
loaders and its probe CLI are omitted. MDE eligibility checks and CPU fallback
live in the thin `TorchCrossMap` adapter, so they do not alter the backend used
by the existing driving and fMRI consumers.

The kEDM route shadowed the complete `pyEDM` module and replaced `Simplex`,
`CCM`, `EmbedDimension`, `Embed`, and `ComputeError`. It belongs to an older
non-packaged MDE layout, predates the current `MDEConfig`, parallel runner, and
`ReverseMDE`, and does not match the implementation used by the two consumer
projects. Replacing only the measured candidate-sweep seam keeps the remaining
scientific pipeline on its established implementation.

## Correctness checks and performance observations

Merge evidence must include:

- unchanged behavior from the default CPU call and no mandatory Torch import;
- candidate order and rho agreement with the scalar Simplex reference at the
  documented tolerance;
- repeatability across candidate-batch and prediction-chunk sizes;
- clear errors for an explicitly unavailable dependency, device, or unsupported
  scientific setting, plus tested `auto` fallback;
- unchanged `MDEOut` schema and continued pyEDM coverage for CCM and embedding
  dimension selection.

Correctness and parity checks are required. Hosted CI also records repeated CPU
reference and Torch-on-CPU timings as a downloadable JSON diagnostic. Shared-
runner timing is variable, so neither duration nor speedup affects pass/fail.
That report is not evidence of CUDA execution or GPU speed.

CUDA measurements run manually on labelled hardware and record the backend
hash, device and software versions, data shape, warm-up, batch/chunk settings,
and repeated-trial median/IQR. They remain non-blocking until stable dedicated
hardware and an agreed statistical regression policy exist.

See [Torch backend](TORCH_BACKEND.md) for installation, use, provenance, and
the current scientific support boundary.
