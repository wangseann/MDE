# Upstream baseline and test provenance

The CI foundation was created from a clean clone of
[`pao-unit/MDE`](https://github.com/pao-unit/MDE) rather than from an existing
working directory.

| Source | Branch | Pinned revision | Role |
| --- | --- | --- | --- |
| `pao-unit/MDE` | `main` | `ae5b1ac0b9e7c443e6dd777a13d552e431c67382` | Integrated v1.4.2 source and bundled tests |
| `pao-unit/EDM_MDE_validation` | `main` | `bae270e568dd52830f57ab661380e700097aa58d` | Independent numerical validation |

The initial CI foundation used `77ca953799b6790403a052d4ce1a19c61e118a6c`,
recorded on 2026-08-27. The current source pin was integrated on 2026-10-06.
Updating either pin requires a review of changed tests, golden outputs,
dependencies, and runtime before the workflow is updated.

## Upstream v1.4.2 integration

The fork merges upstream commits `6b587fb` (v1.4.1) and `ae5b1ac` (v1.4.2)
on top of the Torch integration at `c956b66`. The merge retains both parent
histories, both CI workflows, the optional Torch backend, and the pyEDM 2.5.6
dependency floor. The deployed Torch numerical kernel is unchanged.

Upstream removes `removeTime`; callers now use `noTime=False` when the first
column is time and `noTime=True` when every column is data. Both CPU and Torch
receive the same numeric frame after this single preprocessing step. The
upstream GraphMDE application and `graph` extra coexist with the fork's `test`
and `torch` extras.

Evaluate now aligns targets by `Tp` and rejects overlapping library/prediction
windows. Its bundled test starts prediction at row 302 rather than 301 for
`Tp=1`; the reference CSV is taken unchanged from upstream `ae5b1ac`, not
regenerated to fit the fork. Upstream's restored four-row Lorenz golden is
already identical to the fork's corrected reference.

Focused integration regressions also cover two upstream edge cases. GraphMDE
rejects self-loops even when the node has not yet been added. Evaluate keeps
PCA/DMap training targets inside the specified library, matching Simplex's
`Tp` truncation; prediction targets retain upstream's alignment. GraphMDE's
root-derived default row limit is preserved and stated explicitly in CLI help.

The cproj accepted baseline, certified runtimes, and cluster releases are
separate records and are not advanced by this GitHub integration.

## Upstream ancestry reconciliation

On 2026-09-01, upstream PR #2 unintentionally merged this fork's CI branch as
`e532982`, then restored `main` with additive revert `64444b0`. The revert's
tree is exactly identical to the original `77ca953` baseline, so that
reconciliation did not change the scientific baseline or validation pins.

This fork records `64444b0` as an ancestor with a history-only merge. A normal
content merge would reapply the upstream revert and remove the intentional CI
changes. The history-only merge instead preserves the validated fork tree from
`cd16853` / `dbd0a1b`; this paragraph is its only file-level change.

## Bundled MDE tests

The five pre-existing tests remain in their original files and all run in the
required scientific-regression CI job:

1. Lorenz5D MDE golden output.
2. Fly80 forward-motion MDE golden output.
3. Fly80 left/right-motion MDE golden output.
4. Fly80 `Evaluate` prediction golden output.
5. Internal CCM versus a precomputed CCM slope matrix.

These are integration-level numerical regressions. The added fast tests cover
configuration, validation failures, CLIs, reverse traversal and output, and the
pyEDM call interface without replacing the scientific suite. The three MDE
golden tests now assert selected-variable order as well as rho, and the slope
matrix test compares the three dimensions its original comment specified.

The v1.2.0 multiprocessing refactor (`69c59ae`) introduced a second leading-
column removal after `removeTime=True` had already removed the time vector. On
Lorenz5D this silently removed `V1`, and the same commit deleted the fourth row
from the bundled golden. The pre-refactor bundled golden and independent
validation golden both require `V1,0.976646`. This fork makes numeric-frame
preparation the sole owner of leading-column removal, leaves the input frame
unchanged, and restores that fourth row. After adopting upstream v1.4.1's
removal of `removeTime`, fast tests cover both `noTime` modes and require that
neither the input frame nor the configuration is mutated.

## Independent validation suite

`EDM_MDE_validation` contains 33 pytest cases and 31 golden-output files:

- 14 `Simplex` tests;
- 4 `SMap` tests;
- 6 `CCM` tests;
- 7 `EmbedDimension` tests;
- 2 MDE tests.

The reference workflow pins pyEDM 2.5.6. With pyEDM 2.5.7, external
EmbedDimension cases 1, 3, 4, 6, and 7 disagree with their exact goldens.
That release changed
[neighbor tie resolution and exclusion masking](https://github.com/SugiharaLab/pyEDM/compare/v2.5.6...v2.5.7).
The frozen suite is evaluated with its original runtime; reference files and
assertion tolerances are not changed to accommodate dependency drift.

The validation repository has no license file, so its source is not vendored
here. CI checks out the pinned commit and runs every test. Its two MDE tests
predate `MDEConfig` and still pass `cores`, `title`, and `removeTime`.
The small pytest hook in
`ci/external_validation_adapter.py` translates `cores` to `crossMapCores` and
removes the unused plotting title at runtime. A test-scoped MDE wrapper maps
`removeTime=True` to `noTime=False` after each test constructs its arguments.
It does not alter the public MDE API. The external tests and reference outputs
remain unchanged.

The external `test_simplex7` inserts NaNs directly into the shared
`pyEDM.sampleData["Lorenz5D"]` frame. If all files run in one process without
isolation, those NaNs change `test_smap4`, `test_ccm5`, and EmbedDimension cases
1, 6, and 7. Each of those five cases passes against its golden when started
with pristine data. The pytest hook snapshots and restores the Lorenz sample
after every case, so all 31 external pyEDM tests pass without editing their
source or reference outputs.

Both external MDE cases are checked with the legacy-keyword adapter. The Lorenz
case is a regression oracle for the four-dimensional result restored above;
its source and golden remain unmodified.

Known source defects remain visible rather than silently rewriting an external
snapshot: `test_ccm3` builds NaN data but calls CCM with the clean frame,
`test_simplex7` mutates shared sample data, both MDE tests discard the
selected-variable column, and neither MDE test fixes its CCM seed. The hook
contains the shared-data side effect; project-owned tests cover behavior that
cannot be trusted to the remaining cases.
