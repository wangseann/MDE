#!/usr/bin/env python3
"""Record repeatable diagnostics for the deployed Torch sweep contract.

The benchmark has correctness gates but deliberately has no speed threshold.
Shared GitHub runners are suitable for collecting comparable metadata, not for
making a stable performance claim. The same command can run on a labelled CUDA
runner by changing ``--device``.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
BACKEND_PATH = REPOSITORY / "dimx" / "TorchBackend.py"
SPECIFICATION = importlib.util.spec_from_file_location(
    "_dimx_torch_backend_benchmark", BACKEND_PATH)
backend = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(backend)


def PositiveInteger(value: str) -> int:
    valueInt = int(value)
    if valueInt < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return valueInt


def NonNegativeInteger(value: str) -> int:
    valueInt = int(value)
    if valueInt < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return valueInt


def SyntheticData(libRows, predRows, selectedCount, candidateCount, seed):
    rng = np.random.RandomState(seed)
    columnCount = selectedCount + candidateCount
    xlib = rng.normal(size=(libRows, columnCount)).astype(np.float32)
    xpred = rng.normal(size=(predRows, columnCount)).astype(np.float32)
    signalColumns = np.arange(min(3, columnCount))
    weights = rng.normal(size=len(signalColumns)).astype(np.float32)
    ylib = (xlib[:, signalColumns] @ weights +
            0.1 * rng.normal(size=libRows).astype(np.float32))
    ypred = (xpred[:, signalColumns] @ weights +
             0.1 * rng.normal(size=predRows).astype(np.float32))
    selected = list(range(selectedCount))
    candidates = np.arange(
        selectedCount, columnCount, dtype=np.int64)
    return xlib, ylib.astype(np.float32), \
        xpred, ypred.astype(np.float32), selected, candidates


def TimingSummary(values):
    values = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "values_seconds": values.tolist(),
        "median_seconds": float(np.median(values)),
        "q1_seconds": float(q1),
        "q3_seconds": float(q3),
        "iqr_seconds": float(q3 - q1),
    }


def ValidateComparison(cpuBest, torchBest, cpuRows, torchRows, tolerance):
    comparison = backend.compare_rows(cpuRows, torchRows)
    if not comparison["top1_match"]:
        raise RuntimeError(
            f"CPU and Torch selected different candidates: "
            f"{cpuBest} != {torchBest}")
    if comparison["max_abs_rho_diff"] > tolerance:
        raise RuntimeError(
            "CPU/Torch rho difference exceeded tolerance: "
            f"{comparison['max_abs_rho_diff']} > {tolerance}")
    return comparison


def ParseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lib-rows", type=PositiveInteger, default=240)
    parser.add_argument("--pred-rows", type=PositiveInteger, default=120)
    parser.add_argument("--selected", type=int, default=1)
    parser.add_argument("--candidates", type=PositiveInteger, default=64)
    parser.add_argument("--batch-candidates", type=PositiveInteger, default=16)
    parser.add_argument("--pred-chunk", type=PositiveInteger, default=64)
    parser.add_argument("--repeats", type=PositiveInteger, default=3)
    parser.add_argument("--warmups", type=NonNegativeInteger, default=2)
    parser.add_argument("--torch-threads", type=PositiveInteger, default=1)
    parser.add_argument("--seed", type=int, default=371)
    parser.add_argument("--rho-tolerance", type=float, default=2e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.selected < 0:
        parser.error("--selected must be non-negative")
    if args.rho_tolerance < 0:
        parser.error("--rho-tolerance must be non-negative")
    return args


def main():
    args = ParseArguments()

    import torch

    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested benchmark device {args.device!r}, but CUDA is unavailable")

    xlib, ylib, xpred, ypred, selected, candidates = SyntheticData(
        args.lib_rows, args.pred_rows, args.selected, args.candidates,
        args.seed)
    sweepArgs = {
        "selected": selected,
        "candidates": candidates,
        "batch_candidates": args.batch_candidates,
        "pred_chunk": args.pred_chunk,
        "device_name": args.device,
    }

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Record the first full sweep separately, including allocator/device
    # startup. Warm steady state only after preserving that cold observation.
    cpuColdWallStart = perf_counter()
    cpuColdBest, _, cpuColdRows, cpuColdSeconds = \
        backend.candidate_sweep_cpu_reference(
            xlib, ylib, xpred, ypred,
            selected=selected, candidates=candidates)
    cpuColdWall = perf_counter() - cpuColdWallStart
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    torchColdWallStart = perf_counter()
    torchColdBest, _, torchColdRows, torchColdTimings = \
        backend.candidate_sweep_torch(
            xlib, ylib, xpred, ypred, **sweepArgs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    torchColdWall = perf_counter() - torchColdWallStart
    coldComparison = ValidateComparison(
        cpuColdBest, torchColdBest, cpuColdRows, torchColdRows,
        args.rho_tolerance)

    for _ in range(args.warmups):
        warmCPU = backend.candidate_sweep_cpu_reference(
            xlib, ylib, xpred, ypred,
            selected=selected, candidates=candidates)
        warmTorch = backend.candidate_sweep_torch(
            xlib, ylib, xpred, ypred, **sweepArgs)
        ValidateComparison(
            warmCPU[0], warmTorch[0], warmCPU[2], warmTorch[2],
            args.rho_tolerance)

    cpuSeconds = []
    torchSeconds = []
    cpuWallSeconds = []
    torchWallSeconds = []
    torchDetails = []
    comparisons = []
    selections = []

    for _ in range(args.repeats):
        cpuWallStart = perf_counter()
        cpuBest, _, cpuRows, seconds = \
            backend.candidate_sweep_cpu_reference(
                xlib, ylib, xpred, ypred,
                selected=selected, candidates=candidates)
        cpuWall = perf_counter() - cpuWallStart
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        torchWallStart = perf_counter()
        torchBest, _, torchRows, timings = backend.candidate_sweep_torch(
            xlib, ylib, xpred, ypred, **sweepArgs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        torchWall = perf_counter() - torchWallStart
        comparison = ValidateComparison(
            cpuBest, torchBest, cpuRows, torchRows, args.rho_tolerance)

        cpuSeconds.append(float(seconds))
        torchSeconds.append(float(timings["total_seconds"]))
        cpuWallSeconds.append(float(cpuWall))
        torchWallSeconds.append(float(torchWall))
        torchDetails.append({key: float(value)
                             for key, value in timings.items()})
        comparisons.append(comparison)
        selections.append({"cpu": cpuBest, "torch": torchBest})

    cpuSummary = TimingSummary(cpuSeconds)
    torchSummary = TimingSummary(torchSeconds)
    cpuWallSummary = TimingSummary(cpuWallSeconds)
    torchWallSummary = TimingSummary(torchWallSeconds)
    speedup = (cpuWallSummary["median_seconds"] /
               torchWallSummary["median_seconds"])

    deviceMetadata = {"requested": args.device, "type": device.type}
    if device.type == "cuda":
        deviceMetadata.update({
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(
                torch.cuda.get_device_properties(device).total_memory),
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)),
            "peak_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)),
        })

    modulePath = Path(backend.__file__).resolve()
    report = {
        "schema_version": 1,
        "purpose": "diagnostic-only; no performance pass/fail threshold",
        "backend": {
            "module": str(modulePath),
            "sha256": sha256(modulePath.read_bytes()).hexdigest(),
            "prototype_sha256": backend.__prototype_sha256__,
            "contract": backend.__driving_backend_contract__,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_threads": int(torch.get_num_threads()),
            "device": deviceMetadata,
            "github": {
                "repository": os.environ.get("GITHUB_REPOSITORY"),
                "sha": os.environ.get("GITHUB_SHA"),
                "ref": os.environ.get("GITHUB_REF"),
                "run_id": os.environ.get("GITHUB_RUN_ID"),
                "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            },
        },
        "shape": {
            "lib_rows": args.lib_rows,
            "pred_rows": args.pred_rows,
            "selected_columns": args.selected,
            "candidate_columns": args.candidates,
        },
        "controls": {
            "seed": args.seed,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "torch_threads_requested": args.torch_threads,
            "batch_candidates": args.batch_candidates,
            "pred_chunk": args.pred_chunk,
            "rho_tolerance": args.rho_tolerance,
        },
        "correctness": {
            "cold": coldComparison,
            "measured_each_repeat": comparisons,
        },
        "selection_each_repeat": selections,
        "timing": {
            "cold": {
                "cpu_reported_seconds": float(cpuColdSeconds),
                "cpu_wall_seconds": float(cpuColdWall),
                "torch_reported": {
                    key: float(value)
                    for key, value in torchColdTimings.items()
                },
                "torch_wall_seconds": float(torchColdWall),
            },
            "cpu_reference_reported": cpuSummary,
            "torch_reported": torchSummary,
            "cpu_reference_wall": cpuWallSummary,
            "torch_wall": torchWallSummary,
            "torch_reported_each_repeat": torchDetails,
            "median_wall_speedup_cpu_over_torch": float(speedup),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "device": args.device,
        "max_abs_rho_diff": max(
            comparison["max_abs_rho_diff"] for comparison in comparisons),
        "top1_match": all(
            comparison["top1_match"] for comparison in comparisons),
        "cpu_median_wall_seconds": cpuWallSummary["median_seconds"],
        "torch_median_wall_seconds": torchWallSummary["median_seconds"],
        "median_wall_speedup_cpu_over_torch": speedup,
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
