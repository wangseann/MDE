"""Package-local extraction of the deployed Torch MDE sweep backend.

The numerical functions below are intentionally kept faithful to
``fmri-edm-ccm/planB_audiojepa/scripts/mde_sweep_backend_probe.py``.  Project-
specific loading, command-line, and report-writing code stays outside dimx.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree


__all__ = [
    "simplex_1d_predict",
    "simplex_predict",
    "eval_cols",
    "candidate_sweep_cpu_reference",
    "candidate_sweep_torch",
    "compare_rows",
]
__driving_backend_contract__ = 1
__prototype_sha256__ = \
    "eb1e0f82baf0474fc4794d94898e9948db174c8acd8ca9c7cbcabc8bc7df63c0"


def rho(pred: np.ndarray, true: np.ndarray) -> float:
    if pred.size == 0 or pred.std() <= 1e-12 or true.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(pred, true)[0, 1])


def simplex_predict(
    xlib: np.ndarray, ylib: np.ndarray, xq: np.ndarray, *,
    fixed_neighbors: int | None = None
) -> np.ndarray:
    e = xlib.shape[1]
    k = min(fixed_neighbors if fixed_neighbors is not None else e + 1,
            len(ylib))
    tree = cKDTree(xlib)
    dist, idx = tree.query(xq, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    d1 = np.maximum(dist[:, [0]], 1e-12)
    weights = np.exp(-dist / d1).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    return (weights * ylib[idx]).sum(axis=1)


def simplex_1d_predict(
    xl: np.ndarray, yl: np.ndarray, xq: np.ndarray
) -> np.ndarray:
    order = np.argsort(xl)
    xs = xl[order]
    ys = yl[order]
    pos = np.searchsorted(xs, xq)
    n = len(xs)
    pred = np.empty(len(xq), dtype=np.float32)
    for i, p in enumerate(pos):
        lo = max(0, p - 3)
        hi = min(n, p + 3)
        idx = np.arange(lo, hi)
        if len(idx) < 2:
            idx = np.array([0, min(1, n - 1)])
        d = np.abs(xs[idx] - xq[i])
        take = np.argpartition(d, min(1, len(d) - 1))[:2]
        nn = idx[take]
        dn = d[take]
        d1 = float(np.min(dn))
        weights = (np.ones_like(dn, dtype=np.float32) if d1 <= 1e-12
                   else np.exp(-dn / d1).astype(np.float32))
        weights /= weights.sum()
        pred[i] = float(np.sum(weights * ys[nn]))
    return pred


def eval_cols(
    xlib: np.ndarray,
    ylib: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
    cols: list[int],
    *,
    fixed_neighbors: int | None = None,
) -> float:
    if len(cols) == 1 and fixed_neighbors is None:
        pred = simplex_1d_predict(xlib[:, cols[0]], ylib, xq[:, cols[0]])
    else:
        pred = simplex_predict(
            xlib[:, cols], ylib, xq[:, cols],
            fixed_neighbors=fixed_neighbors)
    return rho(pred, yq)


def candidate_sweep_cpu_reference(
    xlib: np.ndarray,
    ylib: np.ndarray,
    xpred: np.ndarray,
    ypred: np.ndarray,
    *,
    selected: list[int],
    candidates: np.ndarray,
    fixed_neighbors: int | None = None,
) -> tuple[int | None, float, list[dict], float]:
    t0 = time.time()
    rows: list[dict] = []
    best_col: int | None = None
    best_rho = -np.inf
    for col in candidates:
        col_i = int(col)
        r = eval_cols(
            xlib, ylib, xpred, ypred, selected + [col_i],
            fixed_neighbors=fixed_neighbors)
        rows.append({"candidate": col_i, "rho": float(r)})
        if np.isfinite(r) and r > best_rho:
            best_col = col_i
            best_rho = float(r)
    return best_col, best_rho, rows, time.time() - t0


def candidate_sweep_torch(
    xlib: np.ndarray,
    ylib: np.ndarray,
    xpred: np.ndarray,
    ypred: np.ndarray,
    *,
    selected: list[int],
    candidates: np.ndarray,
    batch_candidates: int,
    pred_chunk: int,
    device_name: str,
    fixed_neighbors: int | None = None,
) -> tuple[int | None, float, list[dict], dict]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "torch backend requested, but torch is not importable") from exc

    device = torch.device(device_name)
    t_total = time.time()
    timings = {"transfer_seconds": 0.0, "sweep_seconds": 0.0}

    t = time.time()
    ylib_t = torch.as_tensor(ylib, dtype=torch.float32, device=device)
    ypred_np = np.asarray(ypred, dtype=np.float32)
    selected_lib = None
    selected_pred = None
    if selected:
        selected_lib = torch.as_tensor(
            xlib[:, selected], dtype=torch.float32, device=device)
        selected_pred = torch.as_tensor(
            xpred[:, selected], dtype=torch.float32, device=device)
    timings["transfer_seconds"] += time.time() - t

    rows: list[dict] = []
    best_col: int | None = None
    best_rho = -np.inf
    k = fixed_neighbors if fixed_neighbors is not None else len(selected) + 2
    k = min(k, len(ylib))

    t_sweep = time.time()
    for start in range(0, len(candidates), batch_candidates):
        cand = np.asarray(
            candidates[start:start + batch_candidates], dtype=np.int64)
        t = time.time()
        xlib_c = torch.as_tensor(
            xlib[:, cand], dtype=torch.float32,
            device=device).T.contiguous()
        xpred_c = torch.as_tensor(
            xpred[:, cand], dtype=torch.float32,
            device=device).T.contiguous()
        timings["transfer_seconds"] += time.time() - t

        pred_sums = torch.zeros(
            (len(cand),), dtype=torch.float32, device=device)
        weight_sums = torch.zeros(
            (len(cand),), dtype=torch.float32, device=device)

        for p0 in range(0, xpred.shape[0], pred_chunk):
            p1 = min(p0 + pred_chunk, xpred.shape[0])
            # Candidate-only squared distances: B x P x L.
            diff = xpred_c[:, p0:p1, None] - xlib_c[:, None, :]
            dist2 = diff * diff
            if selected:
                assert selected_pred is not None and selected_lib is not None
                sp = selected_pred[p0:p1]
                # P x L selected-coordinate distances, broadcast over
                # candidates.
                sel_diff = sp[:, None, :] - selected_lib[None, :, :]
                dist2 = dist2 + sel_diff.pow(2).sum(dim=2).unsqueeze(0)
            dist = torch.sqrt(torch.clamp(dist2, min=0.0))
            nn_dist, nn_idx = torch.topk(
                dist, k=k, dim=2, largest=False)
            d1 = torch.clamp(nn_dist[:, :, [0]], min=1e-12)
            weights = torch.exp(-nn_dist / d1)
            weights = weights / torch.clamp(
                weights.sum(dim=2, keepdim=True), min=1e-12)
            pred = (weights * ylib_t[nn_idx]).sum(dim=2)
            pred_sums += pred.sum(dim=1)
            weight_sums += pred.shape[1]
            if p0 == 0:
                preds_all = pred.detach().cpu().numpy()
            else:
                preds_all = np.concatenate(
                    [preds_all, pred.detach().cpu().numpy()], axis=1)

        for i, col_i in enumerate(cand):
            r = rho(preds_all[i].astype(np.float32), ypred_np)
            rows.append({"candidate": int(col_i), "rho": float(r)})
            if np.isfinite(r) and r > best_rho:
                best_col = int(col_i)
                best_rho = float(r)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["sweep_seconds"] = time.time() - t_sweep
    timings["total_seconds"] = time.time() - t_total
    return best_col, best_rho, rows, timings


def compare_rows(
    cpu_rows: list[dict], other_rows: list[dict],
    topk_values: tuple[int, ...] = (10, 50, 100)
) -> dict:
    cpu = {r["candidate"]: r["rho"] for r in cpu_rows}
    other = {r["candidate"]: r["rho"] for r in other_rows}
    common = sorted(set(cpu) & set(other))
    if not common:
        return {"n_common": 0}
    diffs = np.asarray(
        [abs(cpu[c] - other[c]) for c in common], dtype=np.float64)
    cpu_ranked = sorted(common, key=lambda c: cpu[c], reverse=True)
    other_ranked = sorted(common, key=lambda c: other[c], reverse=True)
    cpu_rank = {c: i + 1 for i, c in enumerate(cpu_ranked)}
    other_rank = {c: i + 1 for i, c in enumerate(other_ranked)}
    topk = {}
    for k in topk_values:
        kk = min(k, len(common))
        cpu_top = cpu_ranked[:kk]
        other_top = other_ranked[:kk]
        overlap = sorted(set(cpu_top) & set(other_top))
        displacements = [abs(cpu_rank[c] - other_rank[c]) for c in overlap]
        topk[str(k)] = {
            "k_effective": kk,
            "overlap_n": len(overlap),
            "overlap_fraction": float(len(overlap) / kk) if kk else float("nan"),
            "max_rank_displacement_shared": (
                int(max(displacements)) if displacements else None),
            "mean_rank_displacement_shared": (
                float(np.mean(displacements)) if displacements else None),
            "cpu_top": [int(c) for c in cpu_top],
            "other_top": [int(c) for c in other_top],
        }
    diff_rows = sorted(
        (
            {
                "candidate": int(c),
                "abs_rho_diff": float(abs(cpu[c] - other[c])),
                "cpu_rho": float(cpu[c]),
                "other_rho": float(other[c]),
                "cpu_rank": int(cpu_rank[c]),
                "other_rank": int(other_rank[c]),
            }
            for c in common
        ),
        key=lambda r: r["abs_rho_diff"],
        reverse=True,
    )
    return {
        "n_common": len(common),
        "max_abs_rho_diff": float(diffs.max()),
        "mean_abs_rho_diff": float(diffs.mean()),
        "p95_abs_rho_diff": float(np.percentile(diffs, 95)),
        "top1_match": bool(cpu_ranked[0] == other_ranked[0]) if common else None,
        "cpu_top1": int(cpu_ranked[0]) if common else None,
        "other_top1": int(other_ranked[0]) if common else None,
        "topk": topk,
        "largest_abs_diff_rows": diff_rows[:10],
    }
