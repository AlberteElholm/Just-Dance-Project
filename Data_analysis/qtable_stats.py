# qtable_stats_fixed_range.py
# Prints key statistics for Q-tables stored in .pkl files.
# - Forces state range 0..1250 (inclusive). Missing states are filled with zeros.
# - Handles None/invalid/wrong-length entries robustly by replacing with zeros.
#
# Usage:
#   python qtable_stats_fixed_range.py q_table.pkl
#   python qtable_stats_fixed_range.py q-tables/*.pkl
#   python qtable_stats_fixed_range.py q_table.pkl --actions 27 --action_stats
#   python qtable_stats_fixed_range.py q_table.pkl --per_state --topk 10

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import numpy as np

STATE_MIN = 0
STATE_MAX = 1250  # inclusive


def load_pkl(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def extract_q_raw(obj: Any) -> Tuple[Dict[Any, Any], Dict[str, Any]]:
    """
    Returns:
      Q_raw: mapping state -> value (value may be np array-like, list, or None)
      meta: other fields if present
    Accepts common formats:
      - {"Q": {state: np.array(...)}, "episode_count": ..., ...}
      - {state: np.array(...)}  (raw Q dict)
    """
    meta: Dict[str, Any] = {}
    if isinstance(obj, dict) and "Q" in obj:
        q_raw = obj["Q"]
        meta = {k: v for k, v in obj.items() if k != "Q"}
    else:
        q_raw = obj

    if not isinstance(q_raw, dict):
        raise TypeError(
            f"Could not interpret Q-table: expected dict or dict with 'Q'. Got: {type(q_raw)}"
        )
    return q_raw, meta


def infer_n_actions(Q_raw: Dict[Any, Any]) -> int:
    """
    Infer action count from the most common usable vector length.
    Skips None and scalars.
    """
    lengths = []
    for v in Q_raw.values():
        if v is None:
            continue
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.ndim == 0:
            continue
        try:
            lengths.append(int(arr.reshape(-1).shape[0]))
        except Exception:
            continue

    if not lengths:
        raise ValueError(
            "Could not infer n_actions: all entries were None/scalars/invalid. "
            "Pass --actions N explicitly."
        )

    vals, counts = np.unique(lengths, return_counts=True)
    return int(vals[np.argmax(counts)])


def sanitize_q_table(Q_raw: Dict[Any, Any], n_actions: int) -> Tuple[Dict[Any, np.ndarray], int]:
    """
    Convert values to 1D float32 arrays of length n_actions.
    Any None/invalid/wrong-length entries become zeros.
    Returns (Q_sanitized, replaced_count).
    """
    Q: Dict[Any, np.ndarray] = {}
    replaced = 0

    for k, v in Q_raw.items():
        if v is None:
            Q[k] = np.zeros(n_actions, dtype=np.float32)
            replaced += 1
            continue

        try:
            arr = np.asarray(v, dtype=np.float32)
        except Exception:
            Q[k] = np.zeros(n_actions, dtype=np.float32)
            replaced += 1
            continue

        if arr.ndim == 0:
            Q[k] = np.zeros(n_actions, dtype=np.float32)
            replaced += 1
            continue

        arr = arr.reshape(-1)

        if arr.shape[0] != n_actions:
            Q[k] = np.zeros(n_actions, dtype=np.float32)
            replaced += 1
            continue

        Q[k] = arr

    return Q, replaced


def enforce_state_range(Q: Dict[Any, np.ndarray], n_actions: int) -> Tuple[Dict[int, np.ndarray], int]:
    """
    Force integer states exactly STATE_MIN..STATE_MAX (inclusive).
    Missing states filled with zeros.
    Returns (full_Q, missing_count).
    """
    full_Q: Dict[int, np.ndarray] = {}
    missing = 0
    for s in range(STATE_MIN, STATE_MAX + 1):
        if s in Q:
            full_Q[s] = Q[s]
        else:
            full_Q[s] = np.zeros(n_actions, dtype=np.float32)
            missing += 1
    return full_Q, missing


def safe_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    xf = x[finite]

    return {
        "count": float(x.size),
        "finite": float(xf.size),
        "nan": float(np.isnan(x).sum()),
        "inf": float(np.isinf(x).sum()),
        "min": float(np.min(xf)) if xf.size else float("nan"),
        "max": float(np.max(xf)) if xf.size else float("nan"),
        "mean": float(np.mean(xf)) if xf.size else float("nan"),
        "std": float(np.std(xf)) if xf.size else float("nan"),
        "p05": float(np.quantile(xf, 0.05)) if xf.size else float("nan"),
        "p50": float(np.quantile(xf, 0.50)) if xf.size else float("nan"),
        "p95": float(np.quantile(xf, 0.95)) if xf.size else float("nan"),
        "zeros": float((x == 0).sum()),
        "abs_mean": float(np.mean(np.abs(xf))) if xf.size else float("nan"),
    }


def fmt(k: str, v: Any) -> str:
    if isinstance(v, float):
        return f"{k}={v:.6g}"
    return f"{k}={v}"


def summarize_file(
    path: Path,
    topk: int,
    per_state: bool,
    action_stats: bool,
    n_actions_arg: Optional[int],
) -> None:
    obj = load_pkl(path)
    Q_raw, meta = extract_q_raw(obj)

    n_actions = n_actions_arg or infer_n_actions(Q_raw)

    # Sanitize entries + force state range
    Q_sanitized, replaced = sanitize_q_table(Q_raw, n_actions)
    Q, missing = enforce_state_range(Q_sanitized, n_actions)

    n_states = len(Q)

    # Global values
    all_vals = np.concatenate([Q[s] for s in range(STATE_MIN, STATE_MAX + 1)], axis=0)

    print("=" * 80)
    print(f"FILE: {path}")
    print(f"States: {n_states} (forced range {STATE_MIN}–{STATE_MAX}) | Using n_actions={n_actions}")
    if meta:
        common_order = ["episode_count", "eps", "alpha", "gamma", "lam"]
        ordered = []
        for k in common_order:
            if k in meta:
                ordered.append((k, meta[k]))
        for k in sorted(meta.keys()):
            if k not in common_order:
                ordered.append((k, meta[k]))
        print("Meta:", ", ".join(fmt(k, v) for k, v in ordered))
    else:
        print("Meta: (none found)")

    if replaced:
        print(f"WARNING: Replaced {replaced} invalid/None/wrong-length entries with zeros.")
    print(f"Filled missing states with zeros: {missing}")

    g = safe_stats(all_vals)
    print(
        "Global Q-value stats: "
        + ", ".join(
            [
                fmt("count", g["count"]),
                fmt("finite", g["finite"]),
                fmt("nan", g["nan"]),
                fmt("inf", g["inf"]),
                fmt("min", g["min"]),
                fmt("max", g["max"]),
                fmt("mean", g["mean"]),
                fmt("std", g["std"]),
                fmt("p05", g["p05"]),
                fmt("p50", g["p50"]),
                fmt("p95", g["p95"]),
                fmt("zeros", g["zeros"]),
                fmt("abs_mean", g["abs_mean"]),
            ]
        )
    )

    # Per-state summaries (on forced range)
    state_max = {s: float(np.max(Q[s])) for s in range(STATE_MIN, STATE_MAX + 1)}
    state_mean = {s: float(np.mean(Q[s])) for s in range(STATE_MIN, STATE_MAX + 1)}
    state_span = {s: float(np.max(Q[s]) - np.min(Q[s])) for s in range(STATE_MIN, STATE_MAX + 1)}
    state_nonzero = {s: int(np.any(Q[s] != 0)) for s in range(STATE_MIN, STATE_MAX + 1)}

    top_states = sorted(state_max.items(), key=lambda kv: kv[1], reverse=True)[:topk]
    bottom_states = sorted(state_max.items(), key=lambda kv: kv[1])[:topk]

    print(f"Top {topk} states by max(Q[state]):")
    for s, v in top_states:
        print(
            f"  state={s}  max={v:.6g}  mean={state_mean[s]:.6g}  span={state_span[s]:.6g}  visited={state_nonzero[s]}"
        )

    print(f"Bottom {topk} states by max(Q[state]):")
    for s, v in bottom_states:
        print(
            f"  state={s}  max={v:.6g}  mean={state_mean[s]:.6g}  span={state_span[s]:.6g}  visited={state_nonzero[s]}"
        )

    visited_count = sum(state_nonzero.values())
    print(f"State coverage: {visited_count}/{n_states} = {visited_count / n_states * 100:.2f}% visited (any non-zero Q)")

    if per_state:
        print("Per-state stats (min/mean/max/std/zeros):")
        for s in range(STATE_MIN, STATE_MAX + 1):
            st = safe_stats(Q[s])
            print(
                f"  state={s}  "
                f"min={st['min']:.6g} mean={st['mean']:.6g} max={st['max']:.6g} std={st['std']:.6g} zeros={int(st['zeros'])}"
            )

    if action_stats:
        A = np.stack([Q[s] for s in range(STATE_MIN, STATE_MAX + 1)], axis=0)  # (states, actions)
        action_mean = np.mean(A, axis=0)
        action_std = np.std(A, axis=0)
        action_min = np.min(A, axis=0)
        action_max = np.max(A, axis=0)
        best_actions = np.argmax(A, axis=1)
        best_counts = np.bincount(best_actions, minlength=n_actions)
        best_pct = best_counts / best_actions.size

        print("Action stats across states:")
        print("  idx | best%   mean      std       min       max")
        for i in range(n_actions):
            print(
                f"  {i:>3} | {best_pct[i]*100:>5.1f}% "
                f"{action_mean[i]:>9.6g} {action_std[i]:>9.6g} {action_min[i]:>9.6g} {action_max[i]:>9.6g}"
            )

        top_actions = np.argsort(action_mean)[::-1][:min(topk, n_actions)]
        print(f"Top {min(topk, n_actions)} actions by mean Q:")
        for i in top_actions:
            print(
                f"  action={int(i)}  mean={action_mean[i]:.6g}  best%={best_pct[i]*100:.2f}%  max={action_max[i]:.6g}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Print key statistics about Q-tables stored in .pkl files.")
    ap.add_argument("paths", nargs="+", help="One or more .pkl files (wildcards allowed by shell).")
    ap.add_argument("--topk", type=int, default=8, help="How many top/bottom states/actions to show.")
    ap.add_argument("--per_state", action="store_true", help="Print per-state stats (can be long).")
    ap.add_argument("--action_stats", action="store_true", help="Print per-action stats across states.")
    ap.add_argument("--actions", type=int, default=None, help="Force n_actions (otherwise inferred).")
    args = ap.parse_args()

    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print("=" * 80)
            print(f"FILE: {p}  (NOT FOUND)")
            continue
        try:
            summarize_file(path, args.topk, args.per_state, args.action_stats, args.actions)
        except Exception as e:
            print("=" * 80)
            print(f"FILE: {path}")
            print(f"ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
