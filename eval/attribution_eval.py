"""
eval/attribution_eval.py

CASPIAN attribution evaluation — computes paper §3.4 metrics from
task_metrics.csv + gt_attribution.json.

Ground truth must first be generated via:
  python -m eval.generate_attribution_gt --all

Metrics (paper §3.4):
  Origin accuracy      predicted i_origin == GT i_origin
  Amplifier accuracy   predicted i_amp    == GT i_amp
  Bridge accuracy      predicted i_bridge == GT i_bridge
  Spine Jaccard@3      Jaccard over top-3 spine edge-sets
  Spine Jaccard@5      Jaccard over top-5 spine edge-sets
  Channel accuracy     dominant spine channel match

Usage:
  # Single directory (GT auto-discovered)
  python -m eval.attribution_eval \\
    --dir outputs/ACIArena/LLMDebate_standard

  # All outputs
  python -m eval.attribution_eval --all

  # Explicit GT path
  python -m eval.attribution_eval \\
    --dir  outputs/ACIArena/LLMDebate_standard \\
    --gt   outputs/ACIArena/LLMDebate_standard/gt_attribution.json

  # JSON output
  python -m eval.attribution_eval --all --json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import (
    compute_attribution_metrics,
    print_metrics,
    to_dict,
    AttributionMetrics,
    DetectionMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_combo(task_csv: Path) -> tuple[str, str, str]:
    parts = task_csv.parent.parts
    try:
        benchmark = parts[-2]
        fw_cfg    = parts[-1]
        framework, _, config = fw_cfg.partition("_")
        return benchmark, framework, config or "standard"
    except Exception:
        return "unknown", "unknown", "unknown"


def _find_gt(d: Path) -> Path | None:
    """Auto-discover GT file in directory."""
    candidates = [
        d / "gt_attribution.json",
        d / "gt_attribution.jsonl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _eval_dir(d: Path, gt_path: Path | None = None) -> dict[str, Any] | None:
    task_csv = d / "task_metrics.csv"
    if not task_csv.exists():
        return None

    gt = gt_path or _find_gt(d)
    if gt is None:
        return {
            "benchmark": _parse_combo(task_csv)[0],
            "framework": _parse_combo(task_csv)[1],
            "config":    _parse_combo(task_csv)[2],
            "dir":       str(d),
            "error":     "No gt_attribution.json found. Run: python -m eval.generate_attribution_gt --all",
        }

    benchmark, framework, config = _parse_combo(task_csv)
    atr  = compute_attribution_metrics(task_csv, gt)
    out  = to_dict(DetectionMetrics(), atr).get("attribution", {})
    out["benchmark"]  = benchmark
    out["framework"]  = framework
    out["config"]     = config
    out["dir"]        = str(d)
    out["gt_path"]    = str(gt)
    out["n_detected"] = atr.n_detected
    out["n_with_gt"]  = atr.n_with_gt
    out["notes"]      = atr.notes
    return out


def _find_all_dirs(root: Path) -> list[Path]:
    return sorted(
        p.parent
        for p in (root / "outputs").rglob("task_metrics.csv")
        if "smoke_test" not in str(p)
    )


def _print_table(results: list[dict[str, Any]]) -> None:
    W = 80
    print(f"\n{'='*W}")
    print("  CASPIAN ATTRIBUTION EVALUATION")
    print(f"{'='*W}")
    hdr = (f"  {'Combo':<30} {'Det':>4} {'GT':>4} "
           f"{'Orig':>6} {'Amp':>6} {'Brdg':>6} "
           f"{'Sp@3':>6} {'Sp@5':>6} {'Ch':>6}")
    print(hdr)
    print(f"  {'-'*30} {'-'*4} {'-'*4} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for r in results:
        if "error" in r:
            combo = f"{r.get('benchmark','?')}/{r.get('framework','?')}"
            print(f"  {combo:<30}  ERROR: {r['error'][:40]}")
            continue

        combo = f"{r.get('benchmark','?')}/{r.get('framework','?')}"
        det   = r.get("n_detected", 0)
        gt_n  = r.get("n_with_gt",  0)

        def _f(k: str) -> str:
            v = r.get(k)
            return f"{v:.3f}" if v is not None else "  n/a"

        print(f"  {combo:<30} {det:>4} {gt_n:>4} "
              f"{_f('origin_accuracy'):>6} "
              f"{_f('amplifier_accuracy'):>6} "
              f"{_f('bridge_accuracy'):>6} "
              f"{_f('spine_jaccard_at_3'):>6} "
              f"{_f('spine_jaccard_at_5'):>6} "
              f"{_f('channel_accuracy'):>6}")

    print(f"{'='*W}")

    for r in results:
        notes = r.get("notes", [])
        if notes:
            combo = f"{r.get('benchmark','?')}/{r.get('framework','?')}"
            for n in notes:
                print(f"  NOTE [{combo}]: {n}")
    print()


def _print_verbose(results: list[dict[str, Any]]) -> None:
    for r in results:
        if "error" in r:
            continue
        combo = f"{r.get('benchmark','?')}/{r.get('framework','?')}/{r.get('config','?')}"
        print(f"\n{'─'*60}")
        print(f"  {combo}")
        print(f"  Detected: {r.get('n_detected',0)}  With GT: {r.get('n_with_gt',0)}")
        print(f"{'─'*60}")

        def _row(label: str, key: str) -> None:
            v = r.get(key)
            val = f"{v:.4f}" if v is not None else "n/a"
            print(f"  {label:<25} {val}")

        _row("Origin accuracy",      "origin_accuracy")
        _row("Amplifier accuracy",   "amplifier_accuracy")
        _row("Bridge accuracy",      "bridge_accuracy")
        _row("Spine Jaccard@3",      "spine_jaccard_at_3")
        _row("Spine Jaccard@5",      "spine_jaccard_at_5")
        _row("Channel accuracy",     "channel_accuracy")

        for n in r.get("notes", []):
            print(f"  NOTE: {n}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CASPIAN attribution evaluation — paper §3.4 metrics"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dir",  type=Path, help="Single output directory")
    grp.add_argument("--all",  action="store_true", help="All outputs/ subdirs")

    parser.add_argument("--gt",        type=Path, default=None,
                        help="Explicit GT JSON path (for --dir mode)")
    parser.add_argument("--benchmark", type=str,  default=None)
    parser.add_argument("--framework", type=str,  default=None)
    parser.add_argument("--json",      action="store_true")
    parser.add_argument("--verbose",   action="store_true")
    parser.add_argument("--root",      type=Path, default=ROOT)
    args = parser.parse_args()

    if args.dir:
        dirs = [args.dir]
    else:
        dirs = _find_all_dirs(args.root)

    if args.benchmark:
        b = args.benchmark.lower()
        dirs = [d for d in dirs if b in str(d).lower()]
    if args.framework:
        f = args.framework.lower()
        dirs = [d for d in dirs if f in str(d).lower()]

    if not dirs:
        print("No matching output directories found.")
        return

    results = []
    for d in dirs:
        gt_path = args.gt if args.dir else None
        r = _eval_dir(d, gt_path)
        if r:
            results.append(r)
        else:
            print(f"  [SKIP] {d}")

    if not results:
        print("No evaluable results found.")
        return

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    _print_table(results)

    if args.verbose:
        _print_verbose(results)


if __name__ == "__main__":
    main()