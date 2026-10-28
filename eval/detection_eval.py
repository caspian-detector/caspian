"""
eval/detection_eval.py

CASPIAN detection evaluation — computes all paper §3.3 metrics from
task_metrics.csv outputs.

Supports:
  - single run directory
  - all runs under outputs/ (aggregated + per-combo)
  - benchmark or framework filtering
  - JSON or table output
  - comparison across two run directories

Usage:
  # Single directory
  python -m eval.detection_eval \\
    --dir outputs/ACIArena/LLMDebate_standard

  # All outputs, grouped by benchmark
  python -m eval.detection_eval --all

  # Filter
  python -m eval.detection_eval --all --benchmark ACIArena
  python -m eval.detection_eval --all --framework LLMDebate

  # JSON output
  python -m eval.detection_eval --all --json > results.json

  # Compare two runs
  python -m eval.detection_eval \\
    --compare outputs/ACIArena/LLMDebate_standard \\
               outputs/TAMAS/LLMDebate_standard
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
    compute_detection_metrics_from_rows,
    print_metrics,
    to_dict,
    DetectionMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_combo(task_csv: Path) -> tuple[str, str, str]:
    """Extract benchmark/framework/config from directory name."""
    parts = task_csv.parent.parts
    # Structure: .../outputs/{BENCHMARK}/{FRAMEWORK}_{CONFIG}/task_metrics.csv
    try:
        benchmark = parts[-2]
        fw_cfg    = parts[-1]
        framework, _, config = fw_cfg.partition("_")
        return benchmark, framework, config or "standard"
    except Exception:
        return "unknown", "unknown", "unknown"


def _eval_dir(d: Path) -> dict[str, Any] | None:
    task_csv = d / "task_metrics.csv"
    if not task_csv.exists():
        return None

    rows = _read_csv(task_csv)
    if not rows:
        return None

    benchmark, framework, config = _parse_combo(task_csv)
    det  = compute_detection_metrics_from_rows(rows)
    out  = to_dict(det)
    out["benchmark"] = benchmark
    out["framework"] = framework
    out["config"]    = config
    out["dir"]       = str(d)
    return out


def _find_all_dirs(root: Path) -> list[Path]:
    """Find all directories containing task_metrics.csv under outputs/."""
    return sorted(
        p.parent
        for p in (root / "outputs").rglob("task_metrics.csv")
        if "smoke_test" not in str(p)
    )


def _print_table(results: list[dict[str, Any]]) -> None:
    W = 72
    print(f"\n{'='*W}")
    print("  CASPIAN DETECTION EVALUATION")
    print(f"{'='*W}")
    hdr = f"  {'Combo':<34} {'N':>5} {'F1':>6} {'AUROC':>6} {'P':>6} {'R':>6} {'MRR':>6}"
    print(hdr)
    print(f"  {'-'*34} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for r in results:
        combo = f"{r.get('benchmark','?')}/{r.get('framework','?')}"
        n     = r.get("n_valid", 0)

        def _f(k: str) -> str:
            v = r.get(k)
            return f"{v:.3f}" if v is not None else "  n/a"

        print(f"  {combo:<34} {n:>5} "
              f"{_f('f1'):>6} {_f('auroc'):>6} "
              f"{_f('precision'):>6} {_f('recall'):>6} "
              f"{_f('mrr'):>6}")

    print(f"{'='*W}")

    # Notes
    for r in results:
        notes = r.get("notes", [])
        if notes:
            combo = f"{r.get('benchmark','?')}/{r.get('framework','?')}"
            for n in notes:
                print(f"  NOTE [{combo}]: {n}")
    print()


def _print_verbose(results: list[dict[str, Any]]) -> None:
    from eval.metrics import DetectionMetrics, print_metrics, AttributionMetrics
    import dataclasses

    for r in results:
        combo = f"{r.get('benchmark','?')}/{r.get('framework','?')}/{r.get('config','?')}"
        print(f"\n{'─'*60}")
        print(f"  {combo}")
        print(f"{'─'*60}")

        # Reconstruct DetectionMetrics from dict
        fields = {f.name for f in dataclasses.fields(DetectionMetrics)}
        kwargs = {k: v for k, v in r.items() if k in fields}
        kwargs.setdefault("notes", r.get("notes", []))
        det = DetectionMetrics(**kwargs)
        print_metrics(det)


def _compare(dirs: list[Path]) -> None:
    results = []
    for d in dirs:
        r = _eval_dir(d)
        if r:
            results.append(r)
        else:
            print(f"  [SKIP] no task_metrics.csv in {d}")

    if not results:
        print("No results to compare.")
        return

    _print_table(results)
    _print_verbose(results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CASPIAN detection evaluation — paper §3.3 metrics"
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dir",     type=Path,   help="Single output directory")
    grp.add_argument("--all",     action="store_true", help="All outputs/ subdirs")
    grp.add_argument("--compare", type=Path,   nargs="+",
                     help="Compare two or more output directories")

    parser.add_argument("--benchmark", type=str, default=None,
                        help="Filter by benchmark name")
    parser.add_argument("--framework", type=str, default=None,
                        help="Filter by framework name")
    parser.add_argument("--json",      action="store_true",
                        help="Output JSON instead of table")
    parser.add_argument("--verbose",   action="store_true",
                        help="Print full per-combo metric blocks")
    parser.add_argument("--root",      type=Path, default=ROOT)
    args = parser.parse_args()

    if args.compare:
        _compare(args.compare)
        return

    if args.dir:
        dirs = [args.dir]
    else:
        dirs = _find_all_dirs(args.root)

    # Filter
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
        r = _eval_dir(d)
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

    # Aggregate across all
    if len(results) > 1:
        all_rows: list[dict] = []
        for d in dirs:
            all_rows.extend(_read_csv(d / "task_metrics.csv"))

        if all_rows:
            print("\n  AGGREGATE (all combinations)")
            print(f"  {'─'*50}")
            agg = compute_detection_metrics_from_rows(all_rows)
            from eval.metrics import print_metrics
            print_metrics(agg)


if __name__ == "__main__":
    main()