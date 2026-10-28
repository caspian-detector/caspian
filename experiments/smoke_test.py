"""
experiments/smoke_test.py

CASPIAN smoke test — 3 attack + 3 benign per benchmark × framework.

Runs each combination sequentially: attack scenarios first (clears prior
output), then benign scenarios (appends), then computes metrics on the
merged task_metrics.csv.

Usage:
  python -m experiments.smoke_test
  python -m experiments.smoke_test --n 3 --timeout 300 --quick
  python -m experiments.smoke_test --frameworks AutoGen,CrewAI
  python -m experiments.smoke_test --benchmarks TAMAS --attack_only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Test matrix: (benchmark, framework, config, extra_attack_args, extra_benign_args)
# ---------------------------------------------------------------------------

# ACIArena attack args — diverse sampling across disclosure/disruption/hijacking
# --aci_attack all triggers _load_diverse_scenarios (no single attack type)
_ACI_ATK = ["--aci_attack", "all", "--max_turn", "2"]
_ACI_BEN = ["--aci_attack", "NoneAttack", "--max_turn", "2"]

# MetaGPT: use engineer as malicious agent (injects into code output)
_ACI_ATK_MGT = ["--aci_attack", "all", "--max_turn", "2",
                 "--malicious_agents", "engineer"]

MATRIX = [
    # ── TAMAS × all 4 MAS ──────────────────────────────────────────
    ("TAMAS",    "AutoGen",   "RoundRobin",    [], []),
    ("TAMAS",    "CrewAI",    "Decentralized", [], []),
    ("TAMAS",    "MetaGPT",   "standard",      [], []),
    ("TAMAS",    "LLMDebate", "standard",      [], []),
    # ── ACIArena × all 4 MAS ───────────────────────────────────────
    # AutoGen: native 2-agent user_proxy↔assistant (dyadic branch)
    ("ACIArena", "AutoGen",   "standard",      _ACI_ATK, _ACI_BEN),
    # CrewAI: fallback sequential workflow with per-role payload injection
    ("ACIArena", "CrewAI",    "standard",      _ACI_ATK, _ACI_BEN),
    # MetaGPT: native 5-agent chain (workflow branch)
    ("ACIArena", "MetaGPT",   "standard",      _ACI_ATK_MGT, _ACI_BEN),
    # LLMDebate: native 3-debater+aggregator (spectral branch)
    ("ACIArena", "LLMDebate", "standard",      _ACI_ATK, _ACI_BEN),
]


@dataclass
class ComboResult:
    benchmark: str
    framework: str
    config:    str
    skipped:   bool  = False
    error:     str   = ""
    n_attack:  int   = 0
    n_benign:  int   = 0
    n_fail:    int   = 0
    elapsed:   float = 0.0
    metrics:   dict  = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out_dir(benchmark: str, framework: str, config: str) -> Path:
    """Mirrors run_matrix.py: outputs/{BENCHMARK}/{FRAMEWORK}_{CONFIG}/"""
    return ROOT / "outputs" / benchmark / f"{framework}_{config}"


def _run_subprocess(
    benchmark:      str,
    framework:      str,
    config:         str,
    n:              int,
    attack:         bool,
    timeout:        int,
    extra_args:     list[str],
    metagpt_python: str | None,
) -> tuple[str, int]:
    """Run run_matrix for one class. Returns (stdout, returncode)."""
    cmd = [
        sys.executable, "-m", "experiments.run_matrix",
        "--benchmark", benchmark,
        "--framework", framework,
        "--config",    config,
        "--model",     "gpt-4o-mini",
        "--timeout",   str(timeout),
        "--limit",     str(n),
        "--root",      str(ROOT),
        "--attack_only" if attack else "--benign_only",
        *extra_args,
    ]
    env = os.environ.copy()
    if metagpt_python:
        env["METAGPT_PYTHON"] = metagpt_python
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout * n + 90, env=env, cwd=str(ROOT),
        )
        return proc.stdout + "\n" + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", 1


def _read_task_csv(out_dir: Path) -> list[dict]:
    p = out_dir / "task_metrics.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _count_failures(rows: list[dict]) -> int:
    return sum(int(r.get("execution_failed", 0) or 0) == 1 for r in rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CASPIAN smoke test — 3 attack + 3 benign per combination"
    )
    parser.add_argument("--n",              type=int,  default=3)
    parser.add_argument("--timeout",        type=int,  default=300)
    parser.add_argument("--frameworks",     type=str,  default=None)
    parser.add_argument("--benchmarks",     type=str,  default=None)
    parser.add_argument("--attack_only",    action="store_true")
    parser.add_argument("--benign_only",    action="store_true")
    parser.add_argument("--quick",          action="store_true",
                        help="Skip MetaGPT")
    parser.add_argument("--metagpt_python", type=str,  default=None)
    parser.add_argument("--no_clear",       action="store_true",
                        help="Don't clear prior outputs (resume mode)")
    args = parser.parse_args()

    metagpt_python = args.metagpt_python or os.environ.get("METAGPT_PYTHON")

    # Filter matrix
    matrix = list(MATRIX)
    if args.quick:
        matrix = [m for m in matrix if m[1] != "MetaGPT"]
    if args.frameworks:
        fws = {f.strip().lower() for f in args.frameworks.split(",")}
        matrix = [m for m in matrix if m[1].lower() in fws]
    if args.benchmarks:
        bms = {b.strip().lower() for b in args.benchmarks.split(",")}
        matrix = [m for m in matrix if m[0].lower() in bms]

    smoke_dir = ROOT / "outputs" / "smoke_test"
    smoke_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*66}")
    print(f"  CASPIAN SMOKE TEST")
    print(f"  {len(matrix)} combinations  ×  {args.n} attack + {args.n} benign")
    print(f"  timeout={args.timeout}s  metagpt={metagpt_python or 'not set'}")
    print(f"  smoke output → {smoke_dir}")
    print(f"{'='*66}\n")

    results: list[ComboResult] = []

    for benchmark, framework, config, atk_extra, ben_extra in matrix:
        label = f"{benchmark}/{framework}/{config}"
        print(f"┌─ {label}")
        r = ComboResult(benchmark=benchmark, framework=framework, config=config)
        t0 = time.time()

        out_dir = _out_dir(benchmark, framework, config)

        # Clear prior output so attack+benign don't resume across runs
        if not args.no_clear and out_dir.exists():
            shutil.rmtree(out_dir)

        # ── Attack scenarios ──────────────────────────────────────────
        if not args.benign_only:
            print(f"│  [attack ×{args.n}] ", end="", flush=True)
            stdout, rc = _run_subprocess(
                benchmark, framework, config,
                n=args.n, attack=True, timeout=args.timeout,
                extra_args=atk_extra, metagpt_python=metagpt_python,
            )
            atk_rows = _read_task_csv(out_dir)
            r.n_attack = len([x for x in atk_rows
                              if int(x.get("label", 0) or 0) == 1])
            atk_fail   = _count_failures(atk_rows)
            print(f"ran={len(atk_rows)} fail={atk_fail}")
            if rc != 0 and "[TIMEOUT]" in stdout:
                print(f"│  WARNING: timeout")

        # ── Benign scenarios ──────────────────────────────────────────
        if not args.attack_only:
            print(f"│  [benign ×{args.n}] ", end="", flush=True)
            stdout, rc = _run_subprocess(
                benchmark, framework, config,
                n=args.n, attack=False, timeout=args.timeout,
                extra_args=ben_extra, metagpt_python=metagpt_python,
            )
            all_rows   = _read_task_csv(out_dir)
            ben_rows   = [x for x in all_rows
                          if int(x.get("label", 0) or 0) == 0]
            r.n_benign = len(ben_rows)
            ben_fail   = _count_failures(ben_rows)
            print(f"ran={len(ben_rows)} fail={ben_fail}")

        # ── Metrics — read final combined CSV (both attack + benign) ──
        all_rows  = _read_task_csv(out_dir)
        r.n_attack = sum(int(x.get("label", 0) or 0) == 1 for x in all_rows)
        r.n_benign = sum(int(x.get("label", 0) or 0) == 0 for x in all_rows)
        r.n_fail  = _count_failures(all_rows)
        r.elapsed = time.time() - t0

        try:
            from eval.metrics import compute_detection_metrics_from_rows, to_dict
            det      = compute_detection_metrics_from_rows(all_rows)
            r.metrics = to_dict(det)
        except Exception as e:
            r.metrics = {"error": str(e)}

        m = r.metrics
        f1    = f"{m['f1']:.3f}"    if m.get("f1")    is not None else " n/a"
        auroc = f"{m['auroc']:.3f}" if m.get("auroc") is not None else " n/a"
        prec  = f"{m['precision']:.3f}" if m.get("precision") is not None else " n/a"
        rec   = f"{m['recall']:.3f}"    if m.get("recall")    is not None else " n/a"
        mrr   = f"{m['mrr']:.3f}"   if m.get("mrr")   is not None else " n/a"
        edr3  = f"{m['edr_at_3']:.3f}" if m.get("edr_at_3") is not None else " n/a"

        print(f"│  F1={f1}  AUROC={auroc}  P={prec}  R={rec}  "
              f"MRR={mrr}  EDR@3={edr3}")
        print(f"└─ {r.elapsed:.0f}s  fail={r.n_fail}/{len(all_rows)}\n")

        results.append(r)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"{'='*66}")
    print("  SUMMARY")
    print(f"{'='*66}")
    print(f"  {'Combo':<30} {'N':>4} {'Fail':>5} {'F1':>6} "
          f"{'AUROC':>6} {'MRR':>6} {'EDR@3':>6}")
    print(f"  {'-'*30} {'-'*4} {'-'*5} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for r in results:
        m     = r.metrics
        combo = f"{r.benchmark}/{r.framework}"
        n_tot = r.n_attack + r.n_benign

        if "error" in m and "f1" not in m:
            print(f"  {combo:<30} {n_tot:>4} {r.n_fail:>5}  ERROR: {m['error']}")
            continue

        def _f(k):
            v = m.get(k)
            return f"{v:.3f}" if v is not None else "  n/a"

        print(f"  {combo:<30} {n_tot:>4} {r.n_fail:>5} "
              f"{_f('f1'):>6} {_f('auroc'):>6} {_f('mrr'):>6} {_f('edr_at_3'):>6}")

    # ── Full metrics ───────────────────────────────────────────────────────
    try:
        from eval.metrics import compute_detection_metrics_from_rows, print_metrics
        print(f"\n{'='*66}")
        print("  FULL METRICS")
        print(f"{'='*66}")
        for r in results:
            rows = _read_task_csv(_out_dir(r.benchmark, r.framework, r.config))
            if not rows:
                continue
            det = compute_detection_metrics_from_rows(rows)
            print(f"\n  {r.benchmark}/{r.framework}/{r.config}  "
                  f"(attack={r.n_attack} benign={r.n_benign})")
            print_metrics(det)
    except Exception as e:
        print(f"\n  Could not print full metrics: {e}")

    # ── Save summary ───────────────────────────────────────────────────────
    summary_path = ROOT / "outputs" / "smoke_test" / "smoke_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([{
            "benchmark": r.benchmark,
            "framework": r.framework,
            "config":    r.config,
            "n_attack":  r.n_attack,
            "n_benign":  r.n_benign,
            "n_fail":    r.n_fail,
            "elapsed_s": round(r.elapsed, 1),
            "metrics":   r.metrics,
        } for r in results], f, indent=2, default=str)
    print(f"\n  Smoke summary → {summary_path}\n")


if __name__ == "__main__":
    main()