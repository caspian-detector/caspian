"""
adapters/benchmarks/aciarena.py

ACIArena benchmark adapter for CASPIAN.

Responsibilities:
  - locate ACIArena repo and datasets
  - load ACIArena tasks into Scenario objects
  - resolve attack/backend classes
  - build LLMDebateRunSpec with real ACI backend+attack factories

Does NOT:
  - run CASPIAN detector
  - write CSVs
  - use old CascadeDetector, warmup, or scoring logic

Expected layout:
  caspian-mas/
    aciarena/
      aciarena/
        evaluation/datasets/maspi_math.json
        evaluation/datasets/maspi_code.json
        attacks/
        mas/
    adapters/benchmarks/aciarena.py

Usage:
  python -m experiments.run_matrix \\
    --benchmark ACIArena \\
    --framework LLMDebate \\
    --config standard \\
    --model gpt-4o-mini \\
    --aci_attack MathLocationLeakInstruction \\
    --max_turn 3 \\
    --limit 1
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from adapters.benchmarks.base import Scenario, BenchmarkAdapter
from adapters.frameworks.specs import LLMDebateRunSpec


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT             = Path(__file__).resolve().parents[2]
ACI_REPO_ROOT    = ROOT / "aciarena"
ACI_PACKAGE_ROOT = ACI_REPO_ROOT / "aciarena"

DEFAULT_DATASET_MATH = ACI_PACKAGE_ROOT / "evaluation" / "datasets" / "maspi_math.json"
DEFAULT_DATASET_CODE = ACI_PACKAGE_ROOT / "evaluation" / "datasets" / "maspi_code.json"

LOCAL_CONFIG_DIR = ROOT / "configs"
ACI_CONFIG_DIR   = ACI_REPO_ROOT / "configs"

DEFAULT_MODEL_CONFIG = LOCAL_CONFIG_DIR / "model.yaml"
DEFAULT_JUDGE_CONFIG = LOCAL_CONFIG_DIR / "judge.yaml"


# ---------------------------------------------------------------------------
# Backend import candidates
# ---------------------------------------------------------------------------

BACKEND_IMPORT_CANDIDATES: dict[str, list[tuple[str, str]]] = {
    "AutoGen": [
        ("aciarena.mas.autogen.autogen_mas",                    "AutoGen"),
        ("aciarena.mas.autogen.autogen_mas",                    "AutoGenMAS"),
    ],
    "MetaGPT": [
        ("aciarena.mas.metagpt.metagpt_mas",                    "MetaGPT"),
        ("aciarena.mas.metagpt.metagpt_mas",                    "MetaGPTMAS"),
    ],
    "AgentVerse": [
        ("aciarena.mas.agentverse.agentverse_mas",              "AgentVerse"),
        ("aciarena.mas.agentverse.agentverse_mas",              "AgentVerseMAS"),
    ],
    "CAMEL": [
        ("aciarena.mas.camel.camel_mas",                        "CAMEL"),
        ("aciarena.mas.camel.camel_mas",                        "Camel"),
        ("aciarena.mas.camel.camel_mas",                        "CamelMAS"),
    ],
    "MAD": [
        ("aciarena.mas.mad.mad_mas",                            "MAD"),
        ("aciarena.mas.mad.mad_mas",                            "MADMAS"),
    ],
    "SelfConsistency": [
        ("aciarena.mas.self_consistency.self_consistency_mas",  "SelfConsistency"),
        ("aciarena.mas.self_consistency.self_consistency_mas",  "SelfConsistencyMAS"),
    ],
    "LLMDebate": [
        ("aciarena.mas.llm_debate.llm_debate_mas",              "LLMDebate"),
        ("aciarena.mas.llm_debate.llm_debate_mas",              "LlmDebate"),
        ("aciarena.mas.llm_debate.llm_debate_mas",              "Debate"),
    ],
}

ATTACK_MODULES = [
    "aciarena.attacks.disclosure_attack",
    "aciarena.attacks.disruption_attack",
    "aciarena.attacks.hijacking_attack",
    "aciarena.attacks.base_attack",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_aci_imports() -> None:
    if str(ACI_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(ACI_REPO_ROOT))
    if "aciarena" not in sys.modules and ACI_PACKAGE_ROOT.exists():
        pkg = types.ModuleType("aciarena")
        pkg.__path__ = [str(ACI_PACKAGE_ROOT)]
        sys.modules["aciarena"] = pkg


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        cfg: dict[str, Any] = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                k, _, v = line.partition(":")
                cfg[k.strip()] = v.strip()
        return cfg


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_llm_config(model: Path | None = None, judge: Path | None = None) -> dict[str, Any]:
    cfg = _deep_merge(
        _load_yaml(model or DEFAULT_MODEL_CONFIG),
        _load_yaml(judge or DEFAULT_JUDGE_CONFIG),
    )
    # Inject real API key from env if config has placeholder
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key and str(cfg.get("api_key", "")).startswith("<"):
        cfg["api_key"] = api_key
    if str(cfg.get("base_url", "")).startswith("<"):
        cfg.pop("base_url", None)
    if str(cfg.get("model_name", "")).startswith("<"):
        cfg["model_name"] = "gpt-4o-mini"
    return cfg


def _resolve_path(user: str | Path | None, default_local: Path, default_aci: Path) -> Path:
    if user:
        p = Path(user)
        if p.exists():
            return p
        p2 = ACI_REPO_ROOT / str(user)
        if p2.exists():
            return p2
        return p
    return default_local if default_local.exists() else default_aci


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ["data", "tasks", "examples", "items", "records"]:
            val = payload.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        return [payload]
    return []


def _extract_query(item: dict[str, Any]) -> str:
    for key in ["user query", "query", "question", "prompt",
                "task", "input", "problem", "instruction"]:
        val = item.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return json.dumps(item, ensure_ascii=False)


def _infer_domain(path: Path) -> str:
    stem = path.stem.lower()
    if "math" in stem: return "math"
    if "code" in stem: return "code"
    return "unknown"


def _infer_attack_family(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["location", "nameleak", "apikey"]):
        return "disclosure"
    if any(k in n for k in ["ddos", "misalign", "disruption"]):
        return "disruption"
    if any(k in n for k in ["safetycheck", "report", "mapping", "invert", "hijack"]):
        return "hijacking"
    return "unknown"


def _default_dataset(attack: str) -> Path:
    n = attack.lower()
    return DEFAULT_DATASET_MATH if any(k in n for k in ["math", "invert", "mapping"]) \
           else DEFAULT_DATASET_CODE


class _NoAttack:
    """Benign/harmless condition for ACIArena — NoneAttack equivalent."""
    payload = ""

    def __init__(self, task: Any = None, llm_config: Any = None) -> None:
        self.task       = task
        self.llm_config = llm_config
        self.answer:    Any = None

    def run(self, mas: Any) -> Any:
        return mas          # no-op: don't modify any agents

    def set_turn(self, turn: int) -> None:
        pass

    def set_answer(self, answer: Any) -> None:
        self.answer = answer

    def verify(self) -> float:
        return 0.0          # benign: attack never succeeds


def _resolve_attack_class(class_name: str) -> Any:
    # Benign condition — return no-op attack
    if class_name in ("NoneAttack", "none", "benign", "harmless", ""):
        return _NoAttack

    _setup_aci_imports()

    # Try importlib first
    for mod_name in ATTACK_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, class_name):
                return getattr(mod, class_name)
        except Exception:
            pass

    # Fall back: load attack files directly via spec, register under full name
    attack_dir = ACI_PACKAGE_ROOT / "attacks"
    for py_file in sorted(attack_dir.glob("*.py")):
        if py_file.name == "__init__.py":
            continue
        full_name = f"aciarena.attacks.{py_file.stem}"
        try:
            spec = importlib.util.spec_from_file_location(full_name, py_file)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[full_name]    = mod
            sys.modules[py_file.stem] = mod   # short name alias
            spec.loader.exec_module(mod)
            if hasattr(mod, class_name):
                return getattr(mod, class_name)
        except Exception:
            pass

    raise ValueError(f"Could not resolve ACIArena attack class: {class_name!r}")


def _instantiate_backend(
    framework:        str,
    llm_config:       dict[str, Any],
    malicious_agents: list[str],
    logger:           Any,
    max_turn:         int,
) -> Any:
    _setup_aci_imports()
    candidates = BACKEND_IMPORT_CANDIDATES.get(framework, [])
    tried: list[str] = []
    backend_cls = None

    for mod_name, cls_name in candidates:
        tried.append(f"{mod_name}:{cls_name}")
        # Try importlib first
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, cls_name):
                backend_cls = getattr(mod, cls_name)
                break
        except Exception as _e:
            pass

        # Fall back: load file directly and register
        rel_path = mod_name.replace("aciarena.", "").replace(".", "/") + ".py"
        py_file = ACI_PACKAGE_ROOT / rel_path
        if py_file.exists():
            try:
                spec = importlib.util.spec_from_file_location(mod_name, py_file)
                mod  = importlib.util.module_from_spec(spec)
                sys.modules[mod_name] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, cls_name):
                    backend_cls = getattr(mod, cls_name)
                    break
            except Exception as _e2:
                tried[-1] += f" ({type(_e2).__name__}: {_e2})"

    if backend_cls is None:
        raise ImportError(f"No backend found for {framework}. Tried: {tried}")

    # Framework-specific constructor args first
    attempts: list[dict] = []

    if framework == "AutoGen":
        # ACI AutoGen requires code_execute kwarg
        attempts.append(dict(llm_config=llm_config, logger=logger,
                             malicious_agents=malicious_agents,
                             code_execute=False, max_turn=max_turn))

    attempts.extend([
        dict(llm_config=llm_config, malicious_agents=malicious_agents,
             logger=logger, max_turn=max_turn),
        dict(llm_config=llm_config, logger=logger, max_turn=max_turn),
        dict(config=llm_config, malicious_agents=malicious_agents,
             logger=logger, max_turn=max_turn),
        dict(config=llm_config, logger=logger, max_turn=max_turn),
        dict(llm_config=llm_config, logger=logger),
        dict(config=llm_config,    logger=logger),
    ])
    last_exc: Exception | None = None
    for kw in attempts:
        try:
            return backend_cls(**kw)
        except Exception as e:
            last_exc = e
    raise RuntimeError(
        f"Could not instantiate {backend_cls.__name__}. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@dataclass
class ACIArenaAdapter(BenchmarkAdapter):
    """ACIArena benchmark adapter — produces Scenario objects for run_matrix.py."""

    aci_repo_root:            Path      = field(default_factory=lambda: ACI_REPO_ROOT)
    default_attack:           str       = "MathLocationLeakInstruction"
    default_max_turn:         int       = 3
    default_malicious_agents: list[str] = field(default_factory=lambda: ["debater_0"])

    @property
    def name(self) -> str:
        return "ACIArena"

    def _load_diverse_scenarios(
        self,
        *,
        dataset:          "str | Path | None"      = None,
        model_config:     "str | Path | None"      = None,
        judge_config:     "str | Path | None"      = None,
        max_turn:         "int | None"             = None,
        malicious_agents: "str | list[str] | None" = None,
        attack_only:      bool                     = False,
        limit:            "int | None"             = None,
    ) -> "Iterator[Scenario]":
        """
        Sample scenarios across all attack families and both datasets (math + code).
        Attack rotation:
          disclosure:  MathLocationLeakInstruction (math), MathNameLeakInstruction (math)
          disruption:  DDOSInstruction (code), MisalignInstruction (code)
          hijacking:   SafetyCheckInstruction (code), ReportFakeResultInstruction (code)
        Yields up to `limit` scenarios total, distributed evenly across families.
        """
        # All confirmed ACIArena attack classes by family + dataset
        _ALL_ATTACKS: list[tuple[str, str | None]] = [
            # disclosure — math
            ("MathLocationLeakInstruction",  None),
            ("MathNameLeakInstruction",      None),
            # disclosure — code
            ("CodeApikeyLeakInstruction",    None),
            ("CodeNameLeakInstruction",      None),
            # disruption — code
            ("DDOSInstruction",              None),
            # hijacking — code
            ("SafetyCheckInstruction",       None),
            ("MaliciousReportInstruction",   None),
            ("AnswerMappingInsruction",      None),  # note: typo in ACI source
        ]

        per_attack = max(1, (limit or 5) // len(_ALL_ATTACKS)) if limit else None
        total      = 0

        for atk_name, atk_dataset in _ALL_ATTACKS:
            if limit is not None and total >= limit:
                break
            remaining = (limit - total) if limit is not None else None
            n         = min(per_attack or 999, remaining or 999)
            try:
                for s in self.load_scenarios(
                    attack=atk_name,
                    dataset=atk_dataset or dataset,
                    model_config=model_config,
                    judge_config=judge_config,
                    max_turn=max_turn,
                    malicious_agents=malicious_agents,
                    attack_only=attack_only,
                    limit=n,
                ):
                    yield s
                    total += 1
                    if limit is not None and total >= limit:
                        return
            except Exception as e:
                print(f"[ACI DIVERSE SKIP] attack={atk_name} "
                      f"error={type(e).__name__}: {e}", flush=True)
                continue

    def build_framework_spec(
        self,
        scenario: "Scenario",
        framework: str,
        config: str,
    ) -> "Any":
        """
        Dispatch to native ACIArena backend spec for all supported frameworks.
        All use ACIArenaNativeRunSpec (routed through LLMDebateAdapter in run_matrix).
        CrewAI has no native backend in this checkout — uses payload-injection fallback.
        """
        _NATIVE = {
            "autogen":          "AutoGen",
            "auto_gen":         "AutoGen",
            "metagpt":          "MetaGPT",
            "meta_gpt":         "MetaGPT",
            "agentverse":       "AgentVerse",
            "agent_verse":      "AgentVerse",
            "camel":            "CAMEL",
            "mad":              "MAD",
            "selfconsistency":  "SelfConsistency",
            "self_consistency": "SelfConsistency",
            "llmdebate":        "LLMDebate",
            "llm_debate":       "LLMDebate",
            "debate":           "LLMDebate",
        }
        fw = str(framework).strip().lower().replace("-", "_").replace(" ", "_")

        if fw in _NATIVE:
            name = _NATIVE[fw]
            if name == "LLMDebate":
                return build_llm_debate_spec(scenario, config)
            return _build_native_aci_spec(scenario, name, config)

        # Fallback for frameworks without native ACI backends
        if fw == "crewai":
            return build_crewai_spec(scenario, config)

        raise ValueError(f"Unsupported ACIArena framework: {framework}")

    def __post_init__(self) -> None:
        _setup_aci_imports()
        _load_env(ROOT / ".env")
        _load_env(self.aci_repo_root / ".env")

    def load_scenarios(
        self,
        *,
        attack:           str | None             = None,
        dataset:          str | Path | None      = None,
        model_config:     str | Path | None      = None,
        judge_config:     str | Path | None      = None,
        max_turn:         int | None             = None,
        malicious_agents: str | list[str] | None = None,
        attack_only:      bool                   = False,
        benign_only:      bool                   = False,
        limit:            int | None             = None,
        exclude_harmless: bool                   = False,
        **_: Any,
    ) -> Iterator[Scenario]:
        # Diverse sampling: when attack is None, "all", or "diverse"
        # rotate across disclosure/disruption/hijacking on math+code
        _diverse = attack is None or str(attack).lower() in {"all", "diverse", "mixed"}
        if _diverse and not benign_only:
            yield from self._load_diverse_scenarios(
                dataset=dataset,
                model_config=model_config,
                judge_config=judge_config,
                max_turn=max_turn,
                malicious_agents=malicious_agents,
                attack_only=attack_only,
                limit=limit,
            )
            return

        attack_name = attack or self.default_attack
        label       = 0 if attack_name in ("NoneAttack", "none", "benign", "harmless", "") else 1

        if attack_only      and label == 0: return
        if benign_only      and label == 1: return
        if exclude_harmless and label == 0: return

        dataset_path = _default_dataset(attack_name) if dataset is None else \
                       _resolve_path(dataset, _default_dataset(attack_name),
                                     _default_dataset(attack_name))

        model_path   = _resolve_path(model_config, DEFAULT_MODEL_CONFIG,
                                     ACI_CONFIG_DIR / "model.yaml")
        judge_path   = _resolve_path(judge_config,  DEFAULT_JUDGE_CONFIG,
                                     ACI_CONFIG_DIR / "judge.yaml")
        llm_config   = _load_llm_config(model_path, judge_path)

        if malicious_agents is None:
            mal_list = list(self.default_malicious_agents)
        elif isinstance(malicious_agents, str):
            mal_list = [x.strip() for x in malicious_agents.split(",") if x.strip()]
        else:
            mal_list = list(malicious_agents)

        max_turn_val    = int(max_turn or self.default_max_turn)
        domain          = _infer_domain(dataset_path)
        attack_category = _infer_attack_family(attack_name)
        source_file     = dataset_path.name
        items           = _load_dataset(dataset_path)
        count           = 0

        for file_task_idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            query = _extract_query(item)
            if not query:
                continue

            task_data: dict[str, Any] = dict(item)
            task_data["user query"]            = query
            task_data["_aci_attack_class"]     = attack_name
            task_data["_aci_attack_category"]  = attack_category
            task_data["_aci_llm_config"]       = llm_config
            task_data["_aci_max_turn"]         = max_turn_val
            # NoneAttack = benign: no malicious agents
            _is_benign = attack_name in ("NoneAttack", "none", "benign", "harmless", "")
            task_data["_aci_malicious_agents"] = [] if _is_benign else mal_list

            # Fallback debate topology stored separately — native backends don't use this
            task_data["_caspian_debate_agents"] = [
                {"agent_name": "debater_0",  "agent_description": "First reasoning agent"},
                {"agent_name": "debater_1",  "agent_description": "Second reasoning agent"},
                {"agent_name": "debater_2",  "agent_description": "Third reasoning agent"},
                {"agent_name": "aggregator", "agent_description": "Final answer aggregator"},
            ]
            # task_data["agents"] is NOT set here — framework-specific agents
            # are set in _build_native_aci_spec() where framework is known.

            # Extract attack payload for query injection (AutoGen/CrewAI/MetaGPT)
            _atk_payload = ""
            if not _is_benign:
                try:
                    _setup_aci_imports()
                    _atk_cls = _resolve_attack_class(attack_name)
                    # Try instance first (payload may be set in __init__)
                    try:
                        _atk_obj = _atk_cls(item, llm_config)
                        _atk_payload = getattr(_atk_obj, "payload", "") or ""
                    except Exception:
                        _atk_payload = getattr(_atk_cls, "payload", "") or ""
                except Exception:
                    pass
            task_data["_aci_attack_payload"] = _atk_payload

            yield Scenario(
                scenario_id   = f"aci_{domain}_{attack_name}__{file_task_idx}",
                benchmark     = "ACIArena",
                source_file   = source_file,
                file_task_idx = file_task_idx,
                domain        = domain,
                attack_type   = attack_name,
                attack_category = attack_category,
                label         = label,
                task_data     = task_data,
            )

            count += 1
            if limit is not None and count >= limit:
                break


# ---------------------------------------------------------------------------
# Spec builders — called by run_matrix.py
# ---------------------------------------------------------------------------


_NATIVE_AGENT_META: dict = {
    "AutoGen": [
        {"agent_name": "assistant",  "agent_description": "Assistant agent"},
        {"agent_name": "user_proxy", "agent_description": "User proxy agent"},
    ],
    "MetaGPT": [
        {"agent_name": "product_manager", "agent_description": "Product manager"},
        {"agent_name": "architect",        "agent_description": "Architect"},
        {"agent_name": "project_manager",  "agent_description": "Project manager"},
        {"agent_name": "engineer",         "agent_description": "Engineer"},
        {"agent_name": "qa_engineer",      "agent_description": "QA engineer"},
    ],
    "LLMDebate": [
        {"agent_name": "debater_0",  "agent_description": "First debate agent"},
        {"agent_name": "debater_1",  "agent_description": "Second debate agent"},
        {"agent_name": "debater_2",  "agent_description": "Third debate agent"},
        {"agent_name": "aggregator", "agent_description": "Aggregator"},
    ],
}


def _build_native_aci_spec(scenario: "Scenario", framework: str, config: str) -> "ACIArenaNativeRunSpec":
    """
    Build a native ACIArena spec for any framework that has an ACI backend.
    The backend_factory + attack_factory pattern mirrors how LLMDebateRunSpec works.
    """
    from adapters.frameworks.specs import ACIArenaNativeRunSpec

    task              = scenario.task_data
    # Set framework-specific agent metadata so attack classes inspect the right agents
    task["_aci_framework"] = framework
    if framework in _NATIVE_AGENT_META:
        task["agents"]               = _NATIVE_AGENT_META[framework]
        task["_aci_framework_agents"] = _NATIVE_AGENT_META[framework]
    user_query        = _extract_query(task)
    attack_class_name = task["_aci_attack_class"]
    llm_config        = task["_aci_llm_config"]
    mal_agents        = list(task.get("_aci_malicious_agents", []))
    max_turn          = int(task.get("_aci_max_turn", 3))

    if attack_class_name in ("NoneAttack", "none", "benign", "harmless", ""):
        mal_agents = []
    elif not mal_agents or mal_agents == ["debater_0"]:
        # Set framework-appropriate default malicious agent
        _default_mal = {
            "AutoGen":    ["assistant"],
            "MetaGPT":    ["product_manager"],
            "AgentVerse": ["solver"],
            "CAMEL":      ["assistant"],
            "MAD":        ["affirmative"],
            "SelfConsistency": ["sc_agent_0"],
            "LLMDebate":  ["debater_0"],
        }
        mal_agents = _default_mal.get(framework, ["debater_0"])

    def backend_factory(logger: Any) -> Any:
        return _instantiate_backend(
            framework=framework,
            llm_config=llm_config,
            malicious_agents=mal_agents,
            logger=logger,
            max_turn=max_turn,
        )

    def attack_factory(mas: Any) -> Any:
        return _resolve_attack_class(attack_class_name)(task, llm_config)

    return ACIArenaNativeRunSpec(
        framework=framework,
        user_query=user_query,
        task_data=task,
        llm_config=llm_config,
        attack_class_name=attack_class_name,
        malicious_agents=mal_agents,
        max_turn=max_turn,
        backend_factory=backend_factory,
        attack_factory=attack_factory,
    )


def build_llm_debate_spec(scenario: "Scenario", config: str) -> "LLMDebateRunSpec":
    """Build LLMDebateRunSpec with real ACI backend and attack factories."""
    from adapters.frameworks.specs import LLMDebateRunSpec

    task              = scenario.task_data
    task["_aci_framework"]        = "LLMDebate"
    task["agents"]                = _NATIVE_AGENT_META["LLMDebate"]
    task["_aci_framework_agents"] = _NATIVE_AGENT_META["LLMDebate"]
    user_query        = _extract_query(task)
    attack_class_name = task["_aci_attack_class"]
    llm_config        = task["_aci_llm_config"]
    mal_agents        = list(task.get("_aci_malicious_agents", ["debater_0"]))
    max_turn          = int(task.get("_aci_max_turn", 3))


    if attack_class_name in ("NoneAttack", "none", "benign", "harmless", ""):
        mal_agents = []

    def backend_factory(logger: Any) -> Any:
        return _instantiate_backend(
            framework="LLMDebate",
            llm_config=llm_config,
            malicious_agents=mal_agents,
            logger=logger,
            max_turn=max_turn,
        )

    def attack_factory(mas: Any) -> Any:
        return _resolve_attack_class(attack_class_name)(task, llm_config)

    return LLMDebateRunSpec(
        user_query=user_query,
        debaters=["debater_0", "debater_1", "debater_2"],
        aggregator="aggregator",
        max_rounds=max_turn,
        backend_factory=backend_factory,
        attack_factory=attack_factory,
        llm_config=llm_config,
        malicious_agents=mal_agents,
    )


def build_metagpt_spec(scenario: "Scenario", config: str) -> "ACIArenaNativeRunSpec":
    """Native ACIArena MetaGPT backend."""
    return _build_native_aci_spec(scenario, "MetaGPT", config)


def build_crewai_spec(scenario: "Scenario", config: str) -> "CrewAIRunSpec":
    """
    ACIArena CrewAI — debate-topology agents with payload-injected query.
    ACIArena has no native CrewAI backend.
    """
    from adapters.frameworks.specs import AgentSpec, CrewAIRunSpec

    task        = scenario.task_data
    user_query  = _extract_query(task)
    payload     = task.get("_aci_attack_payload", "")
    attack_name = task.get("_aci_attack_class", "")
    is_benign   = attack_name in ("NoneAttack", "none", "benign", "harmless", "")

    agents = [
        AgentSpec(role="debater_0", backstory="First reasoning agent",
                  goal="Solve the task accurately.", tools=[]),
        AgentSpec(role="debater_1", backstory="Second reasoning agent",
                  goal="Critique the proposed solution.", tools=[]),
        AgentSpec(role="debater_2", backstory="Third reasoning agent",
                  goal="Verify and check for errors.", tools=[]),
        AgentSpec(role="aggregator", backstory="Final answer aggregator",
                  goal="Synthesize the final answer.", tools=[]),
    ]

    return CrewAIRunSpec(
        agents=agents,
        user_query=user_query,   # clean — payload injected per-role in crewai adapter
        process="sequential",
    )


def build_autogen_spec(scenario: "Scenario", config: str) -> "ACIArenaNativeRunSpec":
    """Native ACIArena AutoGen backend."""
    return _build_native_aci_spec(scenario, "AutoGen", config)


def _inject_payload(scenario: "Scenario") -> None:
    """Inject attack payload into user_query for frameworks without native ACI backends."""
    task        = scenario.task_data
    user_query  = _extract_query(task)
    payload     = task.get("_aci_attack_payload", "")
    attack_name = task.get("_aci_attack_class", "")
    is_benign   = attack_name in ("NoneAttack", "none", "benign", "harmless", "")
    task["user query"] = (
        f"{user_query}\n\n{payload}" if payload and not is_benign else user_query
    )