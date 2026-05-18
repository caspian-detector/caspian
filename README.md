<h1 align="center">CASPIAN</h1>

<p align="center">
  <strong>Online Detection and Attribution of Cascade Attacks in LLM Multi-Agent Systems via Cross-Channel Causal Monitoring</strong><br>
  <em>Cross-channel causal monitoring for safer, more interpretable multi-agent AI</em>
</p>

<p align="center">
  Kavana Venkatesh · Jafar Isbarov · Saad Amin · Murat Kantarcioglu · Jiaming Cui<br>
  <strong>Virginia Tech</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-f59e0b?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/Status-Active%20Development-10b981?style=flat-square"></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b?style=flat-square"></a>
</p>


## Abstract 

Cascade attacks in LLM multi-agent systems (MAS) arise when adversarial influence prop-
agates across agents and leads to escalated system-level failures through complex agent
interactions. Detecting such cascades is challenging, as their signals are distributed, tightly
coupled across interaction channels, and often appear plausibly benign locally but may unfold
quickly either within a single turn or gradually across multiple turns. Existing defenses,
being largely local and text-centric, fail to capture such cross-channel, temporally coordinated
dynamics of cascade propagation. Therefore, we propose **CASPIAN**, the first framework that
provides a unified, cross-channel causal analysis of cascade behavior in LLM-MAS through
online monitoring of dynamic influence propagation across agents. CASPIAN models multi-
agent interactions using a unified, dynamic causal influence matrix across channels, estimated
efficiently via a late-interaction conditional transfer entropy (LI-CTE) formulation, thereby
enabling the detection of cascade onset from emergent system-level structure rather than
isolated anomalies. It further performs online causal attribution, identifying the origin, bridge,
and amplifier agents driving the cascade and reconstructing its principal propagation pathways,
capabilities not supported by existing methods. Across diverse multi-agent frameworks and
benchmarks, CASPIAN consistently outperforms semantic guardrails, LLM-based judges,
and graph-based anomaly detectors in both detection accuracy and early cascade identification
while operating with sub-1% relative overhead latency. These results demonstrate that unified
cross-channel causal modeling is essential for reliably detecting and understanding cascade
failures in LLM multi-agent systems.

<img src="images/caspian-method.png" width="1000">


## Setup

```bash
git clone https://github.com/caspian-detector/caspian/
cd caspian-mas
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
```

---

## Benchmarks

Clone both benchmarks into the repo root before running:

```bash
# TAMAS — 400 scenarios across 6 attack categories + 100 benign
git clone https://github.com/your-org/TAMAS TAMAS

# ACIArena — 8 attack classes across disclosure, disruption, hijacking
git clone https://github.com/your-org/ACIArena aciarena
cp configs/model.yaml.example configs/model.yaml  # set model + API key
```

---

## Running Experiments

Runs resume automatically after interruption — completed scenario IDs are read from existing CSVs and skipped.

### TAMAS

```bash
# Full run (attack + benign)
python -m experiments.run_matrix --benchmark TAMAS --framework AutoGen   --config RoundRobin    --model gpt-4o-mini
python -m experiments.run_matrix --benchmark TAMAS --framework CrewAI    --config Decentralized --model gpt-4o-mini
python -m experiments.run_matrix --benchmark TAMAS --framework MetaGPT   --config standard      --model gpt-4o-mini
python -m experiments.run_matrix --benchmark TAMAS --framework LLMDebate --config standard      --model gpt-4o-mini

# Attack scenarios only
python -m experiments.run_matrix --benchmark TAMAS --framework AutoGen --config RoundRobin --model gpt-4o-mini --attack_only

# Benign scenarios only
python -m experiments.run_matrix --benchmark TAMAS --framework AutoGen --config RoundRobin --model gpt-4o-mini --benign_only

# Limit number of scenarios (useful for debugging)
python -m experiments.run_matrix --benchmark TAMAS --framework AutoGen --config RoundRobin --model gpt-4o-mini --limit 10
```

### ACIArena

```bash
# Full run — diverse attack sampling across all 8 attack classes
python -m experiments.run_matrix --benchmark ACIArena --framework AutoGen   --config standard --model gpt-4o-mini --aci_attack all
python -m experiments.run_matrix --benchmark ACIArena --framework CrewAI    --config standard --model gpt-4o-mini --aci_attack all
python -m experiments.run_matrix --benchmark ACIArena --framework MetaGPT   --config standard --model gpt-4o-mini --aci_attack all --malicious_agents engineer
python -m experiments.run_matrix --benchmark ACIArena --framework LLMDebate --config standard --model gpt-4o-mini --aci_attack all

# Specific attack class only
python -m experiments.run_matrix --benchmark ACIArena --framework LLMDebate --config standard --model gpt-4o-mini --aci_attack CodeApikeyLeakInstruction

# Benign only
python -m experiments.run_matrix --benchmark ACIArena --framework LLMDebate --config standard --model gpt-4o-mini --aci_attack NoneAttack
```

---

## Evaluation

```bash
# Detection metrics — precision, recall, F1, AUROC, TPR@5%FPR, EDR, MRR
python -m eval.detection_eval --all --verbose

# Attribution metrics — origin, amplifier, bridge accuracy; spine Jaccard
python -m eval.generate_attribution_gt --all
python -m eval.attribution_eval --all --verbose

# Filter by benchmark or framework
python -m eval.detection_eval --benchmark TAMAS --verbose
python -m eval.detection_eval --framework LLMDebate --verbose

# Export results as JSON
python -m eval.detection_eval --all --json > results/detection.json
python -m eval.attribution_eval --all --json > results/attribution.json
```

---

## Citation

```bibtex
@inproceedings{venkatesh2025caspian,
  title     = {CASPIAN: Online Detection and Attribution of Cascade Attacks in LLM Multi-Agent Systems via Cross-Channel Causal Monitoring},
  author    = {Venkatesh, Kavana and Isbarov, Jafar and Amin, Saad and Kantarcioglu, Murat and Cui, Jiaming},
  journal = {arXiv preprint arXiv:0000.12345},
  year    = {2026}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
