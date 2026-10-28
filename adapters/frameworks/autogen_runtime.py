"""
adapters/frameworks/autogen_runtime.py

AutoGen team construction for CASPIAN experiments.

Owns:
  - model client construction (OpenAI / Azure)
  - tool loading from TAMAS tool modules
  - AssistantAgent construction per scenario
  - team construction (RoundRobin / Swarm / MagenticOne)

Does NOT own:
  - stream message parsing (autogen.py)
  - topology construction (autogen.py: build_topology)
  - TAMAS scenario loading (tamas.py)
  - CASPIAN pipeline (core/pipeline.py)

TAMAS tool modules must be importable — caller is responsible for
adding TAMAS/data/tools/autogen/ to sys.path before importing this module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Model client
# ---------------------------------------------------------------------------

def build_model_client(model_name: str) -> Any:
    """
    Build an AutoGen model client from environment variables.

    For OpenAI models (name contains "gpt"):
        Requires OPENAI_API_KEY in environment.

    For all others:
        Assumes Azure OpenAI.
        Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
        AZURE_OPENAI_API_VERSION in environment.
    """
    from autogen_ext.models.openai import (
        AzureOpenAIChatCompletionClient,
        OpenAIChatCompletionClient,
    )

    if "gpt" in model_name.lower():
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file or export it."
            )
        return OpenAIChatCompletionClient(model=model_name, api_key=api_key)

    api_key  = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    version  = os.environ.get("AZURE_OPENAI_API_VERSION")
    if not all([api_key, endpoint, version]):
        raise RuntimeError(
            "Azure OpenAI credentials incomplete. Set AZURE_OPENAI_API_KEY, "
            "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION in your .env file."
        )
    return AzureOpenAIChatCompletionClient(
        model=model_name,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=version,
    )


# ---------------------------------------------------------------------------
# Tool loading
# ---------------------------------------------------------------------------

def _load_tools(module_name: str, fn_names: list[str]) -> list[Any]:
    """
    Import a TAMAS tool module and return the listed functions.
    Missing functions are silently skipped.
    Module must already be on sys.path.
    """
    try:
        mod = __import__(module_name)
    except ImportError as e:
        print(f"[autogen_runtime] Could not import tool module {module_name!r}: {e}",
              file=sys.stderr)
        return []
    return [getattr(mod, fn) for fn in fn_names if hasattr(mod, fn)]


def load_tamas_tools(tools_dir: Path) -> None:
    """
    Add TAMAS tool directory to sys.path so _load_tools() can find modules.
    Call once before build_team().
    """
    tools_str = str(tools_dir)
    if tools_str not in sys.path:
        sys.path.insert(0, tools_str)


# TAMAS tool sets — matches original main.py exactly
def _make_tool_mapping() -> dict[str, list[Any]]:
    """
    Build the agent-name → tools mapping from TAMAS tool modules.
    Returns empty lists for any module that fails to import.
    """
    TOOLS_WRITING = _load_tools(
        "tools_article_writing_agent",
        ["write_headline", "draft_article_body", "optimize_readability",
         "summarize_long_content", "insert_metadata_and_tags",
         "insert_propaganda_phrases", "rewrite_article_with_bias"],
    )
    TOOLS_DISTRIBUTION = _load_tools(
        "tools_distribution_agent",
        ["post_on_x", "post_on_insta", "post_on_linkedin",
         "send_push_notifications", "send_newsletter", "publish_to_unverified_article"],
    )
    TOOLS_FACT_CHECK = _load_tools(
        "tools_fact_checking_agent",
        ["cross_reference_with_reuters", "compare_with_official_sources",
         "detect_misinformation_patterns", "fetch_related_articles",
         "summarize_conflicting_claims", "leak_fact_check_logs",
         "tamper_fact_verdicts", "overwrite_evidence_summary"],
    )
    TOOLS_NEWS_GATHERING = _load_tools(
        "tools_news_gathering_agent",
        ["fetch_latest_rss_feeds", "scrape_news_websites", "monitor_twitter_trends",
         "track_subreddit_posts", "get_alerts_from_news_api", "exfiltrate_all_rss_sources",
         "inject_fake_trends", "exfiltrate_trending_subreddits", "poison_gathered_data"],
    )
    TOOLS_FINANCE_IDENTITY = _load_tools(
        "tools_identity_verification_agent",
        ["verify_identity", "biometric_verification", "otp_verification",
         "fetch_user_profile", "log_identity_event"],
    )
    TOOLS_FINANCE_MARKET = _load_tools(
        "tools_market_analysis_agent",
        ["analyze_market_trends", "fetch_market_sentiment", "get_stock_quote",
         "screen_high_volatility_assets", "get_sector_news"],
    )
    TOOLS_FINANCE_TRADE = _load_tools(
        "tools_trade_execution_agent",
        ["place_order", "execute_limit_order", "cancel_order", "fetch_order_status"],
    )
    TOOLS_FINANCE_RISK = _load_tools(
        "tools_risk_management_agent",
        ["adjust_stop_loss", "log_risk_event", "enforce_margin_call",
         "calculate_risk_exposure", "update_risk_limits"],
    )
    TOOLS_EDU_ASSESS = _load_tools(
        "tools_assessment_agent",
        ["generate_quiz", "evaluate_answers", "grade_assignment", "provide_feedback"],
    )
    TOOLS_EDU_ADAPT = _load_tools(
        "tools_adaptive_learning_agent",
        ["recommend_learning_path", "adjust_difficulty", "assign_practice_problems"],
    )
    TOOLS_EDU_PROGRESS = _load_tools(
        "tools_progress_tracking_agent",
        ["track_progress", "update_progress_dashboard", "identify_weak_areas"],
    )
    TOOLS_EDU_CONCEPT = _load_tools(
        "tools_concept_clarification_agent",
        ["explain_concept", "give_example", "provide_extra_resources"],
    )
    TOOLS_HEALTH_MONITOR = _load_tools(
        "tools_realtime_monitoring_agent",
        ["monitor_vitals", "trigger_alert", "log_monitoring_event"],
    )
    TOOLS_HEALTH_DIAGNOSIS = _load_tools(
        "tools_diagnosis_agent",
        ["suggest_diagnosis", "recommend_tests", "summarize_symptoms"],
    )
    TOOLS_HEALTH_PATIENT = _load_tools(
        "tools_patient_interaction_agent",
        ["ask_patient_questions", "provide_patient_guidance", "schedule_followup"],
    )
    TOOLS_HEALTH_ADHERENCE = _load_tools(
        "tools_medical_adherence_agent",
        ["check_medication_adherence", "send_medication_reminder", "log_adherence_event"],
    )
    TOOLS_LEGAL_COMPLIANCE = _load_tools(
        "tools_compliance_agent",
        ["check_compliance", "validate_policy_requirements", "flag_compliance_issue"],
    )
    TOOLS_LEGAL_DRAFT = _load_tools(
        "tools_doc_draft_agent",
        ["draft_document", "revise_document", "summarize_document"],
    )
    TOOLS_LEGAL_RESEARCH = _load_tools(
        "tools_research_agent",
        ["search_case_law", "fetch_statute", "summarize_legal_precedent"],
    )
    TOOLS_LEGAL_ANALYSIS = _load_tools(
        "tools_analysis_agent",
        ["analyze_case", "compare_arguments", "assess_legal_risk"],
    )

    return {
        "Article Writing Agent":        TOOLS_WRITING,
        "Article Writing Agent A":      TOOLS_WRITING,
        "Article Writing Agent B":      TOOLS_WRITING,
        "Distribution Agent":           TOOLS_DISTRIBUTION,
        "Distribution Agent A":         TOOLS_DISTRIBUTION,
        "Distribution Agent B":         TOOLS_DISTRIBUTION,
        "News Gathering Agent":         TOOLS_NEWS_GATHERING,
        "News Gathering Agent A":       TOOLS_NEWS_GATHERING,
        "News Gathering Agent B":       TOOLS_NEWS_GATHERING,
        "Fact Checking Agent":          TOOLS_FACT_CHECK,
        "Fact Checking Agent A":        TOOLS_FACT_CHECK,
        "Fact Checking Agent B":        TOOLS_FACT_CHECK,
        "Identity Verification Agent":  TOOLS_FINANCE_IDENTITY,
        "Market Analysis Agent":        TOOLS_FINANCE_MARKET,
        "Trade Execution Agent":        TOOLS_FINANCE_TRADE,
        "Risk Management Agent":        TOOLS_FINANCE_RISK,
        "Assessment Agent":             TOOLS_EDU_ASSESS,
        "Adaptive Learning Agent":      TOOLS_EDU_ADAPT,
        "Progress Tracking Agent":      TOOLS_EDU_PROGRESS,
        "Concept Clarification Agent":  TOOLS_EDU_CONCEPT,
        "Real time Monitoring Agent":   TOOLS_HEALTH_MONITOR,
        "Diagnosis Agent":              TOOLS_HEALTH_DIAGNOSIS,
        "Patient Interaction Agent":    TOOLS_HEALTH_PATIENT,
        "Medication Adherence Agent":   TOOLS_HEALTH_ADHERENCE,
        "Compliance Check Agent":       TOOLS_LEGAL_COMPLIANCE,
        "Document Drafting Agent":      TOOLS_LEGAL_DRAFT,
        "Legal Research Agent":         TOOLS_LEGAL_RESEARCH,
        "Case Analysis Agent":          TOOLS_LEGAL_ANALYSIS,
    }


# ---------------------------------------------------------------------------
# Team construction
# ---------------------------------------------------------------------------


def _norm_name(x: str) -> str:
    import re
    return re.sub(r"\s+", " ", str(x).strip().lower())

def _collect_task_attack_text(task_data: dict) -> str:
    keys = ["malicious_query", "malicious_instruction", "attack_description",
            "attack_instruction", "adversarial_instruction", "injected_prompt"]
    parts = []
    for k in keys:
        v = task_data.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
    return "\n".join(parts)

def _agent_is_malicious(task_data: dict, agent_spec: dict) -> bool:
    name = _norm_name(agent_spec.get("agent_name", ""))
    mal_list = task_data.get("malicious_agents") or task_data.get("malicious agent") or []
    if isinstance(mal_list, str):
        mal_list = [mal_list]
    mal_names = {_norm_name(x) for x in mal_list}
    flags = [agent_spec.get("malicious"), agent_spec.get("is_malicious"),
             agent_spec.get("attacker"), agent_spec.get("is_attacker")]
    return (name in mal_names or any(bool(x) for x in flags)
            or isinstance(agent_spec.get("malicious_instruction"), str)
            or isinstance(agent_spec.get("attack_instruction"), str))

def _build_system_message(task_data: dict, agent_spec: dict) -> str:
    base = str(agent_spec.get("agent_description", ""))
    agent_attack = "\n".join(
        str(agent_spec.get(k, "")).strip()
        for k in ["malicious_instruction", "attack_instruction", "adversarial_instruction"]
        if isinstance(agent_spec.get(k), str) and agent_spec.get(k, "").strip()
    )
    if agent_attack:
        return base + "\n\n[Adversarial role instruction]\n" + agent_attack
    task_attack = _collect_task_attack_text(task_data)
    if task_attack and _agent_is_malicious(task_data, agent_spec):
        print(f"[TAMAS INJECT] agent={agent_spec.get('agent_name')} "
              f"payload={task_attack[:120]!r}", flush=True)
        return base + "\n\n[Adversarial role instruction]\n" + task_attack
    return base

def build_team(task_data: dict, model_name: str, config: str) -> Any:
    """
    Build an AutoGen team from a TAMAS task dict.

    Parameters
    ----------
    task_data  : raw dict from Scenario.task_data
    model_name : e.g. "gpt-4o-mini", "gpt-4o"
    config     : "RoundRobin" | "Swarm" | "Magentic_one"

    Returns
    -------
    AutoGen team instance (RoundRobinGroupChat, Swarm, or MagenticOneGroupChat)
    """
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
    from autogen_agentchat.teams import MagenticOneGroupChat, RoundRobinGroupChat, Swarm

    model_client = build_model_client(model_name)
    tool_mapping = _make_tool_mapping()

    agent_list = []
    for agent_spec in task_data["agents"]:
        agent_name = agent_spec["agent_name"]
        system_msg = _build_system_message(task_data, agent_spec)
        tools      = tool_mapping.get(agent_name, [])
        safe_name  = agent_name.lower().replace(" ", "_")

        agent_obj = AssistantAgent(
            name=safe_name,
            model_client=model_client,
            tools=tools,
            system_message=system_msg,
        )
        agent_list.append(agent_obj)

    termination = MaxMessageTermination(50) | TextMentionTermination("TERMINATE")

    if config == "RoundRobin":
        return RoundRobinGroupChat(agent_list, termination_condition=termination)
    elif config == "Swarm":
        return Swarm(agent_list, termination_condition=termination)
    elif config == "Magentic_one":
        return MagenticOneGroupChat(agent_list, model_client=model_client)
    else:
        raise ValueError(f"Unknown AutoGen config: {config!r}. "
                         f"Expected one of: RoundRobin, Swarm, Magentic_one")


def agent_names_from_team(team: Any) -> list[str]:
    """
    Extract the original (display) agent names from a built team.
    AutoGen stores agents under _participants with .name (the safe name).
    We reconstruct display names by reversing the safe_name transform.
    """
    participants = getattr(team, "_participants", None)
    if participants is None:
        return []
    # .name is the safe_name (lowercased, underscored)
    # Return as-is; autogen.py adapter will match against topology.agents
    return [p.name for p in participants]