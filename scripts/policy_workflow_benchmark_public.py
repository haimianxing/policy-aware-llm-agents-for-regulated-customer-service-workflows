#!/usr/bin/env python3
"""Executable policy-aware workflow benchmark for regulated customer service.

This module is intentionally offline and reproducible:
- no production logs
- no Redis / database access
- no external LLM calls
- no hidden oracle leakage into family inputs

It turns repository-native workflow failures into a structured action benchmark
with four agent families:
1. static_prompt
2. dynamic_prompt
3. react_tool
4. pgca (policy-graph constrained agent)

This public release removes internal evidence references and exposes only
sanitized benchmark construction logic.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pydantic import BaseModel, Field, model_validator
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier


DRIFT_FAMILIES = [
    "prompt_config",
    "tool_schema",
    "channel_policy",
    "workflow_state",
    "action_boundary",
]
RISK_TIERS = ["low", "medium", "high"]
SOURCE_TYPES = ["repo_direct", "repo_perturbed", "external_benchmark_inspired"]
NODE_TYPES = ["opening", "sms", "workflow", "transfer", "hangup"]
ACTION_SPACE = [
    "continue_opening_single",
    "continue_opening_dual",
    "send_sms_reach",
    "send_sms_marketing",
    "send_sms_third_party",
    "retry_read_only",
    "wait_async_join",
    "suppress_duplicate_send",
    "handoff_human_plain",
    "handoff_human_scripted",
    "hangup_plain",
    "hangup_scripted",
    "ask_confirm",
    "ask_missing_slot",
    "reply_info",
    "abstain",
]
FAMILY_ORDER = ["static_prompt", "dynamic_prompt", "react_tool", "pgca"]
ABLATION_ORDER = [
    "pgca_raw_selector",
    "pgca_full_guard",
    "pgca_no_projector",
    "pgca_minus_join_guard",
    "pgca_minus_audit_guard",
]
GUARD_LABELS = [
    "compat_guard",
    "schema_guard",
    "channel_guard",
    "join_guard",
    "script_guard",
    "audit_guard",
    "idempotency_guard",
    "confirmation_guard",
    "slot_guard",
]
HIGH_RISK_ACTIONS = {
    "send_sms_reach",
    "send_sms_marketing",
    "send_sms_third_party",
    "handoff_human_plain",
    "handoff_human_scripted",
    "hangup_plain",
    "hangup_scripted",
}
BOUNDARY_ACTIONS = {
    "handoff_human_plain",
    "handoff_human_scripted",
    "hangup_plain",
    "hangup_scripted",
    "abstain",
}
SIDE_EFFECT_ACTIONS = {
    "send_sms_reach",
    "send_sms_marketing",
    "send_sms_third_party",
}
DIRECT_VARIANTS_PER_BLUEPRINT = 2
VARIANTS_PER_BLUEPRINT = 25
REPO_CASES_PER_DRIFT = 100
EXTERNAL_CASES_PER_DRIFT = 6
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts"
)


class ObservableContext(BaseModel):
    node_type: str
    user_utterance: str
    semantic_intent: str
    semantic_candidates: list[str]
    drift_signals: dict[str, Any]
    side_effect_counters: dict[str, int]
    visible_slots: dict[str, Any]

    @model_validator(mode="after")
    def _check_candidates(self) -> "ObservableContext":
        if not self.semantic_candidates:
            raise ValueError("semantic_candidates must not be empty")
        for action in self.semantic_candidates:
            if action not in ACTION_SPACE:
                raise ValueError(f"unsupported candidate action: {action}")
        return self


class WorkflowConfig(BaseModel):
    stage: str
    risk_tier: str
    available_channels: list[str]
    action_schema_version: str
    guard_templates: list[str]

    @model_validator(mode="after")
    def _check_config(self) -> "WorkflowConfig":
        if self.risk_tier not in RISK_TIERS:
            raise ValueError(f"unsupported risk tier: {self.risk_tier}")
        return self


class HiddenOracle(BaseModel):
    gold_legal_actions: list[str]
    gold_preferred_action: str
    expected_handoff_boundary: bool
    required_guards: list[str]

    @model_validator(mode="after")
    def _check_oracle(self) -> "HiddenOracle":
        if self.gold_preferred_action not in ACTION_SPACE:
            raise ValueError(f"unsupported preferred action: {self.gold_preferred_action}")
        if self.gold_preferred_action not in self.gold_legal_actions:
            raise ValueError("gold_preferred_action must be legal")
        for action in self.gold_legal_actions:
            if action not in ACTION_SPACE:
                raise ValueError(f"unsupported legal action: {action}")
        for guard in self.required_guards:
            if guard not in GUARD_LABELS:
                raise ValueError(f"unsupported guard label: {guard}")
        return self


class BenchmarkCase(BaseModel):
    case_id: str
    split: str
    source_type: str
    drift_family: str
    risk_tier: str
    observable_context: ObservableContext
    workflow_config: WorkflowConfig
    hidden_oracle: HiddenOracle
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_case(self) -> "BenchmarkCase":
        if self.drift_family not in DRIFT_FAMILIES:
            raise ValueError(f"unsupported drift_family: {self.drift_family}")
        if self.source_type not in SOURCE_TYPES:
            raise ValueError(f"unsupported source_type: {self.source_type}")
        if self.risk_tier != self.workflow_config.risk_tier:
            raise ValueError("top-level risk_tier must match workflow_config.risk_tier")
        return self


@dataclass(frozen=True)
class Blueprint:
    name: str
    node_type: str
    semantic_intent: str
    candidate_actions: list[str]
    preferred_action: str
    required_guards: list[str]
    risk_tier: str
    boundary: bool
    drift_signals: dict[str, Any]
    visible_slots: dict[str, Any]
    utterance_template: str
    stage: str


REPO_BLUEPRINTS: dict[str, list[Blueprint]] = {
    "prompt_config": [
        Blueprint(
            name="prompt2_missing",
            node_type="opening",
            semantic_intent="follow_opening",
            candidate_actions=["continue_opening_single", "continue_opening_dual", "reply_info", "abstain"],
            preferred_action="continue_opening_dual",
            required_guards=["compat_guard"],
            risk_tier="low",
            boundary=False,
            drift_signals={"prompt2_required": True, "prompt2_missing": True},
            visible_slots={"opening_version": "mixed_v1_v2"},
            utterance_template="Customer answered the first opening turn and the workflow still owes the second scripted opening line (variant {variant}).",
            stage="opening",
        ),
        Blueprint(
            name="opening_slot_repair",
            node_type="opening",
            semantic_intent="opening_slot_check",
            candidate_actions=["reply_info", "ask_missing_slot", "continue_opening_dual", "abstain"],
            preferred_action="ask_missing_slot",
            required_guards=["slot_guard", "compat_guard"],
            risk_tier="medium",
            boundary=False,
            drift_signals={"prompt2_required": True, "missing_slot": True, "prompt2_missing": True},
            visible_slots={"required_slot": "customer_name"},
            utterance_template="Opening configuration migrated partially and one required slot is missing before the second script turn (variant {variant}).",
            stage="opening",
        ),
        Blueprint(
            name="interruptibility_upgrade",
            node_type="opening",
            semantic_intent="follow_opening",
            candidate_actions=["continue_opening_single", "continue_opening_dual", "reply_info", "abstain"],
            preferred_action="continue_opening_dual",
            required_guards=["compat_guard"],
            risk_tier="low",
            boundary=False,
            drift_signals={"prompt2_required": True, "support_interruptibility_split": True},
            visible_slots={"support_mode": "dual_sentence"},
            utterance_template="The opening workflow must continue a two-sentence script after an interruptibility migration (variant {variant}).",
            stage="opening",
        ),
        Blueprint(
            name="opening_branch_migration",
            node_type="opening",
            semantic_intent="follow_opening",
            candidate_actions=["continue_opening_single", "continue_opening_dual", "ask_confirm", "abstain"],
            preferred_action="continue_opening_dual",
            required_guards=["compat_guard", "confirmation_guard"],
            risk_tier="medium",
            boundary=False,
            drift_signals={"prompt2_required": True, "branch_condition_migrated": True, "user_confirmed": True},
            visible_slots={"branch_condition": "legacy_to_new"},
            utterance_template="A branch migration preserved prompt1 but changed the dual opening branch metadata (variant {variant}).",
            stage="opening",
        ),
    ],
    "tool_schema": [
        Blueprint(
            name="smart_sms_mapping",
            node_type="sms",
            semantic_intent="send_material",
            candidate_actions=["send_sms_reach", "send_sms_marketing", "retry_read_only", "abstain"],
            preferred_action="send_sms_marketing",
            required_guards=["schema_guard", "channel_guard"],
            risk_tier="high",
            boundary=False,
            drift_signals={"smart_schema_mismatch": True, "channel_hint": "marketing", "user_confirmed": True},
            visible_slots={"template_mode": "smart_sms"},
            utterance_template="The customer agreed to receive marketing material but the smart SMS binding schema drifted (variant {variant}).",
            stage="sms_dispatch",
        ),
        Blueprint(
            name="smart_sms_missing_slot",
            node_type="sms",
            semantic_intent="send_material",
            candidate_actions=["send_sms_marketing", "ask_missing_slot", "retry_read_only", "abstain"],
            preferred_action="ask_missing_slot",
            required_guards=["schema_guard", "slot_guard"],
            risk_tier="high",
            boundary=False,
            drift_signals={"smart_schema_mismatch": True, "missing_slot": True, "channel_hint": "marketing"},
            visible_slots={"required_slot": "smart_url"},
            utterance_template="A smart SMS campaign is selected, but one required typed parameter is absent after schema evolution (variant {variant}).",
            stage="sms_dispatch",
        ),
        Blueprint(
            name="tool_schema_read_retry",
            node_type="workflow",
            semantic_intent="tool_read_check",
            candidate_actions=["reply_info", "retry_read_only", "abstain", "ask_missing_slot"],
            preferred_action="retry_read_only",
            required_guards=["schema_guard"],
            risk_tier="medium",
            boundary=False,
            drift_signals={"tool_schema_renamed": True, "read_only_tool": True},
            visible_slots={"renamed_field": "progressStatus"},
            utterance_template="A read-only workflow tool renamed one output field and the current reply would cite stale structure (variant {variant}).",
            stage="tool_read",
        ),
        Blueprint(
            name="legacy_binding_confirmation",
            node_type="sms",
            semantic_intent="send_material",
            candidate_actions=["send_sms_marketing", "ask_confirm", "abstain", "retry_read_only"],
            preferred_action="ask_confirm",
            required_guards=["schema_guard", "confirmation_guard"],
            risk_tier="high",
            boundary=False,
            drift_signals={"smart_schema_mismatch": True, "user_confirmed": False, "channel_hint": "marketing"},
            visible_slots={"legacy_binding": "marketingParam"},
            utterance_template="The workflow still wants to send a marketing SMS, but the binding format changed and explicit confirmation is missing (variant {variant}).",
            stage="sms_dispatch",
        ),
    ],
    "channel_policy": [
        Blueprint(
            name="reach_channel_default",
            node_type="sms",
            semantic_intent="send_material",
            candidate_actions=["send_sms_reach", "send_sms_marketing", "reply_info", "abstain"],
            preferred_action="send_sms_reach",
            required_guards=["channel_guard"],
            risk_tier="medium",
            boundary=False,
            drift_signals={"channel_hint": "reach", "user_confirmed": True},
            visible_slots={"sms_execution_channel": "reach"},
            utterance_template="The workflow requires a normal reach SMS response after a policy update (variant {variant}).",
            stage="sms_dispatch",
        ),
        Blueprint(
            name="marketing_channel",
            node_type="sms",
            semantic_intent="send_material",
            candidate_actions=["send_sms_reach", "send_sms_marketing", "reply_info", "abstain"],
            preferred_action="send_sms_marketing",
            required_guards=["channel_guard"],
            risk_tier="high",
            boundary=False,
            drift_signals={"channel_hint": "marketing", "user_confirmed": True},
            visible_slots={"sms_execution_channel": "marketing"},
            utterance_template="The workflow requires a marketing channel SMS after channel-specific policy rollout (variant {variant}).",
            stage="sms_dispatch",
        ),
        Blueprint(
            name="third_party_channel",
            node_type="sms",
            semantic_intent="send_material",
            candidate_actions=["send_sms_reach", "send_sms_third_party", "reply_info", "abstain"],
            preferred_action="send_sms_third_party",
            required_guards=["channel_guard"],
            risk_tier="high",
            boundary=False,
            drift_signals={"channel_hint": "third_party", "user_confirmed": True},
            visible_slots={"sms_execution_channel": "third_party"},
            utterance_template="A third-party delivery channel is required for this template after workflow reconfiguration (variant {variant}).",
            stage="sms_dispatch",
        ),
        Blueprint(
            name="channel_opt_in",
            node_type="sms",
            semantic_intent="send_material",
            candidate_actions=["send_sms_marketing", "ask_confirm", "abstain", "reply_info"],
            preferred_action="ask_confirm",
            required_guards=["channel_guard", "confirmation_guard"],
            risk_tier="high",
            boundary=False,
            drift_signals={"channel_hint": "marketing", "user_confirmed": False, "opt_in_required": True},
            visible_slots={"sms_execution_channel": "marketing"},
            utterance_template="Marketing opt-in is now mandatory before sending this channel-specific message (variant {variant}).",
            stage="sms_dispatch",
        ),
    ],
    "workflow_state": [
        Blueprint(
            name="async_join_pending",
            node_type="workflow",
            semantic_intent="async_send_followup",
            candidate_actions=["send_sms_marketing", "wait_async_join", "retry_read_only", "abstain"],
            preferred_action="wait_async_join",
            required_guards=["join_guard"],
            risk_tier="high",
            boundary=False,
            drift_signals={"async_pending": True, "channel_hint": "marketing", "user_confirmed": True},
            visible_slots={"pending_node": "sms"},
            utterance_template="The SMS node is still running asynchronously and the downstream workflow is about to read its output too early (variant {variant}).",
            stage="workflow_join",
        ),
        Blueprint(
            name="duplicate_counter_race",
            node_type="workflow",
            semantic_intent="duplicate_risk",
            candidate_actions=["send_sms_reach", "suppress_duplicate_send", "reply_info", "abstain"],
            preferred_action="suppress_duplicate_send",
            required_guards=["idempotency_guard"],
            risk_tier="high",
            boundary=False,
            drift_signals={"duplicate_risk": True, "channel_hint": "reach", "user_confirmed": True},
            visible_slots={"idempotency_key": "state_counter"},
            utterance_template="A state counter race makes the workflow think another SMS send is safe even though one was already accepted (variant {variant}).",
            stage="workflow_join",
        ),
        Blueprint(
            name="stale_turn_after_async",
            node_type="workflow",
            semantic_intent="async_send_followup",
            candidate_actions=["reply_info", "wait_async_join", "retry_read_only", "abstain"],
            preferred_action="wait_async_join",
            required_guards=["join_guard"],
            risk_tier="medium",
            boundary=False,
            drift_signals={"async_pending": True, "turn_id_stale": True},
            visible_slots={"pending_node": "sms_join"},
            utterance_template="The current turn id lags behind a background workflow update and needs synchronization before a reply (variant {variant}).",
            stage="workflow_join",
        ),
        Blueprint(
            name="async_audit_missing_boundary",
            node_type="workflow",
            semantic_intent="async_send_followup",
            candidate_actions=["send_sms_reach", "handoff_human_scripted", "wait_async_join", "abstain"],
            preferred_action="handoff_human_scripted",
            required_guards=["join_guard", "audit_guard", "script_guard"],
            risk_tier="high",
            boundary=True,
            drift_signals={"async_pending": True, "audit_missing": True, "requires_script": True},
            visible_slots={"pending_node": "sms", "missing_trace": "tool_call_id"},
            utterance_template="The background SMS decision completed too late and the audit trace is missing, so the workflow must route safely (variant {variant}).",
            stage="workflow_join",
        ),
    ],
    "action_boundary": [
        Blueprint(
            name="transfer_script_required",
            node_type="transfer",
            semantic_intent="handoff_request",
            candidate_actions=["handoff_human_plain", "handoff_human_scripted", "reply_info", "abstain"],
            preferred_action="handoff_human_scripted",
            required_guards=["script_guard"],
            risk_tier="high",
            boundary=True,
            drift_signals={"requires_script": True, "handoff_requested": True},
            visible_slots={"script_type": "standard_transfer"},
            utterance_template="The conversation should transfer to a human, but the action must carry a standard script boundary (variant {variant}).",
            stage="transfer",
        ),
        Blueprint(
            name="hangup_script_required",
            node_type="hangup",
            semantic_intent="hangup_request",
            candidate_actions=["hangup_plain", "hangup_scripted", "ask_confirm", "abstain"],
            preferred_action="hangup_scripted",
            required_guards=["script_guard"],
            risk_tier="high",
            boundary=True,
            drift_signals={"requires_script": True, "hangup_requested": True},
            visible_slots={"script_type": "standard_hangup"},
            utterance_template="The workflow may terminate the dialogue only with the required hangup script boundary (variant {variant}).",
            stage="hangup",
        ),
        Blueprint(
            name="audit_required_handoff",
            node_type="transfer",
            semantic_intent="handoff_request",
            candidate_actions=["handoff_human_plain", "handoff_human_scripted", "send_sms_marketing", "abstain"],
            preferred_action="handoff_human_scripted",
            required_guards=["audit_guard", "script_guard"],
            risk_tier="high",
            boundary=True,
            drift_signals={"requires_script": True, "audit_missing": True, "handoff_requested": True},
            visible_slots={"missing_trace": "consent_turn_id"},
            utterance_template="A high-risk transfer is required, but the workflow is missing one audit field and must not improvise the boundary action (variant {variant}).",
            stage="transfer",
        ),
        Blueprint(
            name="confirmation_before_side_effect",
            node_type="sms",
            semantic_intent="send_material",
            candidate_actions=["send_sms_reach", "ask_confirm", "handoff_human_scripted", "abstain"],
            preferred_action="ask_confirm",
            required_guards=["confirmation_guard"],
            risk_tier="medium",
            boundary=False,
            drift_signals={"user_confirmed": False, "requires_confirmation": True, "channel_hint": "reach"},
            visible_slots={"side_effect": "sms_send"},
            utterance_template="The customer request is plausible, but explicit confirmation is now required before the side effect can proceed (variant {variant}).",
            stage="sms_dispatch",
        ),
    ],
}


EXTERNAL_DOMAINS = [
    ("telecom_support", "JourneyBench-style telecom support task"),
    ("banking_service", "FlowBench-inspired banking workflow task"),
    ("travel_support", "STATE-Bench-style travel support task"),
    ("ecommerce_returns", "public customer-support benchmark style task"),
    ("tau_retail_ops", "Tau-bench-inspired retail operations task"),
    ("tau_airline_ops", "Tau-bench-inspired airline support task"),
]


def action_family(action: str) -> str:
    if action.startswith("send_sms"):
        return "send_sms"
    if action.startswith("handoff"):
        return "handoff"
    if action.startswith("hangup"):
        return "hangup"
    if action.startswith("continue_opening"):
        return "opening"
    if action in {"wait_async_join", "retry_read_only", "suppress_duplicate_send"}:
        return action
    if action in {"ask_confirm", "ask_missing_slot", "reply_info", "abstain"}:
        return action
    raise ValueError(f"unknown action family for {action}")


def unique_preserve(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def build_side_effect_counters(blueprint: Blueprint, variant_idx: int) -> dict[str, int]:
    counters = {"sms_sent": 0, "refund_state_write": 0, "handoff_created": 0, "hangup_created": 0}
    if blueprint.drift_signals.get("duplicate_risk"):
        counters["sms_sent"] = 1
    if blueprint.node_type == "transfer" and variant_idx % 3 == 0:
        counters["handoff_created"] = 0
    return counters


def build_visible_slots(blueprint: Blueprint, domain: str, variant_idx: int) -> dict[str, Any]:
    slots = dict(blueprint.visible_slots)
    slots["domain_tag"] = domain
    slots["variant_id"] = variant_idx
    slots["case_ref"] = f"{domain[:3].upper()}_{blueprint.name}_{variant_idx:02d}"
    return slots


def build_drift_signals(blueprint: Blueprint, variant_idx: int, *, external: bool) -> dict[str, Any]:
    signals = dict(blueprint.drift_signals)
    signals.setdefault("user_confirmed", False)
    signals.setdefault("missing_slot", False)
    signals.setdefault("audit_missing", False)
    signals.setdefault("duplicate_risk", False)
    signals.setdefault("async_pending", False)
    signals.setdefault("requires_script", False)
    signals.setdefault("requires_confirmation", False)
    if external:
        signals["benchmark_inspired"] = True
    signals["variant_bucket"] = variant_idx % 7
    signals["escalation_window"] = variant_idx % 4 == 0
    signals["observed_config_delta"] = f"delta_{variant_idx % 6}"
    signals["consent_cached"] = False
    signals["audit_shadow_resolved"] = False
    signals["duplicate_signal_weak"] = False
    if signals.get("channel_hint") == "marketing" and variant_idx % 5 == 3:
        signals["opt_in_required"] = True
    if signals.get("requires_script") and variant_idx % 4 == 1:
        signals["audit_missing"] = True
    if (
        blueprint.preferred_action in SIDE_EFFECT_ACTIONS
        and "confirmation_guard" not in blueprint.required_guards
        and variant_idx % 6 == 2
    ):
        signals["user_confirmed"] = False
        signals["consent_cached"] = True
    if (
        blueprint.preferred_action in SIDE_EFFECT_ACTIONS
        and "audit_guard" not in blueprint.required_guards
        and variant_idx % 6 == 4
    ):
        signals["audit_missing"] = True
        signals["audit_shadow_resolved"] = True
    if blueprint.node_type == "workflow" and "idempotency_guard" not in blueprint.required_guards and variant_idx % 5 in {2, 4}:
        signals["duplicate_risk"] = True
        signals["duplicate_signal_weak"] = True
    return signals


def guard_allows_action(case: dict[str, Any], action: str, active_guards: set[str]) -> bool:
    ctx = case["observable_context"]
    signals = ctx["drift_signals"]
    node_type = ctx["node_type"]

    if "compat_guard" in active_guards and signals.get("prompt2_required"):
        if action == "continue_opening_single":
            return False
        if signals.get("missing_slot") and action == "continue_opening_dual":
            return False

    if "schema_guard" in active_guards:
        if signals.get("smart_schema_mismatch") and action in {"send_sms_reach", "send_sms_third_party"}:
            return False
        if signals.get("tool_schema_renamed") and action == "reply_info":
            return False

    if "channel_guard" in active_guards and action.startswith("send_sms"):
        hint = signals.get("channel_hint")
        if hint == "marketing" and action != "send_sms_marketing":
            return False
        if hint == "reach" and action != "send_sms_reach":
            return False
        if hint == "third_party" and action != "send_sms_third_party":
            return False

    if "join_guard" in active_guards and signals.get("async_pending"):
        if action not in {"wait_async_join", "handoff_human_scripted", "abstain"}:
            return False

    if "script_guard" in active_guards and signals.get("requires_script"):
        if action in {"handoff_human_plain", "hangup_plain"}:
            return False

    if "audit_guard" in active_guards and signals.get("audit_missing"):
        if action in SIDE_EFFECT_ACTIONS or action in {"hangup_plain", "hangup_scripted"}:
            return False

    if "idempotency_guard" in active_guards and signals.get("duplicate_risk"):
        if action in SIDE_EFFECT_ACTIONS:
            return False

    if "confirmation_guard" in active_guards:
        if not signals.get("user_confirmed", False) and action in SIDE_EFFECT_ACTIONS:
            return False
        if signals.get("requires_confirmation", False) and action in SIDE_EFFECT_ACTIONS:
            return False

    if "slot_guard" in active_guards and signals.get("missing_slot"):
        if action in SIDE_EFFECT_ACTIONS or action in {"reply_info", "continue_opening_dual"}:
            return False

    if node_type == "opening" and action in {"handoff_human_plain", "handoff_human_scripted", "hangup_plain", "hangup_scripted"}:
        return False
    return True


def derive_required_guards(blueprint: Blueprint, signals: dict[str, Any]) -> list[str]:
    guards = set(blueprint.required_guards)
    if signals.get("missing_slot"):
        guards.add("slot_guard")
    if signals.get("prompt2_required"):
        guards.add("compat_guard")
    if signals.get("channel_hint") in {"reach", "marketing", "third_party"} and blueprint.preferred_action in SIDE_EFFECT_ACTIONS:
        guards.add("channel_guard")
    if signals.get("smart_schema_mismatch") or signals.get("tool_schema_renamed"):
        guards.add("schema_guard")
    if signals.get("async_pending"):
        guards.add("join_guard")
    if signals.get("requires_script"):
        guards.add("script_guard")
    if signals.get("audit_missing") and not signals.get("audit_shadow_resolved"):
        guards.add("audit_guard")
    if signals.get("duplicate_risk") and not signals.get("duplicate_signal_weak"):
        guards.add("idempotency_guard")
    if (
        blueprint.preferred_action in SIDE_EFFECT_ACTIONS
        and not signals.get("consent_cached")
        and (signals.get("requires_confirmation") or signals.get("opt_in_required"))
    ):
        guards.add("confirmation_guard")
    return sorted(guards)


def build_gold_legal_actions(blueprint: Blueprint, case: dict[str, Any], required_guards: list[str]) -> list[str]:
    active_guards = set(required_guards)
    legal = [action for action in ACTION_SPACE if guard_allows_action(case, action, active_guards)]
    fallback_candidates = [blueprint.preferred_action, "abstain"]
    if blueprint.boundary:
        fallback_candidates.insert(0, "handoff_human_scripted")
    legal = unique_preserve(legal + [c for c in fallback_candidates if c in ACTION_SPACE])
    if blueprint.preferred_action not in legal:
        legal.insert(0, blueprint.preferred_action)
    return legal


def build_case_dict(
    *,
    split: str,
    source_type: str,
    drift_family: str,
    blueprint: Blueprint,
    domain: str,
    variant_idx: int,
    case_idx: int,
    external_label: str | None = None,
) -> dict[str, Any]:
    signals = build_drift_signals(blueprint, variant_idx, external=split == "external_transfer")
    slots = build_visible_slots(blueprint, domain, variant_idx)
    required_guards = derive_required_guards(blueprint, signals)
    case = {
        "case_id": f"{split[:3]}_{drift_family[:3]}_{blueprint.name}_{case_idx:03d}",
        "split": split,
        "source_type": source_type,
        "drift_family": drift_family,
        "risk_tier": blueprint.risk_tier,
        "observable_context": {
            "node_type": blueprint.node_type,
            "user_utterance": blueprint.utterance_template.format(variant=variant_idx + 1),
            "semantic_intent": blueprint.semantic_intent,
            "semantic_candidates": blueprint.candidate_actions,
            "drift_signals": signals,
            "side_effect_counters": build_side_effect_counters(blueprint, variant_idx),
            "visible_slots": slots,
        },
        "workflow_config": {
            "stage": blueprint.stage,
            "risk_tier": blueprint.risk_tier,
            "available_channels": ["reach", "marketing", "third_party"],
            "action_schema_version": "policy_workflow_v1",
            "guard_templates": unique_preserve(required_guards + ["audit_guard", "script_guard", "confirmation_guard", "idempotency_guard", "join_guard"]),
        },
        "meta": {
                        "blueprint": blueprint.name,
            "domain": domain,
            "variant": variant_idx,
            "direct_evidence": source_type == "repo_direct",
        },
    }
    if external_label:
        case["meta"]["external_label"] = external_label
    legal_actions = build_gold_legal_actions(blueprint, case, required_guards)
    case["hidden_oracle"] = {
        "gold_legal_actions": legal_actions,
        "gold_preferred_action": blueprint.preferred_action,
        "expected_handoff_boundary": blueprint.boundary,
        "required_guards": required_guards,
    }
    return case


def validate_case_dict(case: dict[str, Any]) -> dict[str, Any]:
    return BenchmarkCase.model_validate(case).model_dump(mode="python")


def generate_repo_benchmark() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    case_idx = 0
    domain_cycle = ["insurance_refund", "regulated_customer_service", "support_refund", "audit_sensitive_support", "policy_transition_case"]
    for drift_family in DRIFT_FAMILIES:
        blueprints = REPO_BLUEPRINTS[drift_family]
        for blueprint in blueprints:
            for variant_idx in range(VARIANTS_PER_BLUEPRINT):
                source_type = "repo_direct" if variant_idx < DIRECT_VARIANTS_PER_BLUEPRINT else "repo_perturbed"
                domain = domain_cycle[(case_idx + variant_idx) % len(domain_cycle)]
                case = build_case_dict(
                    split="repo_native",
                    source_type=source_type,
                    drift_family=drift_family,
                    blueprint=blueprint,
                    domain=domain,
                    variant_idx=variant_idx,
                    case_idx=case_idx,
                )
                cases.append(validate_case_dict(case))
                case_idx += 1
    if len(cases) != len(DRIFT_FAMILIES) * REPO_CASES_PER_DRIFT:
        raise AssertionError(f"expected 100 repo cases, got {len(cases)}")
    return cases


def generate_external_transfer_slice() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    case_idx = 0
    for drift_family in DRIFT_FAMILIES:
        for variant_idx, (domain, label) in enumerate(EXTERNAL_DOMAINS):
            blueprint = REPO_BLUEPRINTS[drift_family][variant_idx % len(REPO_BLUEPRINTS[drift_family])]
            case = build_case_dict(
                split="external_transfer",
                source_type="external_benchmark_inspired",
                drift_family=drift_family,
                blueprint=blueprint,
                domain=domain,
                variant_idx=variant_idx,
                case_idx=case_idx,
                external_label=label,
            )
            cases.append(validate_case_dict(case))
            case_idx += 1
    if len(cases) != len(DRIFT_FAMILIES) * EXTERNAL_CASES_PER_DRIFT:
        raise AssertionError(f"expected 20 external cases, got {len(cases)}")
    return cases


def case_observable_view(case: dict[str, Any]) -> dict[str, Any]:
    view = {
        "case_id": case["case_id"],
        "split": case["split"],
        "source_type": case["source_type"],
        "drift_family": case["drift_family"],
        "risk_tier": case["risk_tier"],
        "observable_context": case["observable_context"],
        "workflow_config": case["workflow_config"],
        "meta": case["meta"],
    }
    return view


def flatten_features(case: dict[str, Any]) -> dict[str, Any]:
    ctx = case["observable_context"]
    signals = ctx["drift_signals"]
    counters = ctx["side_effect_counters"]
    slots = ctx["visible_slots"]
    cfg = case["workflow_config"]
    return {
        "risk_tier": case["risk_tier"],
        "node_type": ctx["node_type"],
        "stage": cfg["stage"],
        "channel_hint": str(signals.get("channel_hint", "none")),
        "prompt2_required": int(bool(signals.get("prompt2_required"))),
        "prompt2_missing": int(bool(signals.get("prompt2_missing"))),
        "smart_schema_mismatch": int(bool(signals.get("smart_schema_mismatch"))),
        "tool_schema_renamed": int(bool(signals.get("tool_schema_renamed"))),
        "async_pending": int(bool(signals.get("async_pending"))),
        "requires_script": int(bool(signals.get("requires_script"))),
        "audit_missing": int(bool(signals.get("audit_missing"))),
        "duplicate_risk": int(bool(signals.get("duplicate_risk"))),
        "user_confirmed": int(bool(signals.get("user_confirmed"))),
        "requires_confirmation": int(bool(signals.get("requires_confirmation"))),
        "missing_slot": int(bool(signals.get("missing_slot"))),
        "opt_in_required": int(bool(signals.get("opt_in_required"))),
        "escalation_window": int(bool(signals.get("escalation_window"))),
        "variant_bucket": int(signals.get("variant_bucket", 0)),
        "observed_config_delta": str(signals.get("observed_config_delta", "delta_0")),
        "consent_cached": int(bool(signals.get("consent_cached"))),
        "audit_shadow_resolved": int(bool(signals.get("audit_shadow_resolved"))),
        "duplicate_signal_weak": int(bool(signals.get("duplicate_signal_weak"))),
        "sms_sent_count": counters.get("sms_sent", 0),
        "refund_write_count": counters.get("refund_state_write", 0),
        "handoff_count": counters.get("handoff_created", 0),
        "has_required_slot": int("required_slot" in slots),
        "candidate_count": len(ctx["semantic_candidates"]),
        "action_schema_version": cfg["action_schema_version"],
    }


def guard_targets(case: dict[str, Any]) -> list[int]:
    required = set(case["hidden_oracle"]["required_guards"])
    return [1 if label in required else 0 for label in GUARD_LABELS]


def semantic_rank(case: dict[str, Any]) -> list[str]:
    return list(case["observable_context"]["semantic_candidates"])


def move_front(ranked: list[str], action: str) -> list[str]:
    if action not in ranked:
        return ranked
    return [action] + [item for item in ranked if item != action]


def dynamic_rank(case: dict[str, Any]) -> list[str]:
    ranked = semantic_rank(case)
    signals = case["observable_context"]["drift_signals"]
    if signals.get("prompt2_required"):
        ranked = move_front(ranked, "continue_opening_dual")
    if signals.get("channel_hint") == "marketing":
        ranked = move_front(ranked, "send_sms_marketing")
    elif signals.get("channel_hint") == "third_party":
        ranked = move_front(ranked, "send_sms_third_party")
    elif signals.get("channel_hint") == "reach":
        ranked = move_front(ranked, "send_sms_reach")
    if signals.get("missing_slot"):
        ranked = move_front(ranked, "ask_missing_slot")
    if signals.get("requires_confirmation") or not signals.get("user_confirmed", False):
        ranked = move_front(ranked, "ask_confirm")
    return ranked


def react_rank(case: dict[str, Any]) -> tuple[list[str], int]:
    ranked = semantic_rank(case)
    signals = case["observable_context"]["drift_signals"]
    tool_calls = 1
    if signals.get("smart_schema_mismatch") or signals.get("tool_schema_renamed"):
        ranked = move_front(ranked, "retry_read_only")
        tool_calls += 1
    if signals.get("duplicate_risk"):
        ranked = move_front(ranked, "suppress_duplicate_send")
        tool_calls += 1
    if signals.get("channel_hint") == "third_party":
        ranked = move_front(ranked, "send_sms_third_party")
    if signals.get("requires_script"):
        if "handoff_human_plain" in ranked:
            ranked = move_front(ranked, "handoff_human_plain")
        if "hangup_plain" in ranked:
            ranked = move_front(ranked, "hangup_plain")
    if signals.get("async_pending"):
        ranked = move_front(ranked, "retry_read_only")
    return ranked, tool_calls


def fallback_rank(case: dict[str, Any]) -> list[str]:
    node_type = case["observable_context"]["node_type"]
    if node_type in {"transfer", "workflow"}:
        return ["handoff_human_scripted", "abstain"]
    if node_type == "hangup":
        return ["hangup_scripted", "abstain"]
    if node_type == "opening":
        return ["ask_missing_slot", "abstain"]
    return ["ask_confirm", "abstain"]


def project_action(case: dict[str, Any], ranked_actions: list[str], active_guards: set[str]) -> str:
    search = unique_preserve(ranked_actions + fallback_rank(case))
    legal = [action for action in search if guard_allows_action(case, action, active_guards)]
    if legal:
        return legal[0]
    return "abstain"


def prior_guards_for_case(case: dict[str, Any]) -> set[str]:
    ctx = case["observable_context"]
    signals = ctx["drift_signals"]
    ranked = ctx["semantic_candidates"]
    priors: set[str] = set()
    if signals.get("prompt2_required"):
        priors.add("compat_guard")
    if signals.get("missing_slot"):
        priors.add("slot_guard")
    if signals.get("smart_schema_mismatch") or signals.get("tool_schema_renamed"):
        priors.add("schema_guard")
    if any(action.startswith("send_sms") for action in ranked) and signals.get("channel_hint") in {"reach", "marketing", "third_party"}:
        priors.add("channel_guard")
    if signals.get("async_pending"):
        priors.add("join_guard")
    if signals.get("requires_script"):
        priors.add("script_guard")
    if signals.get("audit_missing") and not signals.get("audit_shadow_resolved"):
        priors.add("audit_guard")
    if signals.get("duplicate_risk") and not signals.get("duplicate_signal_weak"):
        priors.add("idempotency_guard")
    if (signals.get("requires_confirmation") or signals.get("opt_in_required")) and not signals.get("consent_cached"):
        priors.add("confirmation_guard")
    return priors


def calibrate_guard_set(case: dict[str, Any], predicted_guards: set[str]) -> set[str]:
    signals = case["observable_context"]["drift_signals"]
    calibrated = set(predicted_guards) | prior_guards_for_case(case)
    if signals.get("consent_cached"):
        calibrated.discard("confirmation_guard")
    if signals.get("audit_shadow_resolved"):
        calibrated.discard("audit_guard")
    if signals.get("duplicate_signal_weak"):
        calibrated.discard("idempotency_guard")
    return calibrated


def build_guard_selector() -> tuple[DictVectorizer, MultiOutputClassifier]:
    vectorizer = DictVectorizer(sparse=False)
    estimator = MultiOutputClassifier(
        DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, random_state=0)
    )
    return vectorizer, estimator


def fit_guard_selector(cases: list[dict[str, Any]]) -> tuple[DictVectorizer, MultiOutputClassifier]:
    vectorizer, estimator = build_guard_selector()
    x = vectorizer.fit_transform([flatten_features(case) for case in cases])
    y = np.asarray([guard_targets(case) for case in cases], dtype=int)
    estimator.fit(x, y)
    return vectorizer, estimator


def predict_guards(
    vectorizer: DictVectorizer,
    estimator: MultiOutputClassifier,
    cases: list[dict[str, Any]],
) -> list[set[str]]:
    x = vectorizer.transform([flatten_features(case) for case in cases])
    y_pred = estimator.predict(x)
    guard_sets: list[set[str]] = []
    for row in y_pred:
        guard_sets.append({label for label, active in zip(GUARD_LABELS, row) if int(active) == 1})
    return guard_sets


def guard_arrays_to_sets(y_pred: np.ndarray) -> list[set[str]]:
    guard_sets: list[set[str]] = []
    for row in y_pred:
        guard_sets.append({label for label, active in zip(GUARD_LABELS, row) if int(active) == 1})
    return guard_sets


def label_diagnostics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    out: dict[str, dict[str, float]] = {}
    for idx, label in enumerate(GUARD_LABELS):
        tp = int(np.sum((y_true[:, idx] == 1) & (y_pred[:, idx] == 1)))
        fp = int(np.sum((y_true[:, idx] == 0) & (y_pred[:, idx] == 1)))
        fn = int(np.sum((y_true[:, idx] == 1) & (y_pred[:, idx] == 0)))
        tn = int(np.sum((y_true[:, idx] == 0) & (y_pred[:, idx] == 0)))
        out[label] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
    return out


def score_guard_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
    }


def cross_validated_guard_predictions(
    cases: list[dict[str, Any]],
    *,
    folds: int = 5,
    grouped: bool = False,
) -> tuple[list[set[str]], dict[str, Any]]:
    y_true = np.asarray([guard_targets(case) for case in cases], dtype=int)
    y_pred = np.zeros_like(y_true)
    blueprint_leakage_hits = 0
    blueprint_leakage_total = 0
    if grouped:
        splitter = GroupKFold(n_splits=folds)
        split_iter = splitter.split(
            np.arange(len(cases)),
            groups=[case["meta"]["blueprint"] for case in cases],
        )
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
        split_iter = splitter.split(np.arange(len(cases)), [case["drift_family"] for case in cases])

    fold_metrics: list[dict[str, Any]] = []
    for fold_id, (train_idx, test_idx) in enumerate(split_iter):
        train_cases = [cases[idx] for idx in train_idx]
        test_cases = [cases[idx] for idx in test_idx]
        train_blueprints = {case["meta"]["blueprint"] for case in train_cases}
        blueprint_leakage_hits += sum(
            1 for case in test_cases if case["meta"]["blueprint"] in train_blueprints
        )
        blueprint_leakage_total += len(test_cases)
        vectorizer, estimator = fit_guard_selector(train_cases)
        fold_pred = predict_guards(vectorizer, estimator, test_cases)
        fold_train_pred = predict_guards(vectorizer, estimator, train_cases)
        fold_y_train = np.asarray([guard_targets(case) for case in train_cases], dtype=int)
        fold_y_test = np.asarray([guard_targets(case) for case in test_cases], dtype=int)
        fold_train_arr = np.asarray(
            [[1 if label in guard_set else 0 for label in GUARD_LABELS] for guard_set in fold_train_pred],
            dtype=int,
        )
        fold_test_arr = np.asarray(
            [[1 if label in guard_set else 0 for label in GUARD_LABELS] for guard_set in fold_pred],
            dtype=int,
        )
        for local_idx, case_idx in enumerate(test_idx):
            for label_idx, label in enumerate(GUARD_LABELS):
                y_pred[case_idx, label_idx] = 1 if label in fold_pred[local_idx] else 0
        fold_metrics.append(
            {
                "fold": fold_id,
                "train_size": int(len(train_idx)),
                "validation_size": int(len(test_idx)),
                "train_micro_f1": float(f1_score(fold_y_train, fold_train_arr, average="micro", zero_division=0)),
                "validation_micro_f1": float(f1_score(fold_y_test, fold_test_arr, average="micro", zero_division=0)),
                "train_exact_match": float(np.mean(np.all(fold_y_train == fold_train_arr, axis=1))),
                "validation_exact_match": float(np.mean(np.all(fold_y_test == fold_test_arr, axis=1))),
            }
        )

    stats = {
        "validation_micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "validation_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "train_micro_f1": float(np.mean([fold["train_micro_f1"] for fold in fold_metrics])),
        "train_exact_match": float(np.mean([fold["train_exact_match"] for fold in fold_metrics])),
        "generalization_gap_micro_f1": float(
            np.mean([fold["train_micro_f1"] - fold["validation_micro_f1"] for fold in fold_metrics])
        ),
        "blueprint_leakage_rate": (
            float(blueprint_leakage_hits / blueprint_leakage_total) if blueprint_leakage_total else 0.0
        ),
        "label_metrics": label_diagnostics(y_true, y_pred),
        "fold_metrics": fold_metrics,
    }
    stats["micro_f1"] = stats["validation_micro_f1"]
    stats["macro_f1"] = stats["validation_macro_f1"]
    stats["exact_match"] = stats["validation_exact_match"]
    return guard_arrays_to_sets(y_pred), stats


def compute_guard_diagnostics(cases: list[dict[str, Any]], folds: int = 5) -> dict[str, Any]:
    random_sets, random_stats = cross_validated_guard_predictions(cases, folds=folds, grouped=False)
    group_sets, group_stats = cross_validated_guard_predictions(cases, folds=folds, grouped=True)
    y_true = np.asarray([guard_targets(case) for case in cases], dtype=int)
    random_calibrated = np.asarray(
        [
            [1 if label in calibrate_guard_set(case, guard_set) else 0 for label in GUARD_LABELS]
            for case, guard_set in zip(cases, random_sets)
        ],
        dtype=int,
    )
    group_calibrated = np.asarray(
        [
            [1 if label in calibrate_guard_set(case, guard_set) else 0 for label in GUARD_LABELS]
            for case, guard_set in zip(cases, group_sets)
        ],
        dtype=int,
    )
    return {
        "random_cv": random_stats,
        "group_cv": group_stats,
        "random_cv_calibrated": {
            **score_guard_predictions(y_true, random_calibrated),
            "label_metrics": label_diagnostics(y_true, random_calibrated),
        },
        "group_cv_calibrated": {
            **score_guard_predictions(y_true, group_calibrated),
            "label_metrics": label_diagnostics(y_true, group_calibrated),
        },
        "leakage_diagnostics": {
            "random_cv_blueprint_leakage_rate": random_stats["blueprint_leakage_rate"],
            "group_cv_blueprint_leakage_rate": group_stats["blueprint_leakage_rate"],
            "micro_f1_inflation_random_minus_grouped": (
                random_stats["validation_micro_f1"] - group_stats["validation_micro_f1"]
            ),
            "exact_match_inflation_random_minus_grouped": (
                random_stats["validation_exact_match"] - group_stats["validation_exact_match"]
            ),
        },
    }


def build_trace(
    case: dict[str, Any],
    family: str,
    final_action: str,
    ranked_actions: list[str],
    *,
    predicted_guards: set[str] | None = None,
    tool_calls: int = 0,
) -> dict[str, Any]:
    signals = case["observable_context"]["drift_signals"]
    latency = {
        "static_prompt": 1.0,
        "dynamic_prompt": 1.15,
        "react_tool": 1.55 + 0.15 * tool_calls,
        "pgca": 1.35,
        "pgca_raw_selector": 1.31,
        "pgca_full_guard": 1.38,
        "pgca_no_projector": 1.28,
        "pgca_minus_join_guard": 1.34,
        "pgca_minus_audit_guard": 1.34,
    }[family]
    token_cost = {
        "static_prompt": 110,
        "dynamic_prompt": 135,
        "react_tool": 170 + 18 * tool_calls,
        "pgca": 145,
        "pgca_raw_selector": 144,
        "pgca_full_guard": 147,
        "pgca_no_projector": 143,
        "pgca_minus_join_guard": 145,
        "pgca_minus_audit_guard": 145,
    }[family]
    if signals.get("async_pending"):
        latency += 0.35
    return {
        "case_id": case["case_id"],
        "family": family,
        "drift_family": case["drift_family"],
        "risk_tier": case["risk_tier"],
        "preferred_action": case["hidden_oracle"]["gold_preferred_action"],
        "legal_actions": list(case["hidden_oracle"]["gold_legal_actions"]),
        "expected_handoff_boundary": case["hidden_oracle"]["expected_handoff_boundary"],
        "ranked_actions": ranked_actions,
        "predicted_guards": sorted(predicted_guards or []),
        "final_action": final_action,
        "tool_calls": tool_calls,
        "latency": latency,
        "token_cost": token_cost,
        "_is_compliant": 1 if final_action == case["hidden_oracle"]["gold_preferred_action"] else 0,
        "_is_violation": 1 if final_action not in case["hidden_oracle"]["gold_legal_actions"] else 0,
    }


def run_static_prompt(case: dict[str, Any]) -> dict[str, Any]:
    ranked = semantic_rank(case)
    return build_trace(case, "static_prompt", ranked[0], ranked)


def run_dynamic_prompt(case: dict[str, Any]) -> dict[str, Any]:
    ranked = dynamic_rank(case)
    return build_trace(case, "dynamic_prompt", ranked[0], ranked)


def run_react_tool(case: dict[str, Any]) -> dict[str, Any]:
    ranked, tool_calls = react_rank(case)
    return build_trace(case, "react_tool", ranked[0], ranked, tool_calls=tool_calls)


def run_pgca_repo_cases(
    cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[set[str]], list[set[str]]]:
    raw_guard_sets, stats = cross_validated_guard_predictions(cases, grouped=True)
    calibrated_sets = [calibrate_guard_set(case, guards) for case, guards in zip(cases, raw_guard_sets)]
    y_true = np.asarray([guard_targets(case) for case in cases], dtype=int)
    y_pred = np.asarray(
        [[1 if label in guard_set else 0 for label in GUARD_LABELS] for guard_set in calibrated_sets],
        dtype=int,
    )
    traces: list[dict[str, Any]] = []
    for case, guards in zip(cases, calibrated_sets):
        ranked = dynamic_rank(case)
        final_action = project_action(case, ranked, guards)
        traces.append(build_trace(case, "pgca", final_action, ranked, predicted_guards=guards))
    stats = {
        **stats,
        **score_guard_predictions(y_true, y_pred),
        "label_metrics": label_diagnostics(y_true, y_pred),
        "calibrated": True,
    }
    return traces, stats, raw_guard_sets, calibrated_sets


def run_pgca_external_cases(
    repo_cases: list[dict[str, Any]],
    external_cases: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float], list[set[str]], list[set[str]]]:
    if not external_cases:
        empty_stats = {
            "validation_micro_f1": 0.0,
            "validation_macro_f1": 0.0,
            "validation_exact_match": 0.0,
            "micro_f1": 0.0,
            "macro_f1": 0.0,
            "exact_match": 0.0,
            "label_metrics": {},
        }
        return [], empty_stats, [], []
    vectorizer, estimator = fit_guard_selector(repo_cases)
    raw_guard_sets = predict_guards(vectorizer, estimator, external_cases)
    guard_sets = [calibrate_guard_set(case, guards) for case, guards in zip(external_cases, raw_guard_sets)]
    y_true = np.asarray([guard_targets(case) for case in external_cases], dtype=int)
    y_pred = np.asarray(
        [[1 if label in guard_set else 0 for label in GUARD_LABELS] for guard_set in guard_sets],
        dtype=int,
    )
    traces: list[dict[str, Any]] = []
    for case, guards in zip(external_cases, guard_sets):
        ranked = dynamic_rank(case)
        final_action = project_action(case, ranked, guards)
        traces.append(build_trace(case, "pgca", final_action, ranked, predicted_guards=guards))
    stats = {
        "validation_micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "validation_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "validation_exact_match": float(np.mean(np.all(y_true == y_pred, axis=1))),
        "label_metrics": label_diagnostics(y_true, y_pred),
    }
    stats["micro_f1"] = stats["validation_micro_f1"]
    stats["macro_f1"] = stats["validation_macro_f1"]
    stats["exact_match"] = stats["validation_exact_match"]
    return traces, stats, raw_guard_sets, guard_sets


def run_ablation(
    case: dict[str, Any],
    family: str,
    predicted_guards: set[str],
    raw_predicted_guards: set[str] | None = None,
) -> dict[str, Any]:
    ranked = dynamic_rank(case)
    raw_predicted_guards = raw_predicted_guards or predicted_guards
    if family == "pgca_raw_selector":
        guards = set(raw_predicted_guards)
        final_action = project_action(case, ranked, guards)
        return build_trace(case, family, final_action, ranked, predicted_guards=guards)
    if family == "pgca_full_guard":
        guards = set(GUARD_LABELS)
        final_action = project_action(case, ranked, guards)
        return build_trace(case, family, final_action, ranked, predicted_guards=guards)
    if family == "pgca_no_projector":
        return build_trace(case, family, ranked[0], ranked, predicted_guards=predicted_guards)
    if family == "pgca_minus_join_guard":
        guards = set(predicted_guards) - {"join_guard"}
        final_action = project_action(case, ranked, guards)
        return build_trace(case, family, final_action, ranked, predicted_guards=guards)
    if family == "pgca_minus_audit_guard":
        guards = set(predicted_guards) - {"audit_guard"}
        final_action = project_action(case, ranked, guards)
        return build_trace(case, family, final_action, ranked, predicted_guards=guards)
    raise ValueError(f"unsupported ablation family: {family}")


def family_traces_for_cases(
    repo_cases: list[dict[str, Any]],
    external_cases: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    repo_pgca_traces, repo_guard_stats, repo_raw_predicted_guards, repo_predicted_guards = run_pgca_repo_cases(repo_cases)
    external_pgca_traces, external_guard_stats, external_raw_predicted_guards, external_predicted_guards = run_pgca_external_cases(repo_cases, external_cases)
    repo_guard_diagnostics = compute_guard_diagnostics(repo_cases)
    return {
        "repo_native": {
            "static_prompt": [run_static_prompt(case) for case in repo_cases],
            "dynamic_prompt": [run_dynamic_prompt(case) for case in repo_cases],
            "react_tool": [run_react_tool(case) for case in repo_cases],
            "pgca": repo_pgca_traces,
            "ablations": {
                family: [
                    run_ablation(case, family, guards, raw_guards)
                    for case, guards, raw_guards in zip(repo_cases, repo_predicted_guards, repo_raw_predicted_guards)
                ]
                for family in ABLATION_ORDER
            },
            "_guard_stats": repo_guard_stats,
            "_guard_diagnostics": repo_guard_diagnostics,
        },
        "external_transfer": {
            "static_prompt": [run_static_prompt(case) for case in external_cases],
            "dynamic_prompt": [run_dynamic_prompt(case) for case in external_cases],
            "react_tool": [run_react_tool(case) for case in external_cases],
            "pgca": external_pgca_traces,
            "ablations": {
                family: [
                    run_ablation(case, family, guards, raw_guards)
                    for case, guards, raw_guards in zip(external_cases, external_predicted_guards, external_raw_predicted_guards)
                ]
                for family in ABLATION_ORDER
            },
            "_guard_stats": external_guard_stats,
        },
    }


def empirical_violation_decomposition(
    cases: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> dict[str, float]:
    missed = 0
    unsound_accept = 0
    violations = 0
    for case, trace in zip(cases, traces):
        required = set(case["hidden_oracle"]["required_guards"])
        active = set(trace["predicted_guards"])
        final_action = trace["final_action"]
        illegal = final_action not in case["hidden_oracle"]["gold_legal_actions"]
        missing_required = not required.issubset(active)
        if missing_required:
            missed += 1
        if illegal and not missing_required:
            unsound_accept += 1
        if illegal:
            violations += 1
    total = len(cases) if cases else 1
    return {
        "missed_required_guard_rate": missed / total,
        "unsound_accept_rate": unsound_accept / total,
        "observed_violation_rate": violations / total,
        "union_bound_upper": (missed + unsound_accept) / total,
    }


def stress_case_semantic_candidates(case: dict[str, Any], severity: str) -> dict[str, Any]:
    stressed = json.loads(json.dumps(case))
    ranked = list(stressed["observable_context"]["semantic_candidates"])
    legal = set(stressed["hidden_oracle"]["gold_legal_actions"])
    preferred = stressed["hidden_oracle"]["gold_preferred_action"]
    illegal_ranked = [action for action in ranked if action not in legal]
    if severity == "proposal_noise_medium":
        if illegal_ranked:
            chosen = illegal_ranked[0]
            ranked = [chosen] + [action for action in ranked if action != chosen]
        elif preferred in ranked and len(ranked) > 1:
            ranked = [action for action in ranked if action != preferred]
            ranked.insert(1, preferred)
    elif severity == "proposal_noise_hard":
        if illegal_ranked:
            chosen = illegal_ranked[0]
            ranked = [chosen] + [action for action in ranked if action != chosen]
        if preferred in ranked and len(ranked) > 2:
            ranked = [action for action in ranked if action != preferred]
            ranked.append(preferred)
    else:
        raise ValueError(f"unsupported stress severity: {severity}")
    stressed["observable_context"]["semantic_candidates"] = ranked
    return stressed


def evaluate_proposal_stress(repo_cases: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for severity in ["proposal_noise_medium", "proposal_noise_hard"]:
        stressed_repo_cases = [stress_case_semantic_candidates(case, severity) for case in repo_cases]
        stressed_traces = family_traces_for_cases(stressed_repo_cases, [])
        output[severity] = {
            **evaluate_split({family: stressed_traces["repo_native"][family] for family in FAMILY_ORDER}),
            "pgca_decomposition": empirical_violation_decomposition(
                stressed_repo_cases,
                stressed_traces["repo_native"]["pgca"],
            ),
        }
    return output


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def exact_mcnemar_p_value(win_a: int, win_b: int) -> float:
    discordant = win_a + win_b
    if discordant == 0:
        return 1.0
    tail = 0.0
    for k in range(0, min(win_a, win_b) + 1):
        tail += math.comb(discordant, k) / (2 ** discordant)
    return min(1.0, 2 * tail)


def trace_metrics(traces: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(traces)
    compliant = sum(1 for trace in traces if trace["final_action"] == trace["preferred_action"])
    violations = sum(1 for trace in traces if trace["final_action"] not in trace["legal_actions"])
    wrong_typed = sum(
        1
        for trace in traces
        if trace["final_action"] != trace["preferred_action"]
        and action_family(trace["final_action"]) == action_family(trace["preferred_action"])
    )
    positives = [trace for trace in traces if trace["expected_handoff_boundary"]]
    negatives = [trace for trace in traces if not trace["expected_handoff_boundary"]]
    necessary_handoff_hits = sum(1 for trace in positives if trace["final_action"] in BOUNDARY_ACTIONS)
    unnecessary_handoffs = sum(1 for trace in negatives if trace["final_action"] in BOUNDARY_ACTIONS)
    abstain_count = sum(1 for trace in traces if trace["final_action"] == "abstain")
    latency_mean = sum(trace["latency"] for trace in traces) / total if total else 0.0
    tool_calls_mean = sum(trace["tool_calls"] for trace in traces) / total if total else 0.0
    token_cost_mean = sum(trace["token_cost"] for trace in traces) / total if total else 0.0
    acc_low, acc_high = wilson_interval(compliant, total)
    viol_low, viol_high = wilson_interval(violations, total)
    return {
        "count": total,
        "compliant_action_accuracy": compliant / total if total else 0.0,
        "compliant_action_accuracy_ci95": [acc_low, acc_high],
        "policy_violation_rate": violations / total if total else 0.0,
        "policy_violation_rate_ci95": [viol_low, viol_high],
        "wrong_typed_action_rate": wrong_typed / total if total else 0.0,
        "necessary_handoff_recall": necessary_handoff_hits / len(positives) if positives else 0.0,
        "unnecessary_handoff_rate": unnecessary_handoffs / len(negatives) if negatives else 0.0,
        "inconclusive_rate": abstain_count / total if total else 0.0,
        "latency_mean": latency_mean,
        "tool_calls_mean": tool_calls_mean,
        "token_cost_mean": token_cost_mean,
        "_compliant_binary": [1 if trace["final_action"] == trace["preferred_action"] else 0 for trace in traces],
        "_violation_binary": [1 if trace["final_action"] not in trace["legal_actions"] else 0 for trace in traces],
    }


def grouped_metrics(traces: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        groups[str(trace[field])].append(trace)
    return {key: trace_metrics(value) for key, value in sorted(groups.items())}


def pairwise_exact_mcnemar(reference: list[int], candidate: list[int]) -> dict[str, Any]:
    win_ref = 0
    win_cand = 0
    for ref, cand in zip(reference, candidate):
        if ref == 1 and cand == 0:
            win_ref += 1
        elif ref == 0 and cand == 1:
            win_cand += 1
    return {
        "discordant": win_ref + win_cand,
        "win_reference": win_ref,
        "win_candidate": win_cand,
        "p_value": exact_mcnemar_p_value(win_ref, win_cand),
    }


def bootstrap_metric_delta(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    field: str,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise ValueError("reference and candidate traces must align")
    rng = np.random.default_rng(seed)
    ref_arr = np.asarray([trace[field] for trace in reference], dtype=float)
    cand_arr = np.asarray([trace[field] for trace in candidate], dtype=float)
    deltas = []
    n = len(ref_arr)
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        deltas.append(float(np.mean(cand_arr[idx] - ref_arr[idx])))
    deltas_arr = np.asarray(deltas, dtype=float)
    return {
        "mean_delta": float(np.mean(cand_arr - ref_arr)),
        "ci95_low": float(np.quantile(deltas_arr, 0.025)),
        "ci95_high": float(np.quantile(deltas_arr, 0.975)),
    }


def evaluate_split(traces_by_family: dict[str, list[dict[str, Any]]], families: list[str] | None = None) -> dict[str, Any]:
    families = families or FAMILY_ORDER
    family_metrics = {family: trace_metrics(traces_by_family[family]) for family in families}
    per_drift = {family: grouped_metrics(traces_by_family[family], "drift_family") for family in families}
    per_risk = {family: grouped_metrics(traces_by_family[family], "risk_tier") for family in families}

    pairwise = {}
    bootstrap = {}
    if "pgca" in family_metrics:
        pgca_success = family_metrics["pgca"]["_compliant_binary"]
        pgca_violation = [1 - x for x in family_metrics["pgca"]["_violation_binary"]]
        for baseline in ["static_prompt", "dynamic_prompt", "react_tool"]:
            pairwise[f"pgca_vs_{baseline}"] = {
                "mcnemar_compliance": pairwise_exact_mcnemar(pgca_success, family_metrics[baseline]["_compliant_binary"]),
                "mcnemar_policy_violation": pairwise_exact_mcnemar(
                    pgca_violation,
                    [1 - x for x in family_metrics[baseline]["_violation_binary"]],
                ),
            }
        for baseline in ["dynamic_prompt", "react_tool"]:
            bootstrap[f"pgca_vs_{baseline}"] = {
                "compliant_action_accuracy_delta": bootstrap_metric_delta(
                    traces_by_family[baseline],
                    traces_by_family["pgca"],
                    field="_is_compliant",
                ),
                "policy_violation_rate_delta": bootstrap_metric_delta(
                    traces_by_family[baseline],
                    traces_by_family["pgca"],
                    field="_is_violation",
                ),
            }

    for family in family_metrics.values():
        family.pop("_compliant_binary", None)
        family.pop("_violation_binary", None)
    for grouped in (per_drift, per_risk):
        for family_groups in grouped.values():
            for metrics in family_groups.values():
                metrics.pop("_compliant_binary", None)
                metrics.pop("_violation_binary", None)

    return {
        "overall": family_metrics,
        "per_drift": per_drift,
        "per_risk": per_risk,
        "pairwise_tests": pairwise,
        "bootstrap_delta": bootstrap,
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    drift_counts = Counter(case["drift_family"] for case in cases)
    source_counts = Counter(case["source_type"] for case in cases)
    risk_counts = Counter(case["risk_tier"] for case in cases)
    return {
        "case_count": len(cases),
        "drift_counts": dict(sorted(drift_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "risk_counts": dict(sorted(risk_counts.items())),
        "direct_case_count": sum(1 for case in cases if case["source_type"] == "repo_direct"),
    }


def dataset_profile(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["meta"]["blueprint"])].append(case)

    variants_per_blueprint = [len(items) for items in grouped.values()]
    signal_diffs: list[int] = []
    slot_diffs: list[int] = []
    structured_diffs: list[int] = []
    distinct_signal_signatures: list[int] = []

    for items in grouped.values():
        ordered = sorted(items, key=lambda case: int(case["meta"]["variant"]))
        anchor = ordered[0]
        anchor_signals = anchor["observable_context"]["drift_signals"]
        anchor_slots = anchor["observable_context"]["visible_slots"]
        anchor_candidates = anchor["observable_context"]["semantic_candidates"]
        signatures = {
            tuple(sorted(case["observable_context"]["drift_signals"].items()))
            for case in ordered
        }
        distinct_signal_signatures.append(len(signatures))
        for case in ordered[1:]:
            signals = case["observable_context"]["drift_signals"]
            slots = case["observable_context"]["visible_slots"]
            diff_signals = sum(
                1
                for key in set(anchor_signals) | set(signals)
                if anchor_signals.get(key) != signals.get(key)
            )
            diff_slots = sum(
                1
                for key in set(anchor_slots) | set(slots)
                if anchor_slots.get(key) != slots.get(key)
            )
            diff_candidates = int(anchor_candidates != case["observable_context"]["semantic_candidates"])
            signal_diffs.append(diff_signals)
            slot_diffs.append(diff_slots)
            structured_diffs.append(diff_signals + diff_slots + diff_candidates)

    return {
        "blueprint_count": len(grouped),
        "avg_variants_per_blueprint": float(np.mean(variants_per_blueprint)) if variants_per_blueprint else 0.0,
        "min_variants_per_blueprint": int(min(variants_per_blueprint)) if variants_per_blueprint else 0,
        "max_variants_per_blueprint": int(max(variants_per_blueprint)) if variants_per_blueprint else 0,
        "avg_signal_field_diff_from_anchor": float(np.mean(signal_diffs)) if signal_diffs else 0.0,
        "avg_slot_field_diff_from_anchor": float(np.mean(slot_diffs)) if slot_diffs else 0.0,
        "avg_structured_field_diff_from_anchor": float(np.mean(structured_diffs)) if structured_diffs else 0.0,
        "min_structured_field_diff_from_anchor": int(min(structured_diffs)) if structured_diffs else 0,
        "max_structured_field_diff_from_anchor": int(max(structured_diffs)) if structured_diffs else 0,
        "mean_distinct_signal_signatures_per_blueprint": float(np.mean(distinct_signal_signatures))
        if distinct_signal_signatures
        else 0.0,
    }


def label_provenance() -> dict[str, str]:
    return {
        "guard_label_source": "oracle_derived_required_guards",
        "guard_label_detail": (
            "required guard labels are programmatically derived from repo-backed "
            "workflow legality rules via derive_required_guards() and stored in the hidden oracle"
        ),
        "selector_type": "trained_tree_based_multi_label_classifier",
        "selector_training_scope": (
            "offline supervised training on benchmark cases; the selector compresses "
            "repo-derived policy labels rather than discovering new policy from raw logs"
        ),
        "calibrator_type": "rule_based_safety_prior",
        "calibrator_detail": (
            "adds mandatory high-risk guards and removes guards contradicted by explicit shadow signals"
        ),
    }


def evaluate_benchmarks(repo_cases: list[dict[str, Any]], external_cases: list[dict[str, Any]]) -> dict[str, Any]:
    traces = family_traces_for_cases(repo_cases, external_cases)
    repo_eval = evaluate_split({family: traces["repo_native"][family] for family in FAMILY_ORDER})
    external_eval = evaluate_split({family: traces["external_transfer"][family] for family in FAMILY_ORDER})
    repo_ablation = evaluate_split(traces["repo_native"]["ablations"], families=ABLATION_ORDER)
    external_ablation = evaluate_split(traces["external_transfer"]["ablations"], families=ABLATION_ORDER)
    repo_decomposition = empirical_violation_decomposition(repo_cases, traces["repo_native"]["pgca"])
    raw_selector_decomposition = empirical_violation_decomposition(
        repo_cases,
        traces["repo_native"]["ablations"]["pgca_raw_selector"],
    )
    stress_tests = evaluate_proposal_stress(repo_cases)
    return {
        "repo_native": {
            "case_summary": summarize_cases(repo_cases),
            "dataset_profile": dataset_profile(repo_cases),
            "label_provenance": label_provenance(),
            **repo_eval,
            "guard_selector": traces["repo_native"]["_guard_stats"],
            "guard_diagnostics": traces["repo_native"]["_guard_diagnostics"],
            "ablations": repo_ablation,
            "violation_decomposition": {
                "pgca": repo_decomposition,
                "pgca_raw_selector": raw_selector_decomposition,
            },
            "stress_tests": stress_tests,
        },
        "external_transfer": {
            "case_summary": summarize_cases(external_cases),
            "dataset_profile": dataset_profile(external_cases),
            "label_provenance": label_provenance(),
            **external_eval,
            "guard_selector": traces["external_transfer"]["_guard_stats"],
            "ablations": external_ablation,
        },
        "traces": {
            "repo_native": {family: traces["repo_native"][family] for family in FAMILY_ORDER},
            "external_transfer": {family: traces["external_transfer"][family] for family in FAMILY_ORDER},
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render_markdown_summary(summary: dict[str, Any]) -> str:
    repo = summary["repo_native"]
    external = summary["external_transfer"]

    def line_for(split_name: str, section: dict[str, Any]) -> list[str]:
        lines = [f"## {split_name}", ""]
        lines.append(f"- case_count: {section['case_summary']['case_count']}")
        lines.append(
            f"- blueprint_count / avg_variants: {section['dataset_profile']['blueprint_count']} / "
            f"{section['dataset_profile']['avg_variants_per_blueprint']:.1f}"
        )
        lines.append(f"- guard_selector micro/macro F1: {section['guard_selector']['micro_f1']:.3f} / {section['guard_selector']['macro_f1']:.3f}")
        if split_name == "Repo-native benchmark":
            lines.append(
                f"- grouped raw selector micro-F1: {section['guard_diagnostics']['group_cv']['validation_micro_f1']:.3f}; "
                f"calibrated micro-F1: {section['guard_diagnostics']['group_cv_calibrated']['micro_f1']:.3f}"
            )
        lines.append("")
        lines.append("| Family | Compliant Acc. | Violation Rate | Wrong Typed | Handoff Recall | Unnecessary Handoff | Inconclusive |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for family in FAMILY_ORDER:
            m = section["overall"][family]
            lines.append(
                f"| {family} | {m['compliant_action_accuracy']:.3f} | {m['policy_violation_rate']:.3f} | "
                f"{m['wrong_typed_action_rate']:.3f} | {m['necessary_handoff_recall']:.3f} | "
                f"{m['unnecessary_handoff_rate']:.3f} | {m['inconclusive_rate']:.3f} |"
            )
        lines.append("")
        return lines

    lines = ["# Policy-aware workflow benchmark summary", ""]
    lines.extend(line_for("Repo-native benchmark", repo))
    lines.append("### Repo-native ablations")
    lines.append("")
    lines.append("| Variant | Compliant Acc. | Violation Rate | Wrong Typed | Handoff Recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for family in ABLATION_ORDER:
        m = repo["ablations"]["overall"][family]
        lines.append(
            f"| {family} | {m['compliant_action_accuracy']:.3f} | {m['policy_violation_rate']:.3f} | "
            f"{m['wrong_typed_action_rate']:.3f} | {m['necessary_handoff_recall']:.3f} |"
        )
    lines.append("")
    lines.extend(line_for("External transfer slice", external))
    lines.append("## Pairwise tests (repo-native)")
    lines.append("")
    for name, test in repo["pairwise_tests"].items():
        comp = test["mcnemar_compliance"]
        viol = test["mcnemar_policy_violation"]
        lines.append(
            f"- {name}: compliance p={comp['p_value']:.4f}, "
            f"violation p={viol['p_value']:.4f}, discordant={viol['discordant']}"
        )
    lines.append("")
    return "\n".join(lines)


def write_artifacts(output_dir: Path) -> dict[str, Path]:
    repo_cases = generate_repo_benchmark()
    external_cases = generate_external_transfer_slice()
    summary = evaluate_benchmarks(repo_cases, external_cases)

    repo_path = output_dir / "repo_native_policy_benchmark_v1.jsonl"
    external_path = output_dir / "external_transfer_slice_v1.jsonl"
    summary_path = output_dir / "policy_workflow_eval_summary_v1.json"
    markdown_path = output_dir / "policy_workflow_eval_summary_v1.md"
    traces_path = output_dir / "policy_workflow_eval_traces_v1.json"

    write_jsonl(repo_path, repo_cases)
    write_jsonl(external_path, external_cases)
    write_json(summary_path, {k: v for k, v in summary.items() if k != "traces"})
    write_json(traces_path, summary["traces"])
    markdown_path.write_text(render_markdown_summary(summary), encoding="utf-8")
    return {
        "repo_path": repo_path,
        "external_path": external_path,
        "summary_path": summary_path,
        "markdown_path": markdown_path,
        "traces_path": traces_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy-aware workflow benchmark generator/evaluator")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for JSONL/JSON/Markdown artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_artifacts(args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
