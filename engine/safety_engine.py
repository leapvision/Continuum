from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ─── Load knowledge base once at import time ────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.dirname(__file__))
_KB_PATH  = os.path.join(_BASE_DIR, "data", "drug_knowledge.json")

with open(_KB_PATH) as f:
    _KB = json.load(f)


# ─── Enums & Data Classes ───────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MODERATE = "MODERATE"
    LOW      = "LOW"
    NONE     = "NONE"


class CheckType(str, Enum):
    ALLERGY          = "Allergy / Cross-Reactivity"
    DRUG_INTERACTION = "Drug-Drug Interaction"
    CONTRAINDICATION = "Drug-Disease Contraindication"
    RENAL_DOSE       = "Renal Dose Adjustment Required"


@dataclass
class Violation:
    check_type:       CheckType
    severity:         Severity
    detail:           str
    mechanism:        str
    adverse_effect:   str
    recommendation:   str
    evidence_source:  str
    action:           str          # "BLOCK" | "WARN"
    interacting_drug: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "type":             self.check_type.value,
            "severity":         self.severity.value,
            "detail":           self.detail,
            "mechanism":        self.mechanism,
            "adverse_effect":   self.adverse_effect,
            "recommendation":   self.recommendation,
            "evidence_source":  self.evidence_source,
            "action_required":  self.action,
        }
        if self.interacting_drug:
            d["interacting_with"] = self.interacting_drug
        return d


@dataclass
class SafetyResult:
    status:     str                      # "PASS" | "BLOCK"
    alert_level: Severity
    violations: list[Violation] = field(default_factory=list)
    warnings:   list[Violation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status":      self.status,
            "alert_level": self.alert_level.value,
            "blocks":      [v.to_dict() for v in self.violations],
            "warnings":    [v.to_dict() for v in self.warnings],
        }


# ─── Normalisation helpers ───────────────────────────────────────────────────

# Patient allergy records may use "SEVERE"; map to our canonical Severity enum
_SEVERITY_MAP = {
    "SEVERE":   "CRITICAL",
    "CRITICAL": "CRITICAL",
    "HIGH":     "HIGH",
    "MODERATE": "MODERATE",
    "LOW":      "LOW",
    "NONE":     "NONE",
}

def _map_severity(s: str) -> str:
    return _SEVERITY_MAP.get(s.upper(), "HIGH")


def _norm(s: str) -> str:
    """Lowercase + strip for fuzzy matching."""
    return s.lower().strip()


def _drug_matches(proposed: str, drug_list: list[str]) -> bool:
    """Check if proposed drug name appears in a list (case-insensitive)."""
    p = _norm(proposed)
    return any(p == _norm(d) for d in drug_list)


def _drug_in_patient(proposed: str, patient_meds: list[dict]) -> bool:
    p = _norm(proposed)
    return any(p == _norm(m["name"]) for m in patient_meds)


# ─── Individual Check Functions ─────────────────────────────────────────────

def check_allergies(
    proposed_drug: str,
    patient_allergies: list[dict],
) -> list[Violation]:
    """
    1. Direct allergy match (patient allergic to the proposed drug itself)
    2. Cross-reactivity match (patient allergic to a related substance)
    """
    violations: list[Violation] = []
    p_norm = _norm(proposed_drug)

    # 1 — Direct match
    for allergy in patient_allergies:
        if p_norm == _norm(allergy["substance"]):
            violations.append(Violation(
                check_type=CheckType.ALLERGY,
                severity=Severity[_map_severity(allergy.get("severity", "HIGH"))],
                detail=f"Patient has documented allergy to {proposed_drug} "
                       f"(reaction: {allergy.get('reaction', 'Unknown')}).",
                mechanism="Documented IgE-mediated or T-cell hypersensitivity reaction.",
                adverse_effect=allergy.get("reaction", "Hypersensitivity reaction"),
                recommendation="Do not prescribe. Select an alternative from a different drug class.",
                evidence_source="Patient_Allergy_Record",
                action="BLOCK",
            ))

    # 2 — Cross-reactivity
    for rule in _KB["allergy_cross_reactivity"]:
        allergen = rule["primary_allergen"]
        # Check if patient is allergic to this primary allergen
        patient_has_primary = any(
            _norm(a["substance"]) == _norm(allergen) for a in patient_allergies
        )
        if not patient_has_primary:
            continue

        # Check if proposed drug is a cross-reactive agent
        if _drug_matches(proposed_drug, rule.get("cross_reactive_drugs", [])):
            # Determine effective severity
            allergy_severity = next(
                (a.get("severity", "MODERATE") for a in patient_allergies
                 if _norm(a["substance"]) == _norm(allergen)),
                "MODERATE"
            )
            allergy_severity = _map_severity(allergy_severity)
            rule_severity    = _map_severity(rule["severity"])
            # Use the more severe of rule severity vs patient reaction severity
            effective_severity = rule_severity if allergy_severity == "CRITICAL" else "MODERATE"
            # Always BLOCK if cross-reactivity is 100% or patient had a severe reaction
            is_full_xreact = rule.get("cross_reactivity_rate", "").startswith("100")
            action = "BLOCK" if (allergy_severity in ("CRITICAL", "HIGH") or is_full_xreact) else "WARN"

            violations.append(Violation(
                check_type=CheckType.ALLERGY,
                severity=Severity[effective_severity],
                detail=(
                    f"Cross-reactivity risk: Patient allergic to {allergen}. "
                    f"{proposed_drug} shares structural/mechanistic similarity "
                    f"(cross-reactivity rate: {rule['cross_reactivity_rate']})."
                ),
                mechanism=rule["mechanism"],
                adverse_effect=f"Potential {allergy_severity.lower()} hypersensitivity reaction",
                recommendation=rule["note"],
                evidence_source=rule["evidence_source"],
                action=action,
                interacting_drug=allergen,
            ))

    return violations


def check_drug_interactions(
    proposed_drug: str,
    patient_medications: list[dict],
) -> list[Violation]:
    """
    Check proposed drug against each rule in drug_drug_interactions.
    Handles both directions (drug_a → drug_b and drug_b → drug_a).
    """
    violations: list[Violation] = []
    patient_med_names = [m["name"] for m in patient_medications]

    for rule in _KB["drug_drug_interactions"]:
        a_members = rule.get("drug_a_members", [])
        b_members = rule.get("drug_b_members", [])

        proposed_is_a = _drug_matches(proposed_drug, a_members)
        proposed_is_b = _drug_matches(proposed_drug, b_members)

        if not proposed_is_a and not proposed_is_b:
            continue

        # Find the "other side" in patient's current meds
        other_members = b_members if proposed_is_a else a_members

        for med in patient_med_names:
            if _drug_matches(med, other_members):
                violations.append(Violation(
                    check_type=CheckType.DRUG_INTERACTION,
                    severity=Severity[rule["severity"]],
                    detail=(
                        f"{proposed_drug} interacts with {med} "
                        f"(patient's current medication). "
                        f"Rule: {rule['id']}."
                    ),
                    mechanism=rule["mechanism"],
                    adverse_effect=rule["adverse_effect"],
                    recommendation=rule["recommendation"],
                    evidence_source=rule["evidence_source"],
                    action=rule["action"],
                    interacting_drug=med,
                ))

    return violations


def check_disease_contraindications(
    proposed_drug: str,
    patient_conditions: list[str],
    patient_renal: dict,
) -> list[Violation]:
    """
    Check proposed drug against disease-specific contraindications.
    Includes special renal eGFR threshold checking for renally-cleared drugs.
    """
    violations: list[Violation] = []
    conditions_norm = [_norm(c) for c in patient_conditions]

    for rule in _KB["drug_disease_contraindications"]:
        drugs = rule.get("drug", []) + rule.get("drug_class", [])
        if not _drug_matches(proposed_drug, drugs):
            continue

        condition = rule["condition"]

        # Special case: renal-threshold rules
        if "egfr_threshold" in rule:
            egfr = patient_renal.get("egfr", 999)
            if egfr < rule["egfr_threshold"]:
                violations.append(Violation(
                    check_type=CheckType.RENAL_DOSE,
                    severity=Severity[rule["severity"]],
                    detail=(
                        f"{proposed_drug} contraindicated: patient eGFR is {egfr} mL/min/1.73m² "
                        f"(threshold: {rule['egfr_threshold']}). "
                        f"Renal status: {patient_renal.get('status', 'unknown')}."
                    ),
                    mechanism=rule["mechanism"],
                    adverse_effect=f"Risk in renal impairment: {condition}",
                    recommendation=rule["recommendation"],
                    evidence_source=rule["evidence_source"],
                    action=rule["action"],
                ))
            continue

        # General condition match
        if _norm(condition) in conditions_norm:
            violations.append(Violation(
                check_type=CheckType.CONTRAINDICATION,
                severity=Severity[rule["severity"]],
                detail=(
                    f"{proposed_drug} has a documented contraindication with "
                    f"'{condition}' (present in patient's active condition list)."
                ),
                mechanism=rule["mechanism"],
                adverse_effect=rule["mechanism"],
                recommendation=rule["recommendation"],
                evidence_source=rule["evidence_source"],
                action=rule["action"],
            ))

    return violations


# ─── Main Orchestrator ───────────────────────────────────────────────────────

def run_safety_check(
    proposed_drug: str,
    proposed_dosage: str,
    indication: str,
    patient: dict,
) -> SafetyResult:
    patient_allergies   = patient.get("allergies", [])
    patient_meds        = patient.get("current_medications", [])
    patient_conditions  = patient.get("conditions", [])
    patient_renal       = patient.get("renal_function", {"egfr": 999, "status": "NORMAL"})

    all_violations: list[Violation] = []

    # ── Run all checks ──────────────────────────────────────────────────────
    all_violations += check_allergies(proposed_drug, patient_allergies)
    all_violations += check_drug_interactions(proposed_drug, patient_meds)
    all_violations += check_disease_contraindications(proposed_drug, patient_conditions, patient_renal)

    # ── Separate BLOCKs vs WARNs ────────────────────────────────────────────
    blocks   = [v for v in all_violations if v.action == "BLOCK"]
    warnings = [v for v in all_violations if v.action == "WARN"]

    # ── Determine overall status and alert level ─────────────────────────────
    _severity_rank = {
        Severity.CRITICAL: 4,
        Severity.HIGH:     3,
        Severity.MODERATE: 2,
        Severity.LOW:      1,
        Severity.NONE:     0,
    }

    if blocks:
        status = "BLOCK"
        worst = max(blocks, key=lambda v: _severity_rank[v.severity])
        alert_level = worst.severity
    elif warnings:
        status = "PASS"            # can proceed with caution
        worst = max(warnings, key=lambda v: _severity_rank[v.severity])
        alert_level = worst.severity
    else:
        status = "PASS"
        alert_level = Severity.NONE

    return SafetyResult(
        status=status,
        alert_level=alert_level,
        violations=blocks,
        warnings=warnings,
    )