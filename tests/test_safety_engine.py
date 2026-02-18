"""
tests/test_safety_engine.py
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.safety_engine import run_safety_check, Severity
from engine.patient_store  import get_patient_context


# ─── Fixtures ────────────────────────────────────────────────────────────────

def patient(pid):
    ctx = get_patient_context(pid)
    assert ctx is not None, f"Patient {pid} not in mock DB"
    return ctx


# ─── Drug-Drug Interaction Tests ─────────────────────────────────────────────

class TestDrugDrugInteractions:

    def test_clarithromycin_simvastatin_BLOCKED(self):
        """DDI_001: Macrolide + Statin → CYP3A4 → Rhabdomyolysis"""
        result = run_safety_check(
            proposed_drug="Clarithromycin",
            proposed_dosage="500mg BID",
            indication="Pneumonia",
            patient=patient("pt_55902"),   # has Simvastatin
        )
        assert result.status == "BLOCK"
        assert result.alert_level == Severity.CRITICAL
        block_types = [v.check_type.value for v in result.violations]
        assert "Drug-Drug Interaction" in block_types

    def test_warfarin_amiodarone_WARNING(self):
        """DDI_004: Warfarin + Amiodarone → elevated INR (WARN, not BLOCK)"""
        # Patient pt_88301 already has both - test what happens if we propose
        # a QT-prolonging antibiotic on top. But let's isolate DDI_004:
        pt = {
            "patient_id": "test_pt",
            "allergies": [],
            "current_medications": [{"name": "Amiodarone", "dosage": "200mg QD", "indication": "AF"}],
            "conditions": ["Atrial Fibrillation"],
            "renal_function": {"egfr": 80, "status": "NORMAL"},
        }
        result = run_safety_check("Warfarin", "5mg QD", "AF", pt)
        assert result.status == "PASS"           # WARN doesn't block
        assert len(result.warnings) >= 1
        warn_types = [v.check_type.value for v in result.warnings]
        assert "Drug-Drug Interaction" in warn_types

    def test_maoi_ssri_BLOCKED(self):
        """DDI_005: MAOI + SSRI → Serotonin Syndrome"""
        result = run_safety_check(
            proposed_drug="Sertraline",
            proposed_dosage="50mg QD",
            indication="Depression",
            patient=patient("pt_30019"),   # has Phenelzine (MAOI)
        )
        assert result.status == "BLOCK"
        assert result.alert_level == Severity.CRITICAL

    def test_lithium_ibuprofen_BLOCKED(self):
        """DDI_007: Lithium + NSAID → Lithium toxicity"""
        result = run_safety_check(
            proposed_drug="Ibuprofen",
            proposed_dosage="400mg TID",
            indication="Pain",
            patient=patient("pt_50121"),   # has Lithium
        )
        assert result.status == "BLOCK"
        assert result.alert_level in (Severity.HIGH, Severity.CRITICAL)

    def test_safe_combination_PASSES(self):
        """Amoxicillin on a clean patient should PASS"""
        pt = {
            "patient_id": "test_clean",
            "allergies": [],
            "current_medications": [{"name": "Metformin", "dosage": "1000mg BID", "indication": "DM"}],
            "conditions": ["Type 2 Diabetes"],
            "renal_function": {"egfr": 85, "status": "NORMAL"},
        }
        result = run_safety_check("Amoxicillin", "500mg TID", "Sinusitis", pt)
        assert result.status == "PASS"
        assert result.alert_level == Severity.NONE
        assert len(result.violations) == 0


# ─── Allergy Tests ───────────────────────────────────────────────────────────

class TestAllergies:

    def test_direct_allergy_BLOCKED(self):
        """Patient allergic to Penicillin — prescribe Penicillin directly"""
        pt = {
            "patient_id": "test_allergy_direct",
            "allergies": [{"substance": "Penicillin", "reaction": "Anaphylaxis", "severity": "SEVERE"}],
            "current_medications": [],
            "conditions": [],
            "renal_function": {"egfr": 90, "status": "NORMAL"},
        }
        result = run_safety_check("Penicillin", "500mg QID", "Tonsillitis", pt)
        assert result.status == "BLOCK"
        assert any(v.check_type.value == "Allergy / Cross-Reactivity" for v in result.violations)

    def test_penicillin_amoxicillin_cross_reactivity_BLOCKED(self):
        """Patient allergic to Penicillin, prescribe Amoxicillin (100% cross-reactive)"""
        result = run_safety_check(
            proposed_drug="Amoxicillin",
            proposed_dosage="875mg BID",
            indication="Sinusitis",
            patient=patient("pt_55902"),   # has Penicillin allergy (SEVERE → Anaphylaxis)
        )
        assert result.status == "BLOCK"
        block_types = [v.check_type.value for v in result.violations]
        assert "Allergy / Cross-Reactivity" in block_types

    def test_nsaid_cross_reactivity_BLOCKED(self):
        """Patient has NSAID allergy, prescribe Ibuprofen"""
        result = run_safety_check(
            proposed_drug="Ibuprofen",
            proposed_dosage="400mg TID",
            indication="Pain",
            patient=patient("pt_12045"),   # has NSAIDs allergy (GI Bleeding, SEVERE)
        )
        assert result.status == "BLOCK"

    def test_sulfonamide_cross_reactivity(self):
        """Patient allergic to Sulfonamides, prescribe TMP-SMX"""
        result = run_safety_check(
            proposed_drug="Sulfamethoxazole",
            proposed_dosage="400mg BID",
            indication="UTI",
            patient=patient("pt_55902"),   # has Sulfonamides allergy
        )
        assert result.status == "BLOCK"


# ─── Contraindication / Renal Tests ──────────────────────────────────────────

class TestContraindications:

    def test_metformin_renal_failure_BLOCKED(self):
        """Metformin in eGFR < 30 → lactic acidosis risk"""
        pt = {
            "patient_id": "test_renal",
            "allergies": [],
            "current_medications": [],
            "conditions": ["Type 2 Diabetes"],
            "renal_function": {"egfr": 22, "status": "SEVERE_IMPAIRMENT"},
        }
        result = run_safety_check("Metformin", "1000mg BID", "T2DM", pt)
        assert result.status == "BLOCK"
        assert any(v.check_type.value == "Renal Dose Adjustment Required" for v in result.violations)

    def test_metformin_normal_renal_PASSES(self):
        """Metformin in normal renal function → fine"""
        result = run_safety_check(
            proposed_drug="Metformin",
            proposed_dosage="500mg BID",
            indication="T2DM",
            patient=patient("pt_12045"),   # eGFR 85
        )
        assert result.status == "PASS"


# ─── Multi-violation Tests ───────────────────────────────────────────────────

class TestMultipleViolations:

    def test_multiple_violations_uses_worst_severity(self):
        """
        Patient has Warfarin + Amiodarone. Prescribe Ibuprofen:
          - DDI with Warfarin (CRITICAL BLOCK)
          - DDI with Amiodarone (implicitly through QT? No - but Warfarin+NSAID is CRITICAL)
        """
        pt = {
            "patient_id": "test_multi",
            "allergies": [{"substance": "NSAIDs", "reaction": "GI Bleed", "severity": "SEVERE"}],
            "current_medications": [
                {"name": "Warfarin", "dosage": "5mg QD", "indication": "AF"},
                {"name": "Amiodarone", "dosage": "200mg QD", "indication": "AF"},
            ],
            "conditions": ["Atrial Fibrillation"],
            "renal_function": {"egfr": 50, "status": "MILD_IMPAIRMENT"},
        }
        result = run_safety_check("Ibuprofen", "400mg TID", "Pain", pt)
        assert result.status == "BLOCK"
        assert result.alert_level == Severity.CRITICAL
        # Should have both allergy AND DDI blocks
        block_types = {v.check_type.value for v in result.violations}
        assert "Allergy / Cross-Reactivity" in block_types
        assert "Drug-Drug Interaction" in block_types


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])