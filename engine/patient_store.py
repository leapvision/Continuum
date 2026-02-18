from __future__ import annotations
import json
import os
from typing import Optional

_BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
_PATIENTS_PATH = os.path.join(_BASE_DIR, "data", "patients.json")

with open(_PATIENTS_PATH) as f:
    _PATIENTS: dict = json.load(f)


def get_patient(patient_id: str) -> Optional[dict]:
    """
    Fetch full patient record by ID.
    Returns None if not found.
    """
    return _PATIENTS.get(patient_id)


def get_patient_context(patient_id: str) -> Optional[dict]:
    patient = get_patient(patient_id)
    if not patient:
        return None

    return {
        "patient_id":           patient["patient_id"],
        "age":                  patient.get("age"),
        "weight_kg":            patient.get("weight_kg"),
        "allergies":            patient.get("allergies", []),
        "current_medications":  patient.get("current_medications", []),
        "conditions":           patient.get("conditions", []),
        "renal_function":       patient.get("renal_function", {"egfr": 999, "status": "NORMAL"}),
        "hepatic_function":     patient.get("hepatic_function", {"status": "NORMAL"}),
        "gene_variants":        patient.get("gene_variants", []),
    }


def list_patient_ids() -> list[str]:
    return list(_PATIENTS.keys())