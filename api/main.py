from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from engine.patient_store import get_patient_context, list_patient_ids
from engine.safety_engine  import run_safety_check, SafetyResult, Severity

app = FastAPI(
    title="Prescription Safety Interceptor",
    description=(
        "Continuum Module 4 — Safety Engine. "
        "Intercepts every medication order and validates against "
        "allergies, drug-drug interactions, and disease contraindications."
    ),
    version="1.0.0",
)

class ProposedOrder(BaseModel):
    medication: str  = Field(..., example="Clarithromycin")
    dosage:     str  = Field(..., example="500mg BID")
    indication: str  = Field(..., example="Pneumonia")


class SafetyCheckRequest(BaseModel):
    safety_task_id: str        = Field(default_factory=lambda: f"safe_{uuid.uuid4().hex[:8]}")
    patient_id:     str        = Field(..., example="pt_55902")
    proposed_order: ProposedOrder


class InlinePatientContext(BaseModel):
    """For integration with other modules that forward patient context directly."""
    allergies: list[dict]           = Field(default_factory=list)
    current_medications: list[dict] = Field(default_factory=list)
    conditions: list[str]           = Field(default_factory=list)
    renal_function: dict            = Field(default={"egfr": 999, "status": "NORMAL"})
    hepatic_function: dict          = Field(default={"status": "NORMAL"})
    gene_variants: list[str]        = Field(default_factory=list)
    age: Optional[int]              = None
    weight_kg: Optional[float]      = None


class SafetyCheckWithContextRequest(BaseModel):
    safety_task_id: str        = Field(default_factory=lambda: f"safe_{uuid.uuid4().hex[:8]}")
    patient_id:     str        = Field(..., example="pt_55902")
    proposed_order: ProposedOrder
    patient_context: InlinePatientContext


def _build_response(
    safety_task_id: str,
    patient_id: str,
    proposed_order: ProposedOrder,
    result: SafetyResult,
    patient_context: dict,
    elapsed_ms: int,
) -> dict:
    """Assemble the unified response envelope."""
    return {
        "safety_task_id":  safety_task_id,
        "patient_id":      patient_id,
        "proposed_order":  proposed_order.model_dump(),
        "reasoning_engine": "DrugBank_KnowledgeBase_v5.1",
        "processing_time_ms": elapsed_ms,
        "patient_context_used": {
            "allergies":           patient_context.get("allergies", []),
            "current_medications": patient_context.get("current_medications", []),
            "conditions":          patient_context.get("conditions", []),
            "renal_function":      patient_context.get("renal_function"),
        },
        "safety_result": result.to_dict(),
    }


@app.get("/health")
def health():
    return {"status": "ok", "service": "prescription-safety-interceptor"}


@app.get("/patients")
def list_patients():
    """List all patient IDs available in the mock database."""
    return {"patient_ids": list_patient_ids()}


@app.get("/patient/{patient_id}/context")
def get_context(patient_id: str):
    """
    Inspect the safety-relevant context that will be pulled for a given patient.
    Useful for other modules to verify before calling /safety/check.
    """
    ctx = get_patient_context(patient_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return ctx


@app.post("/safety/check")
def safety_check(req: SafetyCheckRequest):
    # Pull patient context
    patient_ctx = get_patient_context(req.patient_id)
    if not patient_ctx:
        raise HTTPException(status_code=404, detail=f"Patient '{req.patient_id}' not found.")

    start = time.time()
    result = run_safety_check(
        proposed_drug=req.proposed_order.medication,
        proposed_dosage=req.proposed_order.dosage,
        indication=req.proposed_order.indication,
        patient=patient_ctx,
    )
    elapsed_ms = int((time.time() - start) * 1000)

    return _build_response(
        req.safety_task_id,
        req.patient_id,
        req.proposed_order,
        result,
        patient_ctx,
        elapsed_ms,
    )


@app.post("/safety/check-with-context")
def safety_check_with_context(req: SafetyCheckWithContextRequest):
    ctx_dict = req.patient_context.model_dump()
    ctx_dict["patient_id"] = req.patient_id

    start = time.time()
    result = run_safety_check(
        proposed_drug=req.proposed_order.medication,
        proposed_dosage=req.proposed_order.dosage,
        indication=req.proposed_order.indication,
        patient=ctx_dict,
    )
    elapsed_ms = int((time.time() - start) * 1000)

    return _build_response(
        req.safety_task_id,
        req.patient_id,
        req.proposed_order,
        result,
        ctx_dict,
        elapsed_ms,
    )
