"""Pydantic request/response schemas for the TrustLayer API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class CreateInvestigation(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


class InvestigationOut(BaseModel):
    id: str
    name: str
    created_at: str
    status: str
    assessment: Optional[str] = None
    confidence: Optional[str] = None
    confidence_score: Optional[float] = None
    coverage: Optional[float] = None
    risk_score: Optional[float] = None
    summary: Optional[str] = None


class EvidenceOut(BaseModel):
    id: str
    investigation_id: str
    filename: str
    modality: str
    content_type: Optional[str] = None
    size_bytes: int = 0
    uploaded_at: str
    individual_score: Optional[float] = None
    individual_label: Optional[str] = None


class GenericResponse(BaseModel):
    ok: bool = True
    message: str = ""
    data: Optional[Any] = None
