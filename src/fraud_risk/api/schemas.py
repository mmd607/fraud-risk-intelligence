"""Request/response contracts. Every field is explicit; unknown fields are rejected."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False, strict=True)
REFERENCE_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"


class TransactionFeatures(BaseModel):
    """Model input features. ``Time`` and ``Amount`` must be non-negative; all must be finite."""

    model_config = _STRICT

    Time: float = Field(ge=0, description="Seconds since the first transaction of the window.")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(ge=0, description="Transaction amount.")


class PredictionRequest(BaseModel):
    """Single prediction request."""

    model_config = _STRICT

    transaction: TransactionFeatures
    reference: str | None = Field(
        default=None,
        pattern=REFERENCE_PATTERN,
        description="Optional caller reference echoed in the response (not logged).",
    )


class BatchPredictionRequest(BaseModel):
    """Batch prediction request (size limited by FRAUD_RISK_API_MAX_BATCH_SIZE)."""

    model_config = _STRICT

    items: list[PredictionRequest] = Field(min_length=1)


class Explanation(BaseModel):
    """Optional local explanation (association, not causation)."""

    available: bool
    method: str | None = None
    output_space: str | None = None
    top_contributions: list[dict[str, float | str]] | None = None
    caveat: str | None = None
    reason: str | None = None


class PredictionResult(BaseModel):
    """Score, band, action, and reasons for one transaction."""

    reference: str | None
    score: float = Field(description="Model ranking score in [0, 1]; not a calibrated probability.")
    risk_band: str
    action: str = Field(description="Illustrative policy label; no action is executed.")
    reason_codes: list[str]
    explanation: Explanation | None = None


class PredictionResponse(BaseModel):
    """Response for a single prediction."""

    request_id: str
    model_version: str
    policy_version: str
    result: PredictionResult


class BatchPredictionResponse(BaseModel):
    """Response for a batch prediction."""

    request_id: str
    model_version: str
    policy_version: str
    results: list[PredictionResult]


class ModelInfoResponse(BaseModel):
    """Public model metadata."""

    model_id: str
    model_name: str
    model_sha256: str | None
    policy_version: str
    feature_names: list[str]


class HealthResponse(BaseModel):
    """Liveness: the process is up."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness: a model is loaded and can score."""

    status: str
    model_loaded: bool


class ErrorBody(BaseModel):
    """Stable error payload."""

    code: str
    message: str
    details: list[dict[str, str]] | None = None


class ErrorResponse(BaseModel):
    """Error envelope."""

    request_id: str
    error: ErrorBody
