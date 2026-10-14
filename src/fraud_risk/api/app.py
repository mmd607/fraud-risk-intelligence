"""FastAPI application factory.

Run locally::

    uvicorn fraud_risk.api.app:create_app --factory --host 127.0.0.1 --port 8000

Endpoints: ``GET /health`` (liveness), ``GET /ready`` (model loaded), ``GET /v1/model``,
``POST /v1/predict``, ``POST /v1/predict/batch``. OpenAPI docs live at ``/docs``.

Guarantees: strict input validation; controlled JSON errors with no stack traces or echoed
inputs; a request id on every response; request logs with method/path/status/latency only, never
payloads; optional ``X-API-Key`` protection of ``/v1/*`` when ``FRAUD_RISK_API_KEY`` is set. The
service makes no payments and has no external side effects.
"""

from __future__ import annotations

import re
import secrets
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, Header, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fraud_risk import __version__
from fraud_risk.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorBody,
    ErrorResponse,
    Explanation,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
    PredictionResult,
    ReadinessResponse,
)
from fraud_risk.config.settings import Settings, get_settings
from fraud_risk.core.exceptions import FraudRiskError
from fraud_risk.core.logging import configure_logging, get_logger, log_event
from fraud_risk.data.schema import FEATURE_COLUMNS
from fraud_risk.services.inference import (
    InferenceService,
    ModelNotReadyError,
    ScoredTransaction,
    build_model_service,
)

_LOGGER = get_logger("api")
REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class ApiError(FraudRiskError):
    """Error with an HTTP status code."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message, code=code)
        self.status_code = status_code


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else "unknown"


def _error(request: Request, status: int, code: str, message: str, details: list[dict[str, str]] | None = None) -> JSONResponse:  # fmt: skip  # noqa: E501
    body = ErrorResponse(
        request_id=_request_id(request),
        error=ErrorBody(code=code, message=message, details=details),
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(),
        headers={REQUEST_ID_HEADER: _request_id(request)},
    )


def _install_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, error: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in item.get("loc", ())),
                "message": str(item.get("msg", "invalid")),
            }
            for item in error.errors()
        ]  # deliberately excludes the offending input values
        return _error(request, 422, "REQUEST_INVALID", "Request validation failed.", details)

    @app.exception_handler(ModelNotReadyError)
    async def _not_ready(request: Request, _error_: ModelNotReadyError) -> JSONResponse:
        return _error(request, 503, "MODEL_NOT_READY", "No model is loaded.")

    @app.exception_handler(ApiError)
    async def _api(request: Request, error: ApiError) -> JSONResponse:
        return _error(request, error.status_code, error.code, error.message)

    @app.exception_handler(FraudRiskError)
    async def _domain(request: Request, error: FraudRiskError) -> JSONResponse:
        log_event(_LOGGER, "domain_error", level=40, code=error.code)
        return _error(request, 500, error.code, "The request could not be processed.")

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, error: Exception) -> JSONResponse:
        log_event(_LOGGER, "unhandled_error", level=40, exception_type=type(error).__name__)
        return _error(request, 500, "INTERNAL_ERROR", "Internal server error.")


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def _request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _REQUEST_ID_PATTERN.fullmatch(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        start = perf_counter()
        try:
            response = await call_next(request)
        except Exception as error:  # noqa: BLE001 - last-resort guard, logged without details
            log_event(_LOGGER, "unhandled_error", level=40, exception_type=type(error).__name__)
            response = _error(request, 500, "INTERNAL_ERROR", "Internal server error.")
        response.headers[REQUEST_ID_HEADER] = request_id
        log_event(
            _LOGGER,
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round((perf_counter() - start) * 1000, 2),
        )
        return response


def _to_frame(items: list[PredictionRequest]) -> pd.DataFrame:
    rows = [item.transaction.model_dump() for item in items]
    return pd.DataFrame(rows, columns=list(FEATURE_COLUMNS), dtype="float64")


def _to_result(scored: ScoredTransaction) -> PredictionResult:
    decision = scored.decision
    return PredictionResult(
        reference=decision.reference,
        score=decision.score if decision.score is not None else float("nan"),
        risk_band=decision.band.value,
        action=decision.action.value,
        reason_codes=[code.value for code in decision.reason_codes],
        explanation=Explanation(**scored.explanation) if scored.explanation is not None else None,  # type: ignore[arg-type]
    )


def get_service(request: Request) -> InferenceService:
    """Dependency: the loaded inference service, or 503 when no model is ready."""
    current: InferenceService | None = request.app.state.service
    if current is None or not current.ready:
        raise ModelNotReadyError("No model is loaded.")
    return current


def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Dependency: enforce ``X-API-Key`` when ``FRAUD_RISK_API_KEY`` is configured."""
    settings: Settings = request.app.state.settings
    if settings.api_key is None:
        return
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode(), settings.api_key.get_secret_value().encode()
    ):
        raise ApiError(401, "UNAUTHORIZED", "Missing or invalid API key.")


ServiceDep = Annotated[InferenceService, Depends(get_service)]
AuthDep = Annotated[None, Depends(require_api_key)]


def create_app(
    service: InferenceService | None = None, settings: Settings | None = None
) -> FastAPI:
    """Build the app. Without ``service``, one is built from settings at startup."""
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(
            resolved_settings.log_level, json_format=resolved_settings.log_format == "json"
        )
        if getattr(app.state, "service", None) is None:
            app.state.service = build_model_service(resolved_settings)
        yield

    app = FastAPI(
        title="Fraud & Risk Intelligence API",
        version=__version__,
        description=(
            "Educational fraud-risk scoring service. Actions are illustrative labels; no payment "
            "is processed and no external action is taken. Not for real banking decisions."
        ),
        lifespan=lifespan,
        responses={
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    app.state.service = service
    app.state.settings = resolved_settings
    _install_handlers(app)
    _install_middleware(app)

    @app.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        """Liveness: the process is running (does not check the model)."""
        return HealthResponse(status="ok")

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
        tags=["operations"],
    )
    def ready(request: Request) -> JSONResponse:
        """Readiness: 200 when a model is loaded, otherwise 503."""
        current: InferenceService | None = request.app.state.service
        loaded = bool(current is not None and current.ready)
        body = ReadinessResponse(status="ready" if loaded else "not_ready", model_loaded=loaded)
        return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())

    @app.get("/v1/model", response_model=ModelInfoResponse, tags=["inference"])
    def model_info(service: ServiceDep, _auth: AuthDep) -> ModelInfoResponse:
        """Public metadata of the loaded model (no filesystem paths)."""
        description = service.describe()
        return ModelInfoResponse(
            model_id=description.model_id,
            model_name=description.model_name,
            model_sha256=description.model_sha256,
            policy_version=description.policy_version,
            feature_names=list(description.feature_names),
        )

    @app.post("/v1/predict", response_model=PredictionResponse, tags=["inference"])
    def predict(
        payload: PredictionRequest,
        request: Request,
        service: ServiceDep,
        _auth: AuthDep,
        explain: Annotated[bool, Query(description="Include a local explanation.")] = False,
    ) -> PredictionResponse:
        """Score one transaction and apply the (illustrative) risk policy."""
        scored = service.predict(
            _to_frame([payload]), references=[payload.reference], explain=explain
        )
        description = service.describe()
        return PredictionResponse(
            request_id=_request_id(request),
            model_version=description.model_id,
            policy_version=description.policy_version,
            result=_to_result(scored[0]),
        )

    @app.post("/v1/predict/batch", response_model=BatchPredictionResponse, tags=["inference"])
    def predict_batch(
        payload: BatchPredictionRequest, request: Request, service: ServiceDep, _auth: AuthDep
    ) -> BatchPredictionResponse:
        """Score a batch (explanations are not available in batch mode)."""
        if len(payload.items) > resolved_settings.api_max_batch_size:
            raise ApiError(
                413,
                "BATCH_TOO_LARGE",
                f"Batch size exceeds the limit of {resolved_settings.api_max_batch_size}.",
            )
        scored = service.predict(
            _to_frame(payload.items), references=[item.reference for item in payload.items]
        )
        description = service.describe()
        return BatchPredictionResponse(
            request_id=_request_id(request),
            model_version=description.model_id,
            policy_version=description.policy_version,
            results=[_to_result(item) for item in scored],
        )

    return app
