import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    FastAPI,
    Form,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend import store
from backend.cold_start import ColdStartConflict, ColdStartService, imported_labels
from backend.config import Settings
from backend.data import MAX_CSV_BYTES, CSVPreview, merchant_key, preview_csv
from backend.evidence import Evidence
from backend.meta_webhook import MetaWebhook
from backend.meta_webhook import router as meta_router
from backend.reader_serving import ReaderUnavailable
from backend.schemas import (
    Answer,
    AnswerResult,
    ChatRequest,
    ChatResult,
    ColdStartRequest,
    ColdStartState,
    Decision,
    Error,
    ExampleEdit,
    ImportResult,
    Item,
    MetricsResult,
    ModelVersion,
    Page,
    PredictRequest,
    Reasons,
    UndoAnswer,
    WorkspaceId,
)
from backend.workspace import WorkspaceExists, WorkspaceService


class Health(BaseModel):
    status: str = "ok"
    database: str = "ready"


def create_app(
    settings: Settings | None = None,
    *,
    workspace_service: WorkspaceService | None = None,
    cold_start_service: ColdStartService | None = None,
) -> FastAPI:
    config = settings or Settings()
    evidence = Evidence(
        config.data_dir,
        config.evaluation_dir,
        workspace_lock=lambda ws: app.state.workspaces._lock(ws),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = workspace_service or await run_in_threadpool(
            lambda: WorkspaceService(store.Store(config.database_path), config.data_dir)
        )
        app.state.workspaces = service
        service.cold_start = cold_start_service or ColdStartService(service.store, config)
        app.state.meta_webhook = MetaWebhook(config, service)
        try:
            yield
        finally:
            await run_in_threadpool(service.close)

    app = FastAPI(title="Secnd", version="0.1.0", lifespan=lifespan)
    app.include_router(meta_router(config))

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, error: StarletteHTTPException):
        if request.url.path.startswith("/api/v1/"):
            return JSONResponse(
                status_code=error.status_code,
                content=Error(
                    code=f"http_{error.status_code}", message=str(error.detail)
                ).model_dump(),
                headers=error.headers,
            )
        return await http_exception_handler(request, error)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError):
        if request.url.path.startswith("/api/v1/"):
            return JSONResponse(
                status_code=422,
                content=Error(
                    code="validation_error", message="Request fields do not match the API schema."
                ).model_dump(),
            )
        return await request_validation_exception_handler(request, error)

    @app.get("/api/health", response_model=Health)
    def health() -> Health:
        try:
            store.check_connection(config.database_path)
        except sqlite3.Error as error:
            raise HTTPException(status_code=503, detail="Local storage is unavailable.") from error
        return Health()

    @app.post("/api/preview", response_model=CSVPreview)
    async def preview(file: Annotated[UploadFile, "UTF-8 labelled CSV"]) -> CSVPreview:
        try:
            content = await file.read(MAX_CSV_BYTES + 1)
            if len(content) > MAX_CSV_BYTES:
                raise HTTPException(status_code=413, detail="Choose a CSV smaller than 5 MiB.")
            return await run_in_threadpool(preview_csv, content, file.filename or "decisions.csv")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        finally:
            await file.close()

    def fixture_mode(response: Response):
        response.headers["X-Secnd-Mode"] = "fixture"

    contract = APIRouter(
        prefix="/api/v1/workspaces/{ws}",
        tags=["Workspace"],
        responses={status: {"model": Error} for status in (422, 413, 404, 409, 503)},
        dependencies=[Depends(fixture_mode)],
    )
    contract_note = (
        "Contract fixture only: no model runs and nothing is saved. "
        "Responses carry X-Secnd-Mode: fixture. Metrics are unmeasured."
    )

    def decision(item: Item, *, auto: bool = False) -> Decision:
        probability = 1.0 if auto else 0.55
        return Decision(
            item_id=item.item_id,
            label="Travel",
            probabilities={"Travel": probability, "Software": 1 - probability},
            prob=probability,
            auto=auto,
            threshold=0.99 if auto else None,
            model_version="fixture-v0",
            reasons=Reasons(),
            item=item,
        )

    def page[T](items: list[T], cursor: str | None, limit: int) -> Page[T]:
        if cursor not in (None, "fixture-end"):
            raise HTTPException(status_code=422, detail="Unknown fixture cursor.")
        if cursor:
            items = []
        return Page(items=items[:limit], next_cursor=None)

    @contract.post(
        "/decisions:import",
        response_model=ImportResult,
        description="Persist labels with a fixed calibration holdout and fit a first model.",
    )
    async def import_decisions(
        ws: WorkspaceId,
        file: UploadFile,
        response: Response,
        target_column: Annotated[str, Form(min_length=1, max_length=100)],
    ):
        try:
            content = await file.read(MAX_CSV_BYTES + 1)
            if len(content) > MAX_CSV_BYTES:
                raise HTTPException(413, "Choose a CSV smaller than 5 MiB.")
            service = app.state.workspaces
            result = await run_in_threadpool(service.import_decisions, ws, content, target_column)
        except WorkspaceExists as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        finally:
            await file.close()
        workspace_mode(ws, response, service)
        return result

    def stored_workspace(ws):
        service = app.state.workspaces
        try:
            service.store.workspace(ws)
        except KeyError:
            return None
        return service

    @contract.post(
        "/cold-start",
        response_model=ColdStartState,
        description="Invent starting history from label names, with a $1 workspace cap.",
    )
    def cold_start(ws: WorkspaceId, body: ColdStartRequest, response: Response):
        try:
            result = app.state.workspaces.cold_start.start(ws, body.labels)
        except ColdStartConflict as error:
            raise HTTPException(409, str(error)) from error
        except ReaderUnavailable as error:
            raise HTTPException(503, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        response.headers["X-Secnd-Mode"] = "live"
        return result

    @contract.post(
        "/cold-start:import",
        response_model=ColdStartState,
        description="Import a one-column UTF-8 label list and start invented history.",
    )
    async def cold_start_import(
        ws: WorkspaceId, file: UploadFile, response: Response, paid: Annotated[bool, Form()]
    ):
        try:
            content = await file.read(MAX_CSV_BYTES + 1)
            if len(content) > MAX_CSV_BYTES:
                raise HTTPException(413, "Choose a label list smaller than 5 MiB.")
            body = ColdStartRequest(labels=imported_labels(content), paid=paid)
            return await run_in_threadpool(cold_start, ws, body, response)
        except ValueError:
            raise HTTPException(
                422, "Use 2–128 distinct label names in one column and enable the example budget."
            ) from None
        finally:
            await file.close()

    @contract.get(
        "/cold-start",
        response_model=ColdStartState,
        description="Saved invented history and generation progress.",
    )
    def cold_start_state(ws: WorkspaceId, response: Response):
        try:
            result = app.state.workspaces.cold_start.state(ws)
        except KeyError:
            raise HTTPException(404, "This workspace has no invented history.") from None
        response.headers["X-Secnd-Mode"] = "live"
        return result

    @contract.patch("/cold-start/history/{example_id}", response_model=ColdStartState)
    def edit_example(ws: WorkspaceId, example_id: str, body: ExampleEdit, response: Response):
        try:
            with app.state.workspaces._lock(ws):
                result = app.state.workspaces.cold_start.edit(ws, example_id, body.text, body.label)
        except KeyError:
            raise HTTPException(404, "Example was not found in this workspace.") from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        response.headers["X-Secnd-Mode"] = "live"
        return result

    @contract.delete("/cold-start/history/{example_id}", response_model=ColdStartState)
    def delete_example(ws: WorkspaceId, example_id: str, response: Response):
        try:
            with app.state.workspaces._lock(ws):
                result = app.state.workspaces.cold_start.edit(ws, example_id, delete=True)
        except KeyError:
            raise HTTPException(404, "Example was not found in this workspace.") from None
        response.headers["X-Secnd-Mode"] = "live"
        return result

    def workspace_mode(ws, response, service):
        response.headers["X-Secnd-Mode"] = "live" if service.has_model(ws) else "fixture"

    @contract.get("/labels", response_model=list[str], description="All allowed workspace labels.")
    def labels(ws: WorkspaceId, response: Response):
        if service := stored_workspace(ws):
            workspace_mode(ws, response, service)
            return service.store.workspace(ws)["labels"]
        return ["Travel", "Software"]

    @contract.post(
        "/predict",
        response_model=list[Decision],
        description="Routes and logs decisions for stored workspaces; otherwise returns fixtures.",
    )
    def predict(ws: WorkspaceId, body: PredictRequest, response: Response):
        if service := stored_workspace(ws):
            frame = pd.DataFrame(
                [
                    {
                        "id": item.item_id,
                        "text": item.text,
                        "amount": item.amount,
                        "date": item.date.isoformat() if item.date else None,
                        "merchant_key": merchant_key(item.text),
                        "source_app": item.source_app,
                    }
                    for item in body.items
                ]
            )
            try:
                result = service.route(ws, frame)
            except ReaderUnavailable as error:
                raise HTTPException(503, str(error)) from error
            except ModuleNotFoundError as error:
                if error.name != "backend.loop":
                    raise
                raise HTTPException(503, "Learning loop is not installed yet.") from error
            except RuntimeError as error:
                raise HTTPException(409, "Workspace has no trained model yet.") from error
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            workspace_mode(ws, response, service)
            return result
        return [decision(item) for item in body.items]

    def workspace_page(ws, kind, response, cursor, limit):
        if service := stored_workspace(ws):
            try:
                result = service.page(ws, kind, cursor, limit)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            workspace_mode(ws, response, service)
            return result
        items = (
            [decision(Item(item_id="fixture-auto-1", text="Example rail fare"), auto=True)]
            if kind == "feed"
            else [decision(Item(item_id="fixture-queue-1", text="Example subscription"))]
        )
        return page(items, cursor, limit)

    @contract.get(
        "/feed",
        response_model=Page[Decision],
        description="Newest persisted automatic decisions, with cursor pagination.",
    )
    def feed(
        ws: WorkspaceId,
        response: Response,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        return workspace_page(ws, "feed", response, cursor, limit)

    @contract.get(
        "/queue",
        response_model=Page[Decision],
        description="Unanswered persisted questions, least confident first; spot-checks included.",
    )
    def queue(
        ws: WorkspaceId,
        response: Response,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        return workspace_page(ws, "queue", response, cursor, limit)

    @contract.get(
        "/overridden",
        response_model=Page[Decision],
        description="Decisions whose latest active human answer differs from Ren's prediction.",
    )
    def overridden(
        ws: WorkspaceId,
        response: Response,
        cursor: Annotated[str | None, Query(max_length=512)] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        if not stored_workspace(ws):
            return Page[Decision](items=[])
        return workspace_page(ws, "overridden", response, cursor, limit)

    @contract.post(
        "/answers:undo",
        response_model=AnswerResult,
        description="Retract the most recent active human answer; retain its audit record.",
    )
    def undo_answer(ws: WorkspaceId, body: UndoAnswer, response: Response):
        service = stored_workspace(ws)
        if service is None:
            raise HTTPException(404, "Workspace was not found.")
        try:
            result = service.undo_answer(ws, body.idempotency_key)
        except KeyError as error:
            raise HTTPException(404, "Answer was not found in this workspace.") from error
        except store.IdempotencyConflict as error:
            raise HTTPException(409, str(error)) from error
        workspace_mode(ws, response, service)
        return result

    @contract.post(
        "/answers",
        response_model=AnswerResult,
        description="Persists idempotent answers for stored workspaces and queues retraining; "
        "otherwise returns fixtures.",
    )
    def answers(
        ws: WorkspaceId,
        body: Answer,
        response: Response,
        idempotency_key: Annotated[str, Header(min_length=1, max_length=256)],
    ):
        if service := stored_workspace(ws):
            try:
                result = service.answer(ws, body.item_id, body.label, idempotency_key)
            except store.IdempotencyConflict as error:
                raise HTTPException(409, str(error)) from error
            except KeyError as error:
                raise HTTPException(404, "Item was not found in this workspace.") from error
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            workspace_mode(ws, response, service)
            return result
        return AnswerResult(
            item_id=body.item_id, coverage=0, model_version="fixture-v0", retrain="fixture"
        )

    @contract.get(
        "/metrics",
        response_model=MetricsResult,
        description="Workspace calibration history and live guard; optional dataset selects a "
        "published test evaluation.",
    )
    def metrics(
        ws: WorkspaceId,
        response: Response,
        dataset: Literal["mtop", "banking77", "ipsa"] | None = None,
    ):
        if dataset is None and (service := stored_workspace(ws)):
            workspace_mode(ws, response, service)
            return MetricsResult(**service.metrics(ws))
        dataset = dataset or "mtop"
        try:
            measured, curves = evidence.evaluation(dataset)
        except FileNotFoundError as error:
            raise HTTPException(404, "Published evaluation was not found.") from error
        except (ValidationError, ValueError) as error:
            raise HTTPException(503, "Published evaluation is invalid or incomplete.") from error
        response.headers["X-Secnd-Mode"] = "evaluation"
        return MetricsResult(card=measured, curves=curves, dataset=dataset, history=[])

    @contract.get(
        "/models",
        response_model=Page[ModelVersion],
        description="Workspace registry versions and their measured cards; empty before training.",
    )
    def models(ws: WorkspaceId, response: Response):
        try:
            items = evidence.models(ws)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(503, "Model cards are invalid or incomplete.") from error
        response.headers["X-Secnd-Mode"] = "live" if any(v.current for v in items) else "fixture"
        return Page[ModelVersion](items=items)

    @contract.post(
        "/models/{v}:promote",
        response_model=ModelVersion,
        description="Select an existing registry version using the atomic current pointer.",
    )
    @contract.post(
        "/models/{v}:rollback",
        response_model=ModelVersion,
        description="Restore an existing registry version using the atomic current pointer.",
    )
    def rollback(ws: WorkspaceId, v: str, response: Response):
        try:
            result = evidence.promote(ws, v)
        except KeyError as error:
            raise HTTPException(404, "Model version was not found.") from error
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(503, "Model card is invalid or incomplete.") from error
        response.headers["X-Secnd-Mode"] = "live"
        return result

    @contract.post("/chat", response_model=ChatResult, description=contract_note)
    def chat(ws: WorkspaceId, body: ChatRequest):
        return ChatResult(answer="This is an example response. Chat is not connected yet.")

    app.include_router(contract)

    # Same-origin in production; Vite proxies /api and /docs during development.
    dist = config.frontend_dist.resolve()
    if (dist / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def frontend() -> FileResponse:
            return FileResponse(dist / "index.html")

    return app


app = create_app()
