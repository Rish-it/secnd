import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

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
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend import store
from backend.config import Settings
from backend.data import MAX_CSV_BYTES, CSVPreview, preview_csv
from backend.schemas import (
    Answer,
    AnswerResult,
    ChatRequest,
    ChatResult,
    Decision,
    Error,
    EvalCard,
    ImportResult,
    Item,
    MetricsResult,
    ModelVersion,
    Page,
    PredictRequest,
    Reasons,
    WorkspaceId,
)


class Health(BaseModel):
    status: str = "ok"
    database: str = "ready"


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(store.initialize, config.database_path)
        yield

    app = FastAPI(title="Secnd", version="0.1.0", lifespan=lifespan)

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
        tags=["Workspace contract fixtures"],
        responses={422: {"model": Error}, 413: {"model": Error}, 404: {"model": Error}},
        dependencies=[Depends(fixture_mode)],
    )
    contract_note = (
        "Contract fixture only: no model runs and nothing is saved. "
        "Responses carry X-Secnd-Mode: fixture. Metrics are unmeasured."
    )

    def card() -> EvalCard:
        return EvalCard(model_version="fixture-v0", coverage=0)

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

    @contract.post("/decisions:import", response_model=ImportResult, description=contract_note)
    async def import_decisions(
        ws: WorkspaceId,
        file: UploadFile,
        target_column: Annotated[str, Form(min_length=1, max_length=100)],
    ):
        inspected = await preview(file)
        if target_column not in inspected.columns:
            raise HTTPException(status_code=422, detail="The target column is missing.")
        return ImportResult(
            workspace_id=ws,
            row_count=inspected.row_count,
            target_column=target_column,
            card=card(),
        )

    @contract.post("/predict", response_model=list[Decision], description=contract_note)
    def predict(ws: WorkspaceId, body: PredictRequest):
        return [decision(item) for item in body.items]

    @contract.get("/feed", response_model=Page[Decision], description=contract_note)
    def feed(
        ws: WorkspaceId,
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        items = [decision(Item(item_id="fixture-auto-1", text="Example rail fare"), auto=True)]
        return page(items, cursor, limit)

    @contract.get("/queue", response_model=Page[Decision], description=contract_note)
    def queue(
        ws: WorkspaceId,
        cursor: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
    ):
        item = decision(Item(item_id="fixture-queue-1", text="Example subscription"))
        return page([item], cursor, limit)

    @contract.post("/answers", response_model=AnswerResult, description=contract_note)
    def answers(
        ws: WorkspaceId,
        body: Answer,
        idempotency_key: Annotated[str, Header(min_length=1, max_length=256)],
    ):
        return AnswerResult(
            item_id=body.item_id, coverage=0, model_version="fixture-v0", retrain="fixture"
        )

    @contract.get("/metrics", response_model=MetricsResult, description=contract_note)
    def metrics(ws: WorkspaceId):
        return MetricsResult(card=card(), history=[])

    @contract.get("/models", response_model=Page[ModelVersion], description=contract_note)
    def models(ws: WorkspaceId):
        return Page[ModelVersion](
            items=[ModelVersion(model_version="fixture-v0", current=True, card=card())]
        )

    @contract.post("/models/{v}:rollback", response_model=ModelVersion, description=contract_note)
    def rollback(ws: WorkspaceId, v: str):
        if v != "fixture-v0":
            raise HTTPException(status_code=404, detail="Model version was not found.")
        return ModelVersion(model_version=v, current=True, card=card())

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
