import sqlite3
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import store
from backend.config import Settings
from backend.data import MAX_CSV_BYTES, CSVPreview, preview_csv


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

    # Same-origin in production; Vite proxies /api and /docs during development.
    dist = config.frontend_dist.resolve()
    if (dist / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        def frontend() -> FileResponse:
            return FileResponse(dist / "index.html")

    return app


app = create_app()
