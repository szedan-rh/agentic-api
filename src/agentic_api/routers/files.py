from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from agentic_api.store.file_store import FileStore

router = APIRouter()


@router.post("/v1/files")
async def upload_file(
    request: Request,
    file: UploadFile,
    purpose: str = Form(...),
) -> JSONResponse:
    file_store: FileStore = request.app.state.file_store
    content = await file.read()
    file_obj = await file_store.upload(
        filename=file.filename or "upload",
        content=content,
        purpose=purpose,
    )
    return JSONResponse(status_code=200, content=file_obj.model_dump(mode="json"))


@router.get("/v1/files")
async def list_files(
    request: Request,
    limit: int = 20,
    order: str = "desc",
    after: str | None = None,
    purpose: str | None = None,
) -> JSONResponse:
    file_store: FileStore = request.app.state.file_store
    result = await file_store.list(
        limit=limit, order=order, after=after, purpose=purpose
    )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


@router.get("/v1/files/{file_id}")
async def get_file(request: Request, file_id: str) -> JSONResponse:
    file_store: FileStore = request.app.state.file_store
    file_obj = await file_store.get(file_id=file_id)
    if file_obj is None:
        raise HTTPException(
            status_code=404, detail=f"No file found with id '{file_id}'"
        )
    return JSONResponse(status_code=200, content=file_obj.model_dump(mode="json"))


@router.delete("/v1/files/{file_id}")
async def delete_file(request: Request, file_id: str) -> JSONResponse:
    file_store: FileStore = request.app.state.file_store
    result = await file_store.delete(file_id=file_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"No file found with id '{file_id}'"
        )
    return JSONResponse(status_code=200, content=result.model_dump(mode="json"))


@router.get("/v1/files/{file_id}/content")
async def get_file_content(request: Request, file_id: str) -> Response:
    file_store: FileStore = request.app.state.file_store
    content = await file_store.get_content(file_id=file_id)
    if content is None:
        raise HTTPException(
            status_code=404, detail=f"No file found with id '{file_id}'"
        )
    return Response(content=content, media_type="application/octet-stream")
