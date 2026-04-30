"""FastAPI router for the Vector Stores API."""

from fastapi import APIRouter, HTTPException, Request

from agentic_api.store.vector_store import VectorStoreManager
from agentic_api.types.vector_stores import (
    AttachFileRequest,
    CreateVectorStoreRequest,
    SearchVectorStoreRequest,
)

router = APIRouter()


def _get_manager(request: Request) -> VectorStoreManager:
    mgr: VectorStoreManager | None = getattr(
        request.app.state, "vector_store_manager", None
    )
    if mgr is None:
        raise HTTPException(
            status_code=501, detail="Vector store support is not enabled."
        )
    return mgr


# ------------------------------------------------------------------
# Store CRUD
# ------------------------------------------------------------------


@router.post("/v1/vector_stores")
async def create_vector_store(request: Request):
    mgr = _get_manager(request)
    body = CreateVectorStoreRequest.model_validate(await request.json())
    result = await mgr.create_store(
        name=body.name,
        embedding_model=body.embedding_model,
        embedding_dimension=body.embedding_dimension,
        metadata=body.metadata_,
    )
    return result.model_dump(mode="json", by_alias=True)


@router.get("/v1/vector_stores")
async def list_vector_stores(request: Request):
    mgr = _get_manager(request)
    stores = await mgr.list_stores()
    return {
        "object": "list",
        "data": [s.model_dump(mode="json", by_alias=True) for s in stores],
    }


@router.get("/v1/vector_stores/{store_id}")
async def get_vector_store(request: Request, store_id: str):
    mgr = _get_manager(request)
    result = await mgr.get_store(store_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Vector store '{store_id}' not found."
        )
    return result.model_dump(mode="json", by_alias=True)


@router.delete("/v1/vector_stores/{store_id}")
async def delete_vector_store(request: Request, store_id: str):
    mgr = _get_manager(request)
    deleted = await mgr.delete_store(store_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Vector store '{store_id}' not found."
        )
    return {"id": store_id, "object": "vector_store", "deleted": True}


# ------------------------------------------------------------------
# File attachment
# ------------------------------------------------------------------


@router.post("/v1/vector_stores/{store_id}/files")
async def attach_file(request: Request, store_id: str):
    mgr = _get_manager(request)
    body = AttachFileRequest.model_validate(await request.json())
    try:
        result = await mgr.attach_file(
            store_id=store_id,
            file_id=body.file_id,
            chunking_strategy=body.chunking_strategy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.model_dump(mode="json")


@router.get("/v1/vector_stores/{store_id}/files")
async def list_files(request: Request, store_id: str):
    mgr = _get_manager(request)
    files = await mgr.list_files(store_id)
    return {"object": "list", "data": [f.model_dump(mode="json") for f in files]}


@router.get("/v1/vector_stores/{store_id}/files/{file_id}")
async def get_file(request: Request, store_id: str, file_id: str):
    mgr = _get_manager(request)
    result = await mgr.get_file(store_id, file_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"File '{file_id}' not found in store '{store_id}'."
        )
    return result.model_dump(mode="json")


@router.delete("/v1/vector_stores/{store_id}/files/{file_id}")
async def delete_file(request: Request, store_id: str, file_id: str):
    mgr = _get_manager(request)
    deleted = await mgr.delete_file(store_id, file_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"File '{file_id}' not found in store '{store_id}'."
        )
    return {"id": file_id, "object": "vector_store.file.deleted", "deleted": True}


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------


@router.post("/v1/vector_stores/{store_id}/search")
async def search_vector_store(request: Request, store_id: str):
    mgr = _get_manager(request)
    body = SearchVectorStoreRequest.model_validate(await request.json())
    try:
        result = await mgr.search(
            store_id=store_id,
            query=body.query,
            max_num_results=body.max_num_results,
            search_mode=body.search_mode,
            ranking_options=body.ranking_options,
            filters=body.filters,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result.model_dump(mode="json")
