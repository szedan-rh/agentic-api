from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.core.proxy import ProxyClientManager
from agentic_api.database.schema import SchemaManager
from agentic_api.database.db_engine import create_db_engine_async
from agentic_api.routers import files, responses, vector_stores
from agentic_api.store.conversation import ConversationStore
from agentic_api.store.response import ResponseStore


def create_app(runtime_config: RuntimeConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime_config = runtime_config
        app.state.proxy_client_manager = ProxyClientManager()

        embedding_client = None

        if runtime_config.response_store_enabled:
            engine = create_db_engine_async(
                db_url=runtime_config.db_url,
                db_dialect=runtime_config.db_dialect,
            )
            schema_manager = SchemaManager(engine)
            await schema_manager.ensure_ready(
                gateway_workers=runtime_config.gateway_workers,
                db_dialect=runtime_config.db_dialect,
            )
            app.state.response_store = ResponseStore(engine=engine)
            app.state.conversation_store = ConversationStore(engine=engine)

            from agentic_api.store.file_store import FileStore

            app.state.file_store = FileStore(engine=engine)

            if runtime_config.embedding_model:
                from agentic_api.store.embedding_client import EmbeddingClient
                from agentic_api.store.vector_store import VectorStoreManager

                emb_base = (
                    runtime_config.embedding_api_base or runtime_config.llm_api_base
                )
                emb_key = (
                    runtime_config.embedding_api_key or runtime_config.openai_api_key
                )

                embedding_client = EmbeddingClient(
                    base_url=emb_base,
                    api_key=emb_key,
                    model=runtime_config.embedding_model,
                )
                app.state.vector_store_manager = VectorStoreManager(
                    engine=engine,
                    file_store=app.state.file_store,
                    embedding_client=embedding_client,
                    db_path=runtime_config.vector_store_db_path,
                )
            else:
                app.state.vector_store_manager = None
        else:
            app.state.conversation_store = None
            app.state.response_store = None
            app.state.file_store = None
            app.state.vector_store_manager = None

        yield

        await app.state.proxy_client_manager.aclose()
        if embedding_client is not None:
            await embedding_client.aclose()
        if runtime_config.response_store_enabled:
            await engine.dispose()

    app = FastAPI(
        title="Agentic API",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(responses.router)
    app.include_router(files.router)
    app.include_router(vector_stores.router)
    return app
