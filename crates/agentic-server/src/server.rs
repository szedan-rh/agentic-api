use std::sync::Arc;

use agentic_core::config::Config;
use agentic_core::error::Error;
use agentic_core::proxy::ProxyState;
use agentic_core::readiness::wait_llm_ready;
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_core::vector_search::ogx::OgxStore;
use agentic_server::app::{ServerConfig, build_router};
use agentic_server::handler::AppState;
use tokio::net::TcpListener;
use tracing::info;

async fn build_app_state(
    config: Config,
    ogx_base_url: &str,
    max_iterations: u32,
    database_url: Option<&str>,
) -> Result<Arc<AppState>, Error> {
    let proxy = ProxyState::new(config)?;
    let client = reqwest::Client::new();
    let ogx_store = Arc::new(OgxStore::new(ogx_base_url, client));
    let pool = create_pool_with_schema(database_url).await?;
    let response_store = ResponseStore::new(pool.clone());
    let conversation_store = ConversationStore::new(pool);

    Ok(Arc::new(AppState {
        proxy,
        max_iterations,
        vector_search: ogx_store,
        response_store,
        conversation_store,
    }))
}

async fn serve_gateway(
    config: Config,
    host: &str,
    port: u16,
    ogx_base_url: &str,
    max_iterations: u32,
    database_url: Option<&str>,
) -> Result<(), Error> {
    let addr = format!("{host}:{port}");
    let state = build_app_state(config, ogx_base_url, max_iterations, database_url).await?;
    let server_config = ServerConfig::from_env();
    let router = build_router(state, &server_config);
    let listener = TcpListener::bind(&addr).await?;
    info!("gateway listening on {addr}");
    axum::serve(listener, router).await?;
    Ok(())
}

/// Start the gateway after the LLM becomes ready.
///
/// # Errors
///
/// Returns an error if LLM readiness polling fails or the server cannot bind.
pub async fn run(
    config: Config,
    host: &str,
    port: u16,
    ogx_base_url: &str,
    max_iterations: u32,
    database_url: Option<&str>,
) -> Result<(), Error> {
    wait_llm_ready(&config).await?;
    info!("LLM ready: {}", config.llm_api_base);
    serve_gateway(config, host, port, ogx_base_url, max_iterations, database_url).await
}

/// Spawn vLLM as a subprocess and run the gateway in the foreground.
///
/// # Errors
///
/// Returns an error if vLLM fails to start or the gateway errors.
pub async fn run_with_llm(
    config: Config,
    host: &str,
    port: u16,
    llm_args: Vec<String>,
    ogx_base_url: &str,
    max_iterations: u32,
    database_url: Option<&str>,
) -> Result<(), Error> {
    let mut cmd = tokio::process::Command::new("python");
    cmd.arg("-m").arg("vllm.entrypoints.openai.api_server");
    cmd.args(&llm_args);

    let mut child = cmd.spawn()?;
    info!("spawned vLLM subprocess (pid {})", child.id().unwrap_or(0));

    let readiness_result = tokio::select! {
        ready = wait_llm_ready(&config) => ready,
        status = child.wait() => {
            let status = status?;
            Err(Error::LlmProcessExited {
                status: status.to_string(),
            })
        }
    };

    match readiness_result {
        Ok(()) => info!("LLM ready: {}", config.llm_api_base),
        Err(err) => {
            let _ = child.kill().await;
            let _ = child.wait().await;
            return Err(err);
        }
    }

    let result = tokio::select! {
        gateway = serve_gateway(config, host, port, ogx_base_url, max_iterations, database_url) => gateway,
        status = child.wait() => {
            let status = status?;
            Err(Error::LlmProcessExited {
                status: status.to_string(),
            })
        }
    };

    let _ = child.kill().await;
    let _ = child.wait().await;
    result
}
