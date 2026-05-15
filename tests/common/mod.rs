use std::convert::Infallible;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::Request;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use bytes::Bytes;
use futures::stream;
use http::StatusCode;
use praxis_core::config::Config;
use praxis_filter::{FilterFactory, FilterRegistry};
use praxis_test_utils::{free_port, start_proxy_with_registry};
use tokio::net::TcpListener;

use agentic_api::filters::agentic_loop::AgenticLoopFilter;
use agentic_api::filters::responses_proxy::ResponsesProxyFilter;
use agentic_api::filters::state_hydration::StateHydrationFilter;
use agentic_api::filters::tool_dispatch::ToolDispatchFilter;

pub fn agentic_registry() -> FilterRegistry {
    let mut registry = FilterRegistry::with_builtins();
    registry
        .register(
            "responses_proxy",
            FilterFactory::Http(std::sync::Arc::new(ResponsesProxyFilter::from_config)),
        )
        .unwrap();
    registry
        .register(
            "state_hydration",
            FilterFactory::Http(std::sync::Arc::new(StateHydrationFilter::from_config)),
        )
        .unwrap();
    registry
        .register(
            "agentic_loop",
            FilterFactory::Http(std::sync::Arc::new(AgenticLoopFilter::from_config)),
        )
        .unwrap();
    registry
        .register(
            "tool_dispatch",
            FilterFactory::Http(std::sync::Arc::new(ToolDispatchFilter::from_config)),
        )
        .unwrap();
    registry
}

pub fn proxy_yaml(proxy_port: u16, vllm_port: u16, api_key: Option<&str>) -> String {
    let key_line = match api_key {
        Some(k) => format!("\n        openai_api_key: \"{k}\""),
        None => String::new(),
    };
    format!(
        r#"
listeners:
  - name: default
    address: "127.0.0.1:{proxy_port}"
    filter_chains: [main]
filter_chains:
  - name: main
    filters:
      - filter: responses_proxy
        vllm_base_url: "http://127.0.0.1:{vllm_port}"{key_line}
"#
    )
}

pub async fn start_agentic_proxy(vllm_port: u16, api_key: Option<&str>) -> (String, u16) {
    let proxy_port = free_port();
    let yaml = proxy_yaml(proxy_port, vllm_port, api_key);
    let config = Config::from_yaml(&yaml).unwrap();
    let registry = agentic_registry();
    let addr = tokio::task::spawn_blocking(move || {
        let guard = start_proxy_with_registry(&config, &registry);
        let addr = guard.addr().to_owned();
        std::mem::forget(guard);
        addr
    })
    .await
    .unwrap();
    (addr, proxy_port)
}

async fn health_handler() -> impl IntoResponse {
    StatusCode::OK
}

async fn responses_handler(req: Request) -> Response {
    let headers = req.headers().clone();
    let body_bytes = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
        .await
        .unwrap_or_default();

    let body: serde_json::Value = serde_json::from_slice(&body_bytes).unwrap_or_default();

    if body
        .get("echo_auth")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        let auth = headers.get("authorization").and_then(|v| v.to_str().ok()).unwrap_or("");
        let resp_body = serde_json::json!({"authorization": auth});
        return (
            StatusCode::OK,
            [("content-type", "application/json"), ("x-vllm", "responses")],
            serde_json::to_string(&resp_body).unwrap(),
        )
            .into_response();
    }

    if body.get("force_error").and_then(serde_json::Value::as_u64) == Some(429) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [("content-type", "application/json"), ("x-vllm", "error")],
            r#"{"error":{"message":"rate limited","code":"rate_limit"}}"#,
        )
            .into_response();
    }

    if body.get("stream").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        let chunks: Vec<Result<Bytes, Infallible>> = vec![
            Ok(Bytes::from(
                "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
            )),
            Ok(Bytes::from("data: [DONE]\n\n")),
        ];
        let body = Body::from_stream(stream::iter(chunks));
        return (
            StatusCode::OK,
            [
                ("content-type", "text/event-stream; charset=utf-8"),
                ("x-vllm", "responses-stream"),
            ],
            body,
        )
            .into_response();
    }

    let out = r#"{"id":"resp_test","object":"response","status":"completed"}"#;
    (
        StatusCode::OK,
        [
            ("content-type", "application/json"),
            ("x-vllm", "responses"),
            ("connection", "keep-alive"),
        ],
        out,
    )
        .into_response()
}

pub async fn spawn_vllm() -> (u16, tokio::task::JoinHandle<()>) {
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/responses", post(responses_handler));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (port, handle)
}

pub async fn spawn_mid_stream_failure_vllm() -> (u16, tokio::task::JoinHandle<()>) {
    async fn handler(_req: Request) -> Response {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(2);
        tokio::spawn(async move {
            let _ = tx
                .send(Ok(Bytes::from(
                    "data: {\"type\":\"response.output_text.delta\",\"delta\":\"hello\"}\n\n",
                )))
                .await;
            tokio::time::sleep(Duration::from_millis(10)).await;
            drop(tx);
        });
        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let body = Body::from_stream(stream);
        (
            StatusCode::OK,
            [
                ("content-type", "text/event-stream; charset=utf-8"),
                ("x-vllm", "fake-stream"),
            ],
            body,
        )
            .into_response()
    }

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/responses", post(handler));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    (port, handle)
}
