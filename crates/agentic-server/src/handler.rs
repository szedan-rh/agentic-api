use std::sync::Arc;

use axum::body::Body;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Request, State};
use axum::http::HeaderMap;
use axum::http::request::Parts;
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use either::Either;
use futures::StreamExt;
use http::StatusCode;
use serde_json::{Value, json};
use tracing::warn;

use agentic_core::executor::accumulator::ResponseAccumulator;
use agentic_core::executor::{
    BoxStream, ExecutionContext, ExecutorError, RequestContext, call_inference, create_conversation, execute,
    persist_response, rehydrate_conversation,
};
use agentic_core::proxy::{ProxyBody, ProxyRequest, ProxyResponse, error_response, proxy_request};
use agentic_core::types::ResponsePayload;
use agentic_core::types::request_response::RequestPayload;
use agentic_core::utils::common::serialize_to_string;

use crate::app::AppState;

const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

pub async fn health() -> impl IntoResponse {
    StatusCode::OK
}

pub async fn ready(State(state): State<AppState>) -> impl IntoResponse {
    let base = state.llm_api_base.trim_end_matches('/');
    let url = format!("{base}/health");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build();

    let Ok(client) = client else {
        return StatusCode::SERVICE_UNAVAILABLE;
    };

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => StatusCode::OK,
        Ok(resp) => {
            warn!("LLM backend not ready: {}", resp.status());
            StatusCode::SERVICE_UNAVAILABLE
        }
        Err(e) => {
            warn!("LLM backend unreachable: {e}");
            StatusCode::SERVICE_UNAVAILABLE
        }
    }
}

async fn read_bytes(body: Body) -> Result<Bytes, Response> {
    axum::body::to_bytes(body, MAX_BODY_SIZE).await.map_err(|_| {
        convert_response(error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "body_too_large",
            "request body too large",
        ))
    })
}

async fn read_and_parse(body: Body) -> Result<(Bytes, RequestPayload), Response> {
    let bytes = read_bytes(body).await?;
    let payload = serde_json::from_slice::<RequestPayload>(&bytes)
        .map_err(|e| executor_error_response(ExecutorError::from(e)))?;
    Ok((bytes, payload))
}

fn extract_store(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|j| j.get("store").and_then(serde_json::Value::as_bool))
        .unwrap_or(true)
}

/// # Panics
/// Panics if the response builder produces an invalid response (unreachable in practice).
pub fn convert_response(resp: ProxyResponse) -> Response {
    let mut builder = Response::builder().status(resp.status);
    for (name, value) in &resp.headers {
        builder = builder.header(name, value);
    }
    match resp.body {
        ProxyBody::Full(bytes) => builder.body(Body::from(bytes)).expect("valid response"),
        ProxyBody::Stream(stream) => builder.body(Body::from_stream(stream)).expect("valid response"),
    }
}

async fn proxy_responses(state: &AppState, parts: Parts, body: Bytes) -> Response {
    let proxy_req = ProxyRequest {
        headers: parts.headers,
        body,
        query: parts.uri.query().map(str::to_string),
    };
    convert_response(proxy_request(proxy_req, &state.proxy_state).await)
}

fn resolve_exec_ctx_from_headers(state: &AppState, headers: &HeaderMap) -> Arc<ExecutionContext> {
    let request_auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    if request_auth.is_some() && request_auth != state.exec_ctx.client_auth {
        let mut ctx = (*state.exec_ctx).clone();
        ctx.client_auth = request_auth;
        Arc::new(ctx)
    } else {
        Arc::clone(&state.exec_ctx)
    }
}

fn resolve_exec_ctx(state: &AppState, parts: &Parts) -> Arc<ExecutionContext> {
    resolve_exec_ctx_from_headers(state, &parts.headers)
}

fn sse_response(stream: BoxStream) -> Response {
    let byte_stream = stream.map(|line| Ok::<Bytes, std::convert::Infallible>(Bytes::from(line)));
    Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "text/event-stream; charset=utf-8")
        .header("Cache-Control", "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(Body::from_stream(byte_stream))
        .expect("valid SSE response")
}

async fn execute_responses(state: &AppState, parts: Parts, payload: RequestPayload) -> Response {
    match execute(payload, resolve_exec_ctx(state, &parts)).await {
        Ok(Either::Left(response_payload)) => axum::Json(response_payload).into_response(),
        Ok(Either::Right(stream)) => sse_response(stream),
        Err(e) => executor_error_response(e),
    }
}

/// # Panics
/// Panics if the response builder produces an invalid response (unreachable in practice).
pub fn executor_error_response(err: ExecutorError) -> Response {
    let status = err.http_status();
    if !matches!(err, ExecutorError::LLMRequest { .. }) {
        warn!("executor error ({status}): {err}");
    }
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(err.into_response_body()))
        .expect("valid error response")
}

pub async fn conversations(State(state): State<AppState>, req: Request) -> Response {
    let (_, body) = req.into_parts();
    let bytes = match read_bytes(body).await {
        Ok(b) => b,
        Err(e) => return e,
    };

    if !extract_store(&bytes) {
        return executor_error_response(ExecutorError::InvalidRequest("conversations require store=true".into()));
    }

    match create_conversation(&state.exec_ctx).await {
        Ok(data) => axum::Json(json!({
            "id": data.conversation_id,
            "created_at": data.created_at,
            "object": "conversation",
            "metadata": {}
        }))
        .into_response(),
        Err(e) => executor_error_response(e),
    }
}

pub async fn responses(State(state): State<AppState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let (bytes, payload) = match read_and_parse(body).await {
        Ok(v) => v,
        Err(e) => return e,
    };

    let should_persist = payload.store || payload.previous_response_id.is_some() || payload.conversation_id.is_some();

    if should_persist {
        execute_responses(&state, parts, payload).await
    } else {
        proxy_responses(&state, parts, bytes).await
    }
}

pub async fn responses_ws(State(state): State<AppState>, headers: HeaderMap, ws: WebSocketUpgrade) -> Response {
    ws.max_message_size(MAX_BODY_SIZE)
        .max_frame_size(MAX_BODY_SIZE)
        .on_upgrade(move |socket| responses_ws_loop(socket, state, headers))
}

async fn responses_ws_loop(mut socket: WebSocket, state: AppState, headers: HeaderMap) {
    while let Some(message) = socket.recv().await {
        match message {
            Ok(Message::Text(text)) => {
                if !handle_ws_text(&mut socket, &state, &headers, text.as_str()).await {
                    break;
                }
            }
            Ok(Message::Binary(_)) => {
                if !send_ws_error(
                    &mut socket,
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    "websocket messages must be JSON text frames",
                )
                .await
                {
                    break;
                }
            }
            Ok(Message::Close(_)) => break,
            Ok(Message::Ping(payload)) => {
                if socket.send(Message::Pong(payload)).await.is_err() {
                    break;
                }
            }
            Ok(Message::Pong(_)) => {}
            Err(e) => {
                warn!("responses websocket receive error: {e}");
                break;
            }
        }
    }
}

async fn handle_ws_text(socket: &mut WebSocket, state: &AppState, headers: &HeaderMap, text: &str) -> bool {
    let value = match serde_json::from_str::<Value>(text) {
        Ok(value) => value,
        Err(e) => {
            return send_ws_error(
                socket,
                StatusCode::BAD_REQUEST,
                "invalid_json",
                &format!("invalid JSON: {e}"),
            )
            .await;
        }
    };

    if value.get("type").and_then(Value::as_str) != Some("response.create") {
        return send_ws_error(
            socket,
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "websocket message type must be response.create",
        )
        .await;
    }

    let mut payload = match serde_json::from_value::<RequestPayload>(value) {
        Ok(payload) => payload,
        Err(e) => return send_executor_error(socket, ExecutorError::from(e)).await,
    };
    payload.stream = true;

    let exec_ctx = resolve_exec_ctx_from_headers(state, headers);
    let ctx = match rehydrate_conversation(payload, &exec_ctx).await {
        Ok(ctx) => ctx,
        Err(e) => return send_executor_error(socket, e).await,
    };
    let upstream_json = match serialize_to_string(&ctx.enriched_request.to_upstream_request(true)) {
        Ok(upstream_json) => upstream_json,
        Err(e) => return send_executor_error(socket, ExecutorError::from(e)).await,
    };

    let should_persist = ctx.original_request.store
        || ctx.original_request.previous_response_id.is_some()
        || ctx.conversation_id.is_some();
    let mut lines = Vec::new();
    let mut stream = Box::pin(call_inference(
        upstream_json,
        exec_ctx.responses_url(),
        Arc::clone(&exec_ctx.client),
        exec_ctx.client_auth.clone(),
        exec_ctx.streaming_timeout,
    ));

    while let Some(line) = stream.next().await {
        let line = match line {
            Ok(line) => line,
            Err(e) => return send_executor_error(socket, e).await,
        };
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            continue;
        }
        let mut value = match serde_json::from_str::<Value>(data) {
            Ok(value) => value,
            Err(e) => return send_executor_error(socket, ExecutorError::from(e)).await,
        };
        apply_gateway_response_ids(&mut value, &ctx);
        if !send_ws_json(socket, value).await {
            return false;
        }
        lines.push(line);
    }

    if should_persist && !lines.is_empty() {
        let acc = ResponseAccumulator::from_sse_lines(lines, ctx.conversation_id.as_deref());
        let mut payload = acc.finalize(
            &ctx.enriched_request.model,
            ctx.original_request.previous_response_id.as_deref(),
            ctx.original_request.instructions.as_deref(),
        );
        apply_gateway_payload_ids(&mut payload, &ctx);
        let ch = exec_ctx.conv_handler.clone();
        let rh = exec_ctx.resp_handler.clone();
        if let Err(e) = persist_response(payload, ctx, ch, rh).await {
            warn!("persist failed: {e}");
        }
    }
    true
}

fn apply_gateway_response_ids(value: &mut Value, ctx: &RequestContext) {
    let Some(response) = value.get_mut("response").and_then(Value::as_object_mut) else {
        return;
    };
    response.insert("id".to_owned(), Value::String(ctx.response_id.clone()));
    if let Some(previous_response_id) = &ctx.original_request.previous_response_id {
        response.insert(
            "previous_response_id".to_owned(),
            Value::String(previous_response_id.clone()),
        );
    }
    if let Some(conversation_id) = &ctx.conversation_id {
        response.insert("conversation_id".to_owned(), Value::String(conversation_id.clone()));
    }
}

fn apply_gateway_payload_ids(payload: &mut ResponsePayload, ctx: &RequestContext) {
    payload.id.clone_from(&ctx.response_id);
    payload.conversation_id.clone_from(&ctx.conversation_id);
    payload
        .previous_response_id
        .clone_from(&ctx.original_request.previous_response_id);
}

async fn send_executor_error(socket: &mut WebSocket, err: ExecutorError) -> bool {
    let status = err.http_status();
    let code = err.error_code();
    let message = err.to_string();
    send_ws_error(socket, status, code, &message).await
}

async fn send_ws_error(socket: &mut WebSocket, status: StatusCode, code: &str, message: &str) -> bool {
    send_ws_json(
        socket,
        json!({
            "type": "error",
            "status": status.as_u16(),
            "error": {
                "message": message,
                "type": code,
                "code": code
            }
        }),
    )
    .await
}

async fn send_ws_json(socket: &mut WebSocket, value: Value) -> bool {
    match serde_json::to_string(&value) {
        Ok(text) => socket.send(Message::Text(text.into())).await.is_ok(),
        Err(e) => {
            warn!("failed to serialize websocket event: {e}");
            false
        }
    }
}
