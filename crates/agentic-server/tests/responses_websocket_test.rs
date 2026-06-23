#[allow(dead_code)]
mod common;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::body::Bytes;
use axum::http::header;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use futures::{SinkExt, StreamExt};
use http::StatusCode;
use serde_json::{Value, json};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler};
use agentic_core::proxy::ProxyState;
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_server::app::AppState;

use common::{spawn_gateway, test_config};

struct MockResponsesServer {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl MockResponsesServer {
    async fn start(responses: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let queue = Arc::new(Mutex::new(VecDeque::from(responses)));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let route_queue = Arc::clone(&queue);
        let route_requests = Arc::clone(&requests);

        let app = Router::new().route(
            "/v1/responses",
            post(move |body: Bytes| {
                let queue = Arc::clone(&route_queue);
                let requests = Arc::clone(&route_requests);
                async move {
                    let body = serde_json::from_slice::<Value>(&body).expect("request body should be JSON");
                    requests.lock().await.push(body);
                    let response = queue.lock().await.pop_front().expect("mock response queue exhausted");
                    Response::builder()
                        .status(StatusCode::OK)
                        .header(header::CONTENT_TYPE, "text/event-stream; charset=utf-8")
                        .body(axum::body::Body::from(response))
                        .unwrap()
                        .into_response()
                }
            }),
        );

        let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            url: format!("http://{addr}"),
            requests,
            handle,
        }
    }

    async fn request_bodies(&self) -> Vec<Value> {
        self.requests.lock().await.clone()
    }
}

impl Drop for MockResponsesServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

struct TestDb {
    path: PathBuf,
}

impl TestDb {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("agentic_ws_test_{}.db", uuid::Uuid::now_v7()));
        Self { path }
    }

    fn url(&self) -> String {
        format!("sqlite://{}", self.path.display())
    }
}

impl Drop for TestDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
        let _ = std::fs::remove_file(self.path.with_extension("db-shm"));
        let _ = std::fs::remove_file(self.path.with_extension("db-wal"));
    }
}

struct StorageBackedState {
    state: AppState,
    _db: TestDb,
}

async fn storage_backed_state(llm_url: &str) -> StorageBackedState {
    let db = TestDb::new();
    let db_url = db.url();
    let pool = create_pool_with_schema(Some(&db_url)).await.unwrap();
    let config = test_config(llm_url);
    let exec_ctx = Arc::new(ExecutionContext::new(
        ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
        ResponseHandler::new(ResponseStore::new(pool)),
        Arc::new(reqwest::Client::new()),
        config.llm_api_base.clone(),
        config.openai_api_key.clone(),
    ));
    let proxy_state = ProxyState::new(config.clone()).expect("proxy state");

    let state = AppState {
        proxy_state,
        exec_ctx,
        llm_api_base: config.llm_api_base,
    };
    StorageBackedState { state, _db: db }
}

fn ws_url(gateway_url: &str) -> String {
    format!("{}/v1/responses", gateway_url.replacen("http://", "ws://", 1))
}

async fn connect_responses_ws(url: &str) -> WebSocketStream<MaybeTlsStream<TcpStream>> {
    let (ws, response) = connect_async(ws_url(url)).await.expect("websocket handshake");
    assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
    ws
}

async fn recv_json(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>) -> Value {
    loop {
        let message = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .expect("timed out waiting for websocket message")
            .expect("websocket should yield a message")
            .expect("websocket message should be ok");
        match message {
            Message::Text(text) => return serde_json::from_str(&text).expect("message should be JSON"),
            Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => {}
            Message::Close(frame) => panic!("websocket closed before JSON event: {frame:?}"),
            Message::Binary(_) => panic!("unexpected binary websocket message"),
        }
    }
}

async fn recv_until_completed(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>) -> Vec<Value> {
    let mut events = Vec::new();
    loop {
        let event = recv_json(ws).await;
        let is_done = matches!(
            event.get("type").and_then(Value::as_str),
            Some("response.completed" | "error")
        );
        events.push(event);
        if is_done {
            return events;
        }
    }
}

async fn send_json(ws: &mut WebSocketStream<MaybeTlsStream<TcpStream>>, value: Value) {
    ws.send(Message::Text(value.to_string().into())).await.unwrap();
}

fn sse_response(response_id: &str, message_id: &str, text: &str) -> String {
    let created = json!({
        "type": "response.created",
        "sequence_number": 0,
        "response": {"id": response_id, "status": "in_progress"}
    });
    let added = json!({
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {"id": message_id, "type": "message"}
    });
    let delta = json!({
        "type": "response.output_text.delta",
        "sequence_number": 2,
        "item_id": message_id,
        "output_index": 0,
        "content_index": 0,
        "delta": text
    });
    let completed = json!({
        "type": "response.completed",
        "sequence_number": 3,
        "response": {"id": response_id, "status": "completed", "usage": null}
    });
    format!("data: {created}\n\ndata: {added}\n\ndata: {delta}\n\ndata: {completed}\n\ndata: [DONE]\n\n")
}

#[tokio::test]
async fn test_websocket_first_turn_forwards_incremental_response_events() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_upstream_1", "msg_upstream_1", "HELLO")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let events = recv_until_completed(&mut ws).await;
    let event_types = events
        .iter()
        .map(|event| event["type"].as_str().unwrap())
        .collect::<Vec<_>>();

    assert_eq!(
        event_types,
        vec![
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.completed"
        ]
    );
    assert_ne!(events[0]["response"]["id"], "resp_upstream_1");
    assert_eq!(events[2]["delta"], "HELLO");
    assert_eq!(events[3]["response"]["id"], events[0]["response"]["id"]);
    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0]["stream"], true);
    assert_eq!(requests[0]["input"][0]["content"], "hi");
    assert!(requests[0].get("type").is_none());
}

#[tokio::test]
async fn test_websocket_continuation_rehydrates_previous_response() {
    let mock = MockResponsesServer::start(vec![
        sse_response("resp_upstream_1", "msg_upstream_1", "HELLO"),
        sse_response("resp_upstream_2", "msg_upstream_2", "WORLD"),
    ])
    .await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "store": true,
            "stream": true
        }),
    )
    .await;
    let first = recv_until_completed(&mut ws).await;
    let first_completed = first.last().unwrap();
    let previous_response_id = first_completed["response"]["id"].as_str().unwrap();

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "previous_response_id": previous_response_id,
            "input": [{"type": "message", "role": "user", "content": "continue"}],
            "store": true,
            "stream": true
        }),
    )
    .await;
    let second = recv_until_completed(&mut ws).await;
    let event_types = second
        .iter()
        .map(|event| event["type"].as_str().unwrap())
        .collect::<Vec<_>>();
    let delta = second
        .iter()
        .find(|event| event["type"] == "response.output_text.delta")
        .unwrap();
    let completed = second.last().unwrap();

    assert_eq!(
        event_types,
        vec![
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.completed"
        ]
    );
    assert_eq!(delta["delta"], "WORLD");
    assert_eq!(completed["response"]["previous_response_id"], previous_response_id);

    let requests = mock.request_bodies().await;
    assert_eq!(requests.len(), 2);
    assert!(requests[1].get("previous_response_id").is_none());
    assert_eq!(requests[1]["input"][0]["content"], "hi");
    assert_eq!(requests[1]["input"][1]["role"], "assistant");
    assert_eq!(requests[1]["input"][1]["content"][0]["text"], "HELLO");
    assert_eq!(requests[1]["input"][2]["content"], "continue");
}

#[tokio::test]
async fn test_websocket_unknown_previous_response_returns_error_event() {
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    send_json(
        &mut ws,
        json!({
            "type": "response.create",
            "model": "test-model",
            "previous_response_id": "resp_missing",
            "input": [{"type": "message", "role": "user", "content": "continue"}],
            "store": true,
            "stream": true
        }),
    )
    .await;

    let error = recv_json(&mut ws).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["status"], StatusCode::NOT_FOUND.as_u16());
    assert_eq!(error["error"]["code"], "not_found");
    assert!(mock.request_bodies().await.is_empty());
}

#[tokio::test]
async fn test_websocket_rejects_binary_json_without_upstream_request() {
    let mock = MockResponsesServer::start(vec![sse_response("resp_upstream_1", "msg_upstream_1", "HELLO")]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    ws.send(Message::Binary(
        json!({
            "type": "response.create",
            "model": "test-model",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "store": true,
            "stream": true
        })
        .to_string()
        .into(),
    ))
    .await
    .unwrap();

    let error = recv_json(&mut ws).await;
    assert_eq!(error["type"], "error");
    assert_eq!(error["status"], StatusCode::BAD_REQUEST.as_u16());
    assert_eq!(error["error"]["code"], "invalid_request_error");
    assert!(mock.request_bodies().await.is_empty());
}

#[tokio::test]
async fn test_websocket_rejects_messages_larger_than_http_body_limit() {
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    if ws
        .send(Message::Text("x".repeat(10 * 1024 * 1024 + 1).into()))
        .await
        .is_ok()
    {
        let message = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .expect("timed out waiting for websocket close/error")
            .expect("websocket should yield a close or error");
        assert!(message.is_err() || matches!(message, Ok(Message::Close(_))));
    }
    assert!(mock.request_bodies().await.is_empty());
}

#[tokio::test]
async fn test_websocket_ping_returns_pong_without_upstream_request() {
    let mock = MockResponsesServer::start(vec![]).await;
    let fixture = storage_backed_state(&mock.url).await;
    let (gateway_url, _gateway) = spawn_gateway(fixture.state.clone()).await;
    let mut ws = connect_responses_ws(&gateway_url).await;

    ws.send(Message::Ping(Bytes::from_static(b"ping"))).await.unwrap();

    loop {
        let message = tokio::time::timeout(std::time::Duration::from_secs(2), ws.next())
            .await
            .expect("timed out waiting for websocket pong")
            .expect("websocket should yield a message")
            .expect("websocket message should be ok");
        match message {
            Message::Pong(payload) => {
                assert_eq!(payload, Bytes::from_static(b"ping"));
                break;
            }
            Message::Ping(_) | Message::Frame(_) => {}
            Message::Text(text) => panic!("unexpected text websocket message: {text}"),
            Message::Close(frame) => panic!("websocket closed before pong: {frame:?}"),
            Message::Binary(_) => panic!("unexpected binary websocket message"),
        }
    }

    assert!(mock.request_bodies().await.is_empty());
}
