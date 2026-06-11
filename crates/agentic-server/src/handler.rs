use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use tracing::{debug, error, warn};

use agentic_core::proxy::{ProxyBody, ProxyRequest, ProxyResponse, ProxyState, error_response};
use agentic_core::storage::{ConversationStore, InOutItem, ResponseMetadata, ResponseStore};
use agentic_core::types::io::{
    FunctionToolCall, InputItem, InputMessage, InputMessageContent, OutputItem, OutputMessage, ToolChoice,
};
use agentic_core::utils::uuid7_str;
use agentic_core::vector_search::VectorSearch;
use agentic_core::vector_search::types::{ResponseBody, ResponseInput, ResponseRequest, SearchResult, VllmOutputItem};

const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;

pub struct AppState {
    pub proxy: ProxyState,
    pub max_iterations: u32,
    pub vector_search: Arc<dyn VectorSearch>,
    pub response_store: ResponseStore,
    pub conversation_store: ConversationStore,
}

#[allow(clippy::unused_async)]
pub async fn health() -> StatusCode {
    StatusCode::OK
}

pub async fn ready(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let base = state.proxy.config.llm_api_base.trim_end_matches('/');
    let url = format!("{base}/health");

    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = state.proxy.config.openai_api_key.as_deref() {
        let trimmed = key.trim();
        if !trimmed.is_empty() {
            if let Ok(v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {trimmed}")) {
                headers.insert(reqwest::header::AUTHORIZATION, v);
            }
        }
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .default_headers(headers)
        .build();

    let Ok(client) = client else {
        return StatusCode::SERVICE_UNAVAILABLE;
    };

    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => StatusCode::OK,
        Ok(resp) => {
            warn!("LLM backend not ready: status {}", resp.status());
            StatusCode::SERVICE_UNAVAILABLE
        }
        Err(e) => {
            warn!("LLM backend unreachable: {e}");
            StatusCode::SERVICE_UNAVAILABLE
        }
    }
}

fn convert_response(resp: ProxyResponse) -> Response {
    let mut builder = Response::builder().status(resp.status);
    for (name, value) in &resp.headers {
        builder = builder.header(name, value);
    }
    match resp.body {
        ProxyBody::Full(bytes) => builder.body(Body::from(bytes)).expect("valid response"),
        ProxyBody::Stream(stream) => builder.body(Body::from_stream(stream)).expect("valid response"),
    }
}

pub async fn handle_responses(State(state): State<Arc<AppState>>, req: axum::extract::Request) -> Response {
    let (parts, body) = req.into_parts();

    let Ok(body) = axum::body::to_bytes(body, MAX_BODY_SIZE).await else {
        return convert_response(error_response(
            StatusCode::PAYLOAD_TOO_LARGE,
            "body_too_large",
            "Request body too large",
        ));
    };

    let request: ResponseRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                serde_json::json!({"error": {"message": format!("invalid request body: {e}")}}).to_string(),
            )
                .into_response();
        }
    };

    let has_file_search = request.tools.iter().any(|t| t.r#type == "file_search");

    if has_file_search {
        if request.stream {
            return json_error_response(
                StatusCode::BAD_REQUEST,
                "streaming file_search is not supported by this gateway path yet",
            );
        }

        match agentic_loop(&state, &parts.headers, request).await {
            Ok(resp) => resp,
            Err(e) => {
                error!(error = %e, "agentic loop failed");
                json_error_response(StatusCode::BAD_GATEWAY, &format!("agentic loop error: {e}"))
            }
        }
    } else {
        let proxy_req = ProxyRequest {
            headers: parts.headers,
            body,
            query: parts.uri.query().map(String::from),
        };
        convert_response(agentic_core::proxy::proxy_request(proxy_req, &state.proxy).await)
    }
}

async fn agentic_loop(
    state: &AppState,
    client_headers: &HeaderMap,
    mut request: ResponseRequest,
) -> Result<Response, agentic_core::error::Error> {
    let original_input = request.input.clone();
    let original_previous_response_id = request.previous_response_id.clone();
    let mut turn_items = input_values_to_storage_items(&original_input);

    hydrate_previous_response(state, &mut request).await?;

    let vector_store_ids: Vec<String> = request
        .tools
        .iter()
        .filter(|t| t.r#type == "file_search")
        .filter_map(|t| t.vector_store_ids.clone())
        .flatten()
        .collect();

    for iteration in 0..state.max_iterations {
        debug!(iteration, "agentic loop iteration");

        let response_body = match send_agentic_request(state, client_headers, &request).await? {
            Ok(response_body) => response_body,
            Err(error_response) => return Ok(error_response),
        };
        let tool_calls = file_search_tool_calls(&response_body.output);

        for output_item in &response_body.output {
            if let Some(item) = output_item_to_storage_item(output_item) {
                turn_items.push(item);
            }
        }

        if tool_calls.is_empty() {
            debug!(iteration, "no tool calls, returning final response");
            persist_response(
                state,
                &request,
                &response_body,
                original_previous_response_id.as_deref(),
                turn_items,
            )
            .await?;
            let final_json = serde_json::to_string(&response_body)
                .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_owned());
            return Ok((StatusCode::OK, [("content-type", "application/json")], final_json).into_response());
        }

        append_output_items_to_request(&mut request, &response_body.output);
        append_tool_outputs(state, &mut request, &mut turn_items, &vector_store_ids, &tool_calls).await;

        debug!(
            iteration,
            tool_calls = tool_calls.len(),
            "fed tool results back, continuing loop"
        );
    }

    warn!(
        max_iterations = state.max_iterations,
        "agentic loop reached max iterations"
    );
    Err(agentic_core::error::Error::MaxIterations {
        max_iterations: state.max_iterations,
    })
}

fn build_agentic_body(request: &ResponseRequest) -> Result<serde_json::Value, agentic_core::error::Error> {
    let mut body = serde_json::to_value(request)
        .map_err(|e| agentic_core::error::Error::Config(format!("failed to serialize request: {e}")))?;
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".to_owned(), serde_json::Value::Bool(false));
        if let Some(serde_json::Value::Array(tools)) = obj.get_mut("tools") {
            let had_file_search = tools
                .iter()
                .any(|t| t.get("type").and_then(serde_json::Value::as_str) == Some("file_search"));
            tools.retain(|t| t.get("type").and_then(serde_json::Value::as_str) != Some("file_search"));
            if had_file_search {
                tools.push(serde_json::json!({
                    "type": "function",
                    "name": "file_search",
                    "description": "Search uploaded files for relevant content. Use this when the user asks about documents or needs information from files.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant content in files"
                            }
                        },
                        "required": ["query"]
                    }
                }));
            }
        }
    }
    Ok(body)
}

async fn send_agentic_request(
    state: &AppState,
    client_headers: &HeaderMap,
    request: &ResponseRequest,
) -> Result<Result<ResponseBody, Response>, agentic_core::error::Error> {
    let body = build_agentic_body(request)?;
    let resp = build_vllm_request(state, client_headers)
        .json(&body)
        .send()
        .await
        .map_err(agentic_core::error::Error::Proxy)?;

    let status = resp.status();
    if !status.is_success() {
        let resp_body = resp.text().await.unwrap_or_default();
        return Ok(Err((
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            resp_body,
        )
            .into_response()));
    }

    resp.json().await.map(Ok).map_err(agentic_core::error::Error::Proxy)
}

fn file_search_tool_calls(output: &[VllmOutputItem]) -> Vec<(String, String)> {
    output
        .iter()
        .filter_map(|item| match item {
            VllmOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } if name == "file_search" => Some((call_id.clone(), arguments.clone())),
            _ => None,
        })
        .collect()
}

fn append_output_items_to_request(request: &mut ResponseRequest, output: &[VllmOutputItem]) {
    for output_item in output {
        request
            .input
            .push(serde_json::to_value(output_item).unwrap_or_default());
    }
}

async fn append_tool_outputs(
    state: &AppState,
    request: &mut ResponseRequest,
    turn_items: &mut Vec<InOutItem>,
    vector_store_ids: &[String],
    tool_calls: &[(String, String)],
) {
    for (call_id, arguments) in tool_calls {
        let query = extract_query(arguments);
        debug!(%call_id, %query, "executing file_search tool call");

        let results = execute_file_search(state, vector_store_ids, &query).await;
        let tool_output = serde_json::json!({
            "type": "function_call_output",
            "call_id": call_id,
            "output": serde_json::to_string(&results).unwrap_or_default()
        });
        if let Some(item) = input_value_to_storage_item(&tool_output) {
            turn_items.push(item);
        }
        request.input.push(tool_output);
    }
}

async fn hydrate_previous_response(
    state: &AppState,
    request: &mut ResponseRequest,
) -> Result<(), agentic_core::error::Error> {
    let Some(previous_response_id) = request.previous_response_id.clone() else {
        return Ok(());
    };

    let history = state.response_store.rehydrate(&previous_response_id).await?;
    let history_values = history
        .into_iter()
        .filter_map(|item| match item {
            InOutItem::Input(input) => serde_json::to_value(input).ok(),
            InOutItem::Output(output) => serde_json::to_value(output).ok(),
        })
        .collect();

    request.input.prepend(history_values);
    request.previous_response_id = None;
    Ok(())
}

async fn persist_response(
    state: &AppState,
    request: &ResponseRequest,
    response_body: &ResponseBody,
    previous_response_id: Option<&str>,
    turn_items: Vec<InOutItem>,
) -> Result<(), agentic_core::error::Error> {
    let store = request
        .rest
        .get("store")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    if !store {
        return Ok(());
    }

    let metadata = ResponseMetadata {
        model: request.model.clone(),
        previous_response_id: previous_response_id.map(str::to_owned),
        effective_tools: None,
        effective_tool_choice: ToolChoice::Auto,
        effective_instructions: request
            .rest
            .get("instructions")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
    };

    if let Some(conversation_id) = request.rest.get("conversation_id").and_then(serde_json::Value::as_str) {
        state.conversation_store.get_or_create(conversation_id).await?;
        state
            .conversation_store
            .persist(
                conversation_id,
                &response_body.id,
                previous_response_id,
                turn_items,
                &metadata,
            )
            .await?;
    } else {
        state
            .response_store
            .persist(&response_body.id, previous_response_id, turn_items, &metadata)
            .await?;
    }

    Ok(())
}

fn input_values_to_storage_items(input: &ResponseInput) -> Vec<InOutItem> {
    input
        .to_values()
        .iter()
        .filter_map(input_value_to_storage_item)
        .collect()
}

fn input_value_to_storage_item(value: &serde_json::Value) -> Option<InOutItem> {
    if let Some(text) = value.as_str() {
        return Some(InOutItem::Input(InputItem::Message(InputMessage {
            role: "user".to_owned(),
            content: InputMessageContent::Text(text.to_owned()),
        })));
    }

    let mut value = value.clone();
    if let Some(obj) = value.as_object_mut() {
        if !obj.contains_key("type") && obj.contains_key("role") && obj.contains_key("content") {
            obj.insert("type".to_owned(), serde_json::Value::String("message".to_owned()));
        }
    }

    serde_json::from_value::<InputItem>(value)
        .ok()
        .filter(|item| !matches!(item, InputItem::Unknown))
        .map(InOutItem::Input)
}

fn output_item_to_storage_item(item: &VllmOutputItem) -> Option<InOutItem> {
    match item {
        VllmOutputItem::Message { fields } => {
            let id = fields
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map_or_else(|| uuid7_str("msg_"), str::to_owned);
            let role = fields
                .get("role")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("assistant")
                .to_owned();
            let status = fields
                .get("status")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("completed")
                .to_owned();
            let content = fields
                .get("content")
                .cloned()
                .and_then(|v| serde_json::from_value(v).ok())
                .unwrap_or_default();
            Some(InOutItem::Output(OutputItem::Message(OutputMessage {
                id,
                role,
                status,
                content,
            })))
        }
        VllmOutputItem::FunctionCall {
            id,
            call_id,
            name,
            arguments,
            rest,
        } => Some(InOutItem::Output(OutputItem::FunctionCall(FunctionToolCall {
            id: id.clone(),
            call_id: call_id.clone(),
            name: name.clone(),
            arguments: arguments.clone(),
            status: rest
                .get("status")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("completed")
                .to_owned(),
        }))),
        VllmOutputItem::Other => None,
    }
}

async fn execute_file_search(state: &AppState, vector_store_ids: &[String], query: &str) -> Vec<SearchResult> {
    let mut all_results = Vec::new();
    for store_id in vector_store_ids {
        match state.vector_search.search(store_id, query).await {
            Ok(results) => all_results.extend(results),
            Err(e) => {
                warn!(%store_id, error = %e, "file_search failed for vector store");
            }
        }
    }
    all_results
}

fn extract_query(arguments: &str) -> String {
    serde_json::from_str::<serde_json::Value>(arguments)
        .ok()
        .and_then(|v| v.get("query").and_then(serde_json::Value::as_str).map(String::from))
        .unwrap_or_default()
}

fn build_vllm_request(state: &AppState, client_headers: &HeaderMap) -> reqwest::RequestBuilder {
    let url = format!("{}/v1/responses", state.proxy.config.llm_api_base);
    let mut req = state
        .proxy
        .non_stream_client
        .post(&url)
        .header("content-type", "application/json");

    if let Some(auth) = client_headers.get(http::header::AUTHORIZATION) {
        if let Ok(v) = auth.to_str() {
            req = req.header("authorization", v);
        }
    } else if let Some(key) = &state.proxy.config.openai_api_key {
        req = req.header("authorization", format!("Bearer {key}"));
    }

    req
}

fn json_error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        [("content-type", "application/json")],
        serde_json::json!({"error": {"message": message}}).to_string(),
    )
        .into_response()
}
