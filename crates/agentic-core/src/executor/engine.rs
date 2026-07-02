//! Stateful conversation executor.
//!
//! Exposes each step of the conversation pipeline as a public function so consumers
//! can compose them directly (e.g. as Praxis filters). [`ExecuteRequest`] is the
//! primary entry point; [`execute`] is a convenience shim for callers that don't
//! need per-request configuration.

use std::sync::Arc;

use async_stream::stream;
use either::Either;
use futures::future::join_all;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::warn;

use crate::executor::accumulator::ResponseAccumulator;
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::inference::{DONE_MARKER, call_inference, fetch_response_json};
use crate::executor::persist::persist_response;
use crate::executor::rehydrate::rehydrate_conversation;
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::tool::{ToolError, ToolOutput, ToolRegistry, ToolType};
use crate::types::io::output::{FunctionToolCall, WebSearchCall, WebSearchCallStatus, WebSearchSource};
use crate::types::io::{InputItem, OutputItem, ResponseUsage, ResponsesInput, ToolChoice};
use crate::types::request_response::{RequestPayload, ResponsePayload};
use crate::utils::common::serialize_to_string;

pub use crate::executor::inference::BoxStream;

const MAX_GATEWAY_TOOL_ROUNDS: usize = 10;

fn should_persist(ctx: &RequestContext) -> bool {
    ctx.original_request.store
        || ctx.original_request.previous_response_id.is_some()
        || ctx.original_request.conversation_id.is_some()
}

async fn fetch_blocking_payload(
    ctx: &RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
) -> ExecutorResult<ResponsePayload> {
    let url = exec_ctx.responses_url();
    // Non-streaming request: stream=false → full JSON body → from_json.
    let upstream_json =
        serialize_to_string(&ctx.enriched_request.to_upstream_request(false)).map_err(ExecutorError::JsonError)?;

    let body = fetch_response_json(upstream_json, &url, &exec_ctx.client, auth).await?;

    let acc = ResponseAccumulator::from_json(&body, ctx.conversation_id.as_deref())?;
    let mut payload = acc.finalize(
        &ctx.enriched_request.model,
        ctx.original_request.previous_response_id.as_deref(),
        ctx.original_request.instructions.as_deref(),
    );
    ctx.inject_ids(&mut payload);

    Ok(payload)
}

async fn fetch_stream_payload(
    ctx: &RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
) -> ExecutorResult<ResponsePayload> {
    let url = exec_ctx.responses_url();
    let upstream_json =
        serialize_to_string(&ctx.enriched_request.to_upstream_request(true)).map_err(ExecutorError::JsonError)?;
    let line_stream = Box::pin(call_inference(
        upstream_json,
        url,
        Arc::clone(&exec_ctx.client),
        auth.map(str::to_owned),
        exec_ctx.streaming_timeout,
    ));
    let acc = ResponseAccumulator::from_stream(line_stream, ctx.conversation_id.as_deref()).await?;
    let mut payload = acc.finalize(
        &ctx.enriched_request.model,
        ctx.original_request.previous_response_id.as_deref(),
        ctx.original_request.instructions.as_deref(),
    );
    ctx.inject_ids(&mut payload);
    Ok(payload)
}

fn append_input_item(input: &mut ResponsesInput, item: InputItem) {
    match input {
        ResponsesInput::Items(items) => items.push(item),
        ResponsesInput::Text(text) => {
            let text_input = ResponsesInput::Text(std::mem::take(text));
            let mut items = Vec::<InputItem>::from(&text_input);
            items.push(item);
            *input = ResponsesInput::Items(items);
        }
    }
}

fn append_output_items_to_input(input: &mut ResponsesInput, output_items: &[OutputItem]) {
    for output_item in output_items {
        let input_item = match output_item {
            OutputItem::Message(message) => Some(InputItem::Message(message.clone().into())),
            OutputItem::FunctionCall(call) => Some(InputItem::FunctionCall(call.clone())),
            OutputItem::Reasoning(reasoning) => Some(InputItem::Reasoning(reasoning.clone())),
            OutputItem::WebSearchCall(_) | OutputItem::Unknown => None,
        };
        if let Some(input_item) = input_item {
            append_input_item(input, input_item);
        }
    }
}

fn append_tool_outputs(ctx: &mut RequestContext, tool_outputs: Vec<InputItem>) {
    for output in tool_outputs {
        ctx.new_input_items.push(output.clone());
        append_input_item(&mut ctx.enriched_request.input, output);
    }
}

fn function_calls(output_items: &[OutputItem]) -> Vec<FunctionToolCall> {
    output_items
        .iter()
        .filter_map(|item| match item {
            OutputItem::FunctionCall(call) => Some(call.clone()),
            _ => None,
        })
        .collect()
}

fn is_gateway_owned_call(call: &FunctionToolCall, registry: &ToolRegistry) -> bool {
    registry
        .lookup(&call.name)
        .is_some_and(|entry| entry.tool_type != ToolType::Function)
}

fn append_gateway_calls_to_new_input(ctx: &mut RequestContext, output_items: &[OutputItem], registry: &ToolRegistry) {
    ctx.new_input_items.extend(output_items.iter().filter_map(|item| {
        let OutputItem::FunctionCall(call) = item else {
            return None;
        };
        is_gateway_owned_call(call, registry).then(|| InputItem::FunctionCall(call.clone()))
    }));
}

fn public_output_items(
    output_items: Vec<OutputItem>,
    registry: &ToolRegistry,
    gateway_results: &[GatewayCallResult],
) -> Vec<OutputItem> {
    output_items
        .into_iter()
        .map(|item| match item {
            OutputItem::FunctionCall(call) if is_gateway_owned_call(&call, registry) => gateway_results
                .iter()
                .find(|result| result.call.call_id == call.call_id)
                .and_then(|result| result.public_output.clone())
                .unwrap_or(OutputItem::FunctionCall(call)),
            other => other,
        })
        .collect()
}

fn add_usage(total: ResponseUsage, usage: ResponseUsage) -> ResponseUsage {
    ResponseUsage {
        input_tokens: total.input_tokens.saturating_add(usage.input_tokens),
        output_tokens: total.output_tokens.saturating_add(usage.output_tokens),
        total_tokens: total.total_tokens.saturating_add(usage.total_tokens),
        input_tokens_details: crate::types::io::InputTokenDetails {
            cached_tokens: total
                .input_tokens_details
                .cached_tokens
                .saturating_add(usage.input_tokens_details.cached_tokens),
        },
        output_tokens_details: crate::types::io::OutputTokenDetails {
            reasoning_tokens: total
                .output_tokens_details
                .reasoning_tokens
                .saturating_add(usage.output_tokens_details.reasoning_tokens),
        },
    }
}

fn accumulate_usage(total: &mut Option<ResponseUsage>, usage: Option<ResponseUsage>) {
    if let Some(usage) = usage {
        *total = Some(total.map_or(usage, |current| add_usage(current, usage)));
    }
}

struct GatewayCallDispatch {
    call: FunctionToolCall,
    tool_type: ToolType,
    config: Value,
    executor: Arc<dyn crate::tool::GatewayExecutor>,
}

struct GatewayCallExecution {
    call: FunctionToolCall,
    tool_type: ToolType,
    output: Result<ToolOutput, ToolError>,
}

#[derive(Clone)]
struct GatewayCallResult {
    call: FunctionToolCall,
    input_item: InputItem,
    public_output: Option<OutputItem>,
}

struct GatewayCallEventPlan {
    call_id: String,
    output_index: u32,
    started_output: Option<OutputItem>,
}

fn gateway_dispatches(
    calls: &[FunctionToolCall],
    registry: &ToolRegistry,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<Vec<GatewayCallDispatch>> {
    registry
        .gateway_owned(calls)
        .into_iter()
        .map(|call| {
            let entry = registry.lookup(&call.name).ok_or_else(|| {
                ToolError::Config(format!(
                    "gateway tool '{}' was not found in the request registry",
                    call.name
                ))
            })?;
            let executor = exec_ctx
                .gateway_executors
                .get(entry.tool_type)
                .ok_or_else(|| ToolError::Config(format!("no gateway executor registered for tool '{}'", call.name)))?;
            Ok(GatewayCallDispatch {
                call: call.clone(),
                tool_type: entry.tool_type,
                config: entry.config.clone(),
                executor,
            })
        })
        .collect::<Result<Vec<_>, ToolError>>()
        .map_err(ExecutorError::from)
}

async fn execute_gateway_dispatch(dispatch: GatewayCallDispatch) -> GatewayCallExecution {
    let GatewayCallDispatch {
        call,
        tool_type,
        config,
        executor,
    } = dispatch;
    let output = executor
        .execute(&call.call_id, &call.name, &call.arguments, &config)
        .await;
    GatewayCallExecution {
        call,
        tool_type,
        output,
    }
}

fn execution_error_output(call: &FunctionToolCall, message: &str) -> ExecutorResult<ToolOutput> {
    let output = serialize_to_string(&serde_json::json!({ "error": message })).map_err(ExecutorError::JsonError)?;
    Ok(ToolOutput {
        call_id: call.call_id.clone(),
        output,
    })
}

fn gateway_public_output(
    tool_type: ToolType,
    call: &FunctionToolCall,
    output: &ToolOutput,
    status: WebSearchCallStatus,
) -> Option<OutputItem> {
    match tool_type {
        ToolType::WebSearch => Some(web_search_output_item(call, output, status)),
        ToolType::Function | ToolType::Mcp | ToolType::FileSearch | ToolType::CodeInterpreter => None,
    }
}

async fn execute_gateway_calls(
    calls: &[FunctionToolCall],
    registry: &ToolRegistry,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<Vec<GatewayCallResult>> {
    let dispatches = gateway_dispatches(calls, registry, exec_ctx)?;
    let executions = join_all(dispatches.into_iter().map(execute_gateway_dispatch)).await;
    let mut results = Vec::with_capacity(executions.len());

    for execution in executions {
        let GatewayCallExecution {
            call,
            tool_type,
            output,
        } = execution;
        let (output, status) = match output {
            Ok(output) => (output, WebSearchCallStatus::Completed),
            Err(ToolError::Execution(message)) => {
                (execution_error_output(&call, &message)?, WebSearchCallStatus::Failed)
            }
            Err(err @ ToolError::Config(_)) => return Err(err.into()),
        };
        let public_output = gateway_public_output(tool_type, &call, &output, status);
        results.push(GatewayCallResult {
            call,
            input_item: InputItem::FunctionCallOutput(output.into()),
            public_output,
        });
    }

    Ok(results)
}

fn web_search_output_item(call: &FunctionToolCall, output: &ToolOutput, status: WebSearchCallStatus) -> OutputItem {
    let parsed_output = serde_json::from_str::<Value>(&output.output).ok();
    let query = parsed_output
        .as_ref()
        .and_then(|value| clean_json_str(value.get("query")))
        .or_else(|| web_search_query_from_arguments(&call.arguments))
        .unwrap_or_default();
    let sources = parsed_output
        .as_ref()
        .map(web_search_sources_from_output)
        .unwrap_or_default();
    OutputItem::WebSearchCall(WebSearchCall::new(web_search_call_id(call), status, query, sources))
}

fn started_web_search_output_item(call: &FunctionToolCall) -> OutputItem {
    OutputItem::WebSearchCall(WebSearchCall::new(
        web_search_call_id(call),
        WebSearchCallStatus::InProgress,
        web_search_query_from_arguments(&call.arguments).unwrap_or_default(),
        Vec::new(),
    ))
}

fn web_search_call_id(call: &FunctionToolCall) -> String {
    if let Some(suffix) = call.id.strip_prefix("fc_").filter(|suffix| !suffix.is_empty()) {
        return format!("ws_{suffix}");
    }
    if let Some(suffix) = call.call_id.strip_prefix("call_").filter(|suffix| !suffix.is_empty()) {
        return format!("ws_{suffix}");
    }
    crate::utils::uuid7_str("ws_")
}

fn web_search_query_from_arguments(arguments: &str) -> Option<String> {
    let args = serde_json::from_str::<Value>(arguments).ok()?;
    clean_json_str(args.get("query"))
}

fn web_search_sources_from_output(output: &Value) -> Vec<WebSearchSource> {
    ["web", "news"]
        .into_iter()
        .filter_map(|section| output.get("results")?.get(section)?.as_array())
        .flat_map(|results| results.iter())
        .filter_map(web_search_source_from_result)
        .collect()
}

fn web_search_source_from_result(result: &Value) -> Option<WebSearchSource> {
    let url = clean_json_str(result.get("url"))?;
    Some(WebSearchSource {
        url,
        title: clean_json_str(result.get("title")),
    })
}

fn clean_json_str(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn gateway_event_plans(
    output_items: &[OutputItem],
    registry: &ToolRegistry,
    output_offset: usize,
) -> Vec<GatewayCallEventPlan> {
    let mut output_index = output_offset;
    let mut plans = Vec::new();
    for item in output_items {
        if let OutputItem::FunctionCall(call) = item
            && let Some(entry) = registry.lookup(&call.name)
            && entry.tool_type != ToolType::Function
        {
            plans.push(GatewayCallEventPlan {
                call_id: call.call_id.clone(),
                output_index: u32::try_from(output_index).unwrap_or(u32::MAX),
                started_output: match entry.tool_type {
                    ToolType::WebSearch => Some(started_web_search_output_item(call)),
                    ToolType::Function | ToolType::Mcp | ToolType::FileSearch | ToolType::CodeInterpreter => None,
                },
            });
        }
        output_index = output_index.saturating_add(1);
    }
    plans
}

fn emit_sse_json(sender: &mpsc::UnboundedSender<String>, event: &Value) -> ExecutorResult<()> {
    let event_json = serialize_to_string(&event).map_err(ExecutorError::JsonError)?;
    let _ = sender.send(format!("data: {event_json}\n\n"));
    Ok(())
}

fn output_item_value(item: &OutputItem) -> ExecutorResult<Value> {
    serde_json::to_value(item).map_err(ExecutorError::JsonError)
}

fn emit_gateway_start_events(
    plans: &[GatewayCallEventPlan],
    stream_events: Option<&mpsc::UnboundedSender<String>>,
) -> ExecutorResult<()> {
    let Some(sender) = stream_events else {
        return Ok(());
    };
    for plan in plans {
        let Some(output_item) = &plan.started_output else {
            continue;
        };
        let OutputItem::WebSearchCall(web_search_call) = output_item else {
            continue;
        };
        let item = output_item_value(output_item)?;
        let added_event = serde_json::json!({
                "type": "response.output_item.added",
                "output_index": plan.output_index,
                "item": item
        });
        emit_sse_json(sender, &added_event)?;
        let in_progress_event = serde_json::json!({
                "type": "response.web_search_call.in_progress",
                "item_id": web_search_call.id,
                "output_index": plan.output_index
        });
        emit_sse_json(sender, &in_progress_event)?;
        let searching_event = serde_json::json!({
                "type": "response.web_search_call.searching",
                "item_id": web_search_call.id,
                "output_index": plan.output_index
        });
        emit_sse_json(sender, &searching_event)?;
    }
    Ok(())
}

fn emit_gateway_completed_events(
    results: &[GatewayCallResult],
    plans: &[GatewayCallEventPlan],
    stream_events: Option<&mpsc::UnboundedSender<String>>,
) -> ExecutorResult<()> {
    let Some(sender) = stream_events else {
        return Ok(());
    };
    for result in results {
        let Some(OutputItem::WebSearchCall(web_search_call)) = &result.public_output else {
            continue;
        };
        let output_index = plans
            .iter()
            .find(|plan| plan.call_id == result.call.call_id)
            .map_or(0, |plan| plan.output_index);
        let output_item = OutputItem::WebSearchCall(web_search_call.clone());
        let item = output_item_value(&output_item)?;
        let completed_event = serde_json::json!({
                "type": "response.web_search_call.completed",
                "item_id": web_search_call.id,
                "output_index": output_index,
                "item": item.clone()
        });
        emit_sse_json(sender, &completed_event)?;
        let done_event = serde_json::json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item
        });
        emit_sse_json(sender, &done_event)?;
    }
    Ok(())
}

async fn run_until_gateway_tools_complete(
    mut ctx: RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
    stream_upstream: bool,
    stream_events: Option<&mpsc::UnboundedSender<String>>,
) -> ExecutorResult<(ResponsePayload, RequestContext)> {
    let registry = ctx
        .enriched_request
        .tools
        .as_ref()
        .map_or_else(ToolRegistry::default, |tools| ToolRegistry::build(tools));
    let mut combined_output = Vec::new();
    let mut combined_usage = None;

    for _ in 0..MAX_GATEWAY_TOOL_ROUNDS {
        let mut payload = if stream_upstream {
            fetch_stream_payload(&ctx, exec_ctx, auth).await?
        } else {
            fetch_blocking_payload(&ctx, exec_ctx, auth).await?
        };
        accumulate_usage(&mut combined_usage, payload.usage);
        let current_output = std::mem::take(&mut payload.output);
        let calls = function_calls(&current_output);
        let has_client_owned_calls = !registry.client_owned(&calls).is_empty();
        let event_plans = gateway_event_plans(&current_output, &registry, combined_output.len());
        emit_gateway_start_events(&event_plans, stream_events)?;
        let gateway_results = execute_gateway_calls(&calls, &registry, exec_ctx).await?;
        emit_gateway_completed_events(&gateway_results, &event_plans, stream_events)?;
        let public_output = public_output_items(current_output.clone(), &registry, &gateway_results);

        if has_client_owned_calls {
            combined_output.extend(public_output);
            append_gateway_calls_to_new_input(&mut ctx, &current_output, &registry);
            append_tool_outputs(
                &mut ctx,
                gateway_results.into_iter().map(|result| result.input_item).collect(),
            );
            payload.output = combined_output;
            payload.usage = combined_usage;
            ctx.inject_ids(&mut payload);
            return Ok((payload, ctx));
        }

        if gateway_results.is_empty() {
            combined_output.extend(public_output);
            payload.output = combined_output;
            payload.usage = combined_usage;
            ctx.inject_ids(&mut payload);
            return Ok((payload, ctx));
        }

        combined_output.extend(public_output);
        ctx.enriched_request.tool_choice = ToolChoice::Auto;
        append_output_items_to_input(&mut ctx.enriched_request.input, &current_output);
        append_gateway_calls_to_new_input(&mut ctx, &current_output, &registry);
        append_tool_outputs(
            &mut ctx,
            gateway_results.into_iter().map(|result| result.input_item).collect(),
        );
    }

    Err(ExecutorError::InvalidRequest(format!(
        "gateway tool execution exceeded {MAX_GATEWAY_TOOL_ROUNDS} rounds"
    )))
}

async fn run_blocking(
    ctx: RequestContext,
    exec_ctx: &ExecutionContext,
    auth: Option<&str>,
) -> ExecutorResult<ResponsePayload> {
    let (payload, ctx) = run_until_gateway_tools_complete(ctx, exec_ctx, auth, false, None).await?;

    if should_persist(&ctx) {
        let ch = exec_ctx.conv_handler.clone();
        let rh = exec_ctx.resp_handler.clone();
        if let Err(e) = persist_response(payload.clone(), ctx, ch, rh).await {
            warn!("persist failed: {e}");
        }
    }

    Ok(payload)
}

fn run_stream(ctx: RequestContext, exec_ctx: Arc<ExecutionContext>, auth: Option<String>) -> BoxStream {
    Box::pin(stream! {
        let should_persist = should_persist(&ctx);
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let exec_ctx_for_run = Arc::clone(&exec_ctx);
        let mut run_handle = tokio::spawn(async move {
            run_until_gateway_tools_complete(
                ctx,
                exec_ctx_for_run.as_ref(),
                auth.as_deref(),
                true,
                Some(&event_tx),
            )
            .await
        });

        loop {
            tokio::select! {
                Some(event) = event_rx.recv() => {
                    yield event;
                }
                result = &mut run_handle => {
                    while let Ok(event) = event_rx.try_recv() {
                        yield event;
                    }
                    match result {
                        Err(e) => {
                            yield format!("data: {{\"error\": \"stream task failed: {e}\"}}\n\n");
                            yield DONE_MARKER.to_string();
                        }
                        Ok(Err(e)) => {
                            yield format!("data: {{\"error\": \"{e}\"}}\n\n");
                            yield DONE_MARKER.to_string();
                        }
                        Ok(Ok((payload, ctx))) => {
                            yield payload.as_responses_chunk();
                            yield DONE_MARKER.to_string();

                            if should_persist {
                                let ch = exec_ctx.conv_handler.clone();
                                let rh = exec_ctx.resp_handler.clone();
                                if let Err(e) = persist_response(payload, ctx, ch, rh).await {
                                    warn!("persist failed: {e}");
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
    })
}

/// Create a new conversation and return its data.
///
/// Exposes the conversation-creation step as a standalone function so callers
/// (e.g. `agentic-server`, Praxis filters, or tests) can pre-create a
/// conversation before submitting response turns.
///
/// # Errors
/// Returns [`ExecutorError`] if the conversation store is unavailable.
pub async fn create_conversation(exec_ctx: &ExecutionContext) -> ExecutorResult<crate::ConversationData> {
    exec_ctx.conv_handler.create().await
}

/// Builder for a stateful conversation turn.
///
/// ```ignore
/// ExecuteRequest::new(payload, exec_ctx).with_auth(token).run().await
/// ```
pub struct ExecuteRequest {
    payload: RequestPayload,
    exec_ctx: Arc<ExecutionContext>,
    client_auth: Option<String>,
}

impl ExecuteRequest {
    #[must_use]
    pub fn new(payload: RequestPayload, exec_ctx: Arc<ExecutionContext>) -> Self {
        Self {
            payload,
            exec_ctx,
            client_auth: None,
        }
    }

    /// Override the bearer token for this request only; does not touch the shared [`ExecutionContext`].
    #[must_use]
    pub fn with_auth(mut self, token: Option<String>) -> Self {
        self.client_auth = token;
        self
    }

    /// Execute one stateful conversation turn.
    ///
    /// Returns `Either::Left(ResponsePayload)` for non-streaming requests, or
    /// `Either::Right(BoxStream)` for streaming, each yielded `String` is an SSE
    /// line ready to forward to the client.
    ///
    /// # Errors
    /// Returns [`ExecutorError`] if rehydration or (non-streaming) LLM inference fails.
    pub async fn run(self) -> ExecutorResult<Either<ResponsePayload, BoxStream>> {
        let ctx = rehydrate_conversation(self.payload, &self.exec_ctx).await?;
        if ctx.original_request.stream {
            Ok(Either::Right(run_stream(ctx, self.exec_ctx, self.client_auth)))
        } else {
            Ok(Either::Left(
                run_blocking(ctx, &self.exec_ctx, self.client_auth.as_deref()).await?,
            ))
        }
    }
}

/// Execute one stateful conversation turn.
///
/// Thin shim over [`ExecuteRequest`] for callers that don't need per-request auth override.
///
/// # Errors
/// Returns [`ExecutorError`] if rehydration or (non-streaming) LLM inference fails.
pub async fn execute(
    request: RequestPayload,
    exec_ctx: Arc<ExecutionContext>,
) -> ExecutorResult<Either<ResponsePayload, BoxStream>> {
    ExecuteRequest::new(request, exec_ctx).run().await
}
