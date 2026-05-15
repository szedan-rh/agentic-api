use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext};
use tracing::debug;

pub struct ToolDispatchFilter;

impl ToolDispatchFilter {
    /// # Errors
    ///
    /// Returns an error if the config is invalid.
    pub fn from_config(_config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        Ok(Box::new(Self))
    }
}

#[async_trait]
impl HttpFilter for ToolDispatchFilter {
    fn name(&self) -> &'static str {
        "tool_dispatch"
    }

    fn response_body_access(&self) -> BodyAccess {
        BodyAccess::ReadOnly
    }

    fn response_body_mode(&self) -> BodyMode {
        BodyMode::StreamBuffer {
            max_bytes: Some(10 * 1024 * 1024),
        }
    }

    async fn on_request(&self, _ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        Ok(FilterAction::Continue)
    }

    fn on_response_body(
        &self,
        _ctx: &mut HttpFilterContext<'_>,
        body: &mut Option<Bytes>,
        end_of_stream: bool,
    ) -> Result<FilterAction, FilterError> {
        if !end_of_stream {
            return Ok(FilterAction::Continue);
        }

        if let Some(data) = body.as_ref() {
            if let Ok(v) = serde_json::from_slice::<serde_json::Value>(data) {
                if let Some(output) = v.get("output").and_then(|o| o.as_array()) {
                    for item in output {
                        if item.get("type").and_then(|t| t.as_str()) == Some("function_call") {
                            let name = item.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                            debug!(tool_name = name, "tool_dispatch: would execute tool call (stub)");
                        }
                    }
                }
            }
        }

        Ok(FilterAction::Continue)
    }
}
