use async_trait::async_trait;
use bytes::Bytes;
use praxis_filter::{
    BodyAccess, BodyMode, FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config,
};
use serde::Deserialize;
use tracing::debug;

#[derive(Deserialize)]
struct Config {
    #[serde(default = "default_max_iterations")]
    max_iterations: u32,
}

fn default_max_iterations() -> u32 {
    10
}

pub struct AgenticLoopFilter {
    max_iterations: u32,
}

impl AgenticLoopFilter {
    /// # Errors
    ///
    /// Returns an error if the config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: Config = parse_filter_config("agentic_loop", config)?;
        Ok(Box::new(Self {
            max_iterations: cfg.max_iterations,
        }))
    }
}

#[async_trait]
impl HttpFilter for AgenticLoopFilter {
    fn name(&self) -> &'static str {
        "agentic_loop"
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
                let has_tool_calls = v.get("output").and_then(|o| o.as_array()).is_some_and(|arr| {
                    arr.iter()
                        .any(|item| item.get("type").and_then(|t| t.as_str()) == Some("function_call"))
                });

                if has_tool_calls {
                    debug!(
                        max_iterations = self.max_iterations,
                        "agentic_loop: tool call detected in response, would re-enter inference loop (stub)"
                    );
                }
            }
        }

        Ok(FilterAction::Continue)
    }
}
