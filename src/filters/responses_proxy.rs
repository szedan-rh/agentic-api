use std::sync::Arc;

use async_trait::async_trait;
use praxis_core::connectivity::{ConnectionOptions, Upstream};
use praxis_filter::{FilterAction, FilterError, HttpFilter, HttpFilterContext, parse_filter_config};
use serde::Deserialize;
use tracing::debug;

use crate::config::normalize_base_url;

#[derive(Deserialize)]
struct Config {
    vllm_base_url: String,
    #[serde(default)]
    openai_api_key: Option<String>,
}

pub struct ResponsesProxyFilter {
    vllm_address: Arc<str>,
    openai_api_key: Option<String>,
}

impl ResponsesProxyFilter {
    /// # Errors
    ///
    /// Returns an error if the config is invalid.
    pub fn from_config(config: &serde_yaml::Value) -> Result<Box<dyn HttpFilter>, FilterError> {
        let cfg: Config = parse_filter_config("responses_proxy", config)?;

        let base = normalize_base_url(&cfg.vllm_base_url);
        let address = base
            .strip_prefix("https://")
            .or_else(|| base.strip_prefix("http://"))
            .unwrap_or(&base);

        Ok(Box::new(Self {
            vllm_address: Arc::from(address),
            openai_api_key: cfg.openai_api_key.filter(|k| !k.trim().is_empty()),
        }))
    }
}

#[async_trait]
impl HttpFilter for ResponsesProxyFilter {
    fn name(&self) -> &'static str {
        "responses_proxy"
    }

    async fn on_request(&self, ctx: &mut HttpFilterContext<'_>) -> Result<FilterAction, FilterError> {
        debug!("responses_proxy: routing to {}", self.vllm_address);

        ctx.upstream = Some(Upstream {
            address: Arc::clone(&self.vllm_address),
            tls: None,
            connection: Arc::new(ConnectionOptions::default()),
        });

        ctx.rewritten_path = Some("/v1/responses".to_owned());

        if !ctx.request.headers.contains_key(http::header::AUTHORIZATION) {
            if let Some(key) = &self.openai_api_key {
                ctx.extra_request_headers
                    .push(("authorization".into(), format!("Bearer {key}")));
            }
        }

        Ok(FilterAction::Continue)
    }
}
