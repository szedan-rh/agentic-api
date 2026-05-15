use std::path::PathBuf;

use clap::Parser;
use praxis_filter::register_filters;

use agentic_api::filters::agentic_loop::AgenticLoopFilter;
use agentic_api::filters::responses_proxy::ResponsesProxyFilter;
use agentic_api::filters::state_hydration::StateHydrationFilter;
use agentic_api::filters::tool_dispatch::ToolDispatchFilter;

register_filters! {
    http "responses_proxy" => ResponsesProxyFilter::from_config,
    http "state_hydration" => StateHydrationFilter::from_config,
    http "agentic_loop" => AgenticLoopFilter::from_config,
    http "tool_dispatch" => ToolDispatchFilter::from_config,
}

#[derive(Parser)]
#[command(name = "agentic-api", about = "Stateful API gateway for vLLM Responses API")]
struct Cli {
    #[arg(long, short = 'c')]
    config: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    let config_path = praxis::resolve_config_path(cli.config.as_deref().and_then(|p| p.to_str()));
    let effective_path = config_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("config/agentic-api.yaml"));
    let config = praxis_core::config::Config::from_file(&effective_path).unwrap_or_else(|e| praxis::fatal(&e));

    praxis::init_tracing(&config).unwrap_or_else(|e| praxis::fatal(&e));

    let registry = custom_registry();
    praxis::run_server_with_registry(config, registry, config_path);
}
