//! # srfm-orchestrator
//!
//! CLI entry point for the Special Relativity in Financial Modeling
//! async LLM prompt orchestration pipeline.
//!
//! ## Usage
//!
//! ```text
//! srfm-orchestrator [OPTIONS]
//! ```
//!
//! ## Options
//!
//! | Flag | Description |
//! |------|-------------|
//! | `--mock` | Run the TUI dashboard in mock-data mode (no live workers required) |
//! | `--metrics-url <URL>` | Connect the TUI to a live Prometheus metrics endpoint |
//! | `--web` | Start the HTTP/WebSocket API server (requires `web-api` feature) |
//! | `--host <HOST>` | Bind host for the web API (default: `0.0.0.0`) |
//! | `--port <PORT>` | Bind port for the web API (default: `8080`) |

use std::process;

fn main() {
    // Initialise a minimal tracing subscriber so any library-level spans are
    // captured even when no specific subcommand is active.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("tokio_prompt_orchestrator=info".parse().expect("valid directive")),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();

    // Print help when requested or when no arguments are supplied.
    if args.len() == 1 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }

    // Dispatch sub-commands.
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("srfm-orchestrator {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    if args.iter().any(|a| a == "--mock" || a == "--tui") {
        #[cfg(feature = "tui")]
        {
            eprintln!("TUI feature enabled. Launching dashboard…");
            // The TUI event loop is implemented in `src/tui/mod.rs`.
            // Call it on a Tokio runtime.
            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            if let Err(e) = rt.block_on(tokio_prompt_orchestrator::tui::run_mock()) {
                eprintln!("TUI error: {e}");
                process::exit(1);
            }
        }
        #[cfg(not(feature = "tui"))]
        {
            eprintln!("Error: the `tui` feature is not enabled. Rebuild with `--features tui`.");
            process::exit(1);
        }
        return;
    }

    if args.iter().any(|a| a == "--web") {
        #[cfg(feature = "web-api")]
        {
            let host = flag_value(&args, "--host").unwrap_or("0.0.0.0");
            let port: u16 = flag_value(&args, "--port")
                .and_then(|s| s.parse().ok())
                .unwrap_or(8080);

            eprintln!("Starting web API on {}:{}", host, port);

            let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
            let (tx, _rx) = tokio::sync::mpsc::channel(256);
            let cfg = tokio_prompt_orchestrator::web_api::ServerConfig {
                host: host.to_string(),
                port,
                ..Default::default()
            };
            if let Err(e) = rt.block_on(tokio_prompt_orchestrator::web_api::start_server(cfg, tx))
            {
                eprintln!("Web API error: {e}");
                process::exit(1);
            }
        }
        #[cfg(not(feature = "web-api"))]
        {
            eprintln!(
                "Error: the `web-api` feature is not enabled. Rebuild with `--features web-api`."
            );
            process::exit(1);
        }
        return;
    }

    eprintln!("Unknown arguments. Run with --help for usage.");
    process::exit(1);
}

/// Returns the value following `flag` in `args`, if present.
fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].as_str())
}

fn print_help() {
    println!(
        "srfm-orchestrator {ver}
Async LLM prompt orchestration pipeline for Special Relativity in Financial Modeling.

USAGE:
    srfm-orchestrator [OPTIONS]

OPTIONS:
    --mock, --tui            Launch the TUI dashboard in mock-data mode
    --metrics-url <URL>      Connect TUI to a live Prometheus endpoint
    --web                    Start the HTTP/WebSocket API server
    --host <HOST>            Web API bind host (default: 0.0.0.0)
    --port <PORT>            Web API bind port (default: 8080)
    -V, --version            Print version
    -h, --help               Print this help

ENVIRONMENT VARIABLES:
    API_KEY                  Bearer token for web API authentication
    ALLOWED_ORIGINS          Comma-separated CORS allowed origins
    OPENAI_API_KEY           Required for OpenAI worker
    ANTHROPIC_API_KEY        Required for Anthropic worker
    LLAMA_CPP_URL            llama.cpp server URL (default: http://localhost:8080)
    VLLM_URL                 vLLM server URL (default: http://localhost:8000)
    RUST_LOG                 Tracing log level (e.g. info, debug)
",
        ver = env!("CARGO_PKG_VERSION")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flag_value_found() {
        let args: Vec<String> = vec!["prog".into(), "--port".into(), "9090".into()];
        assert_eq!(flag_value(&args, "--port"), Some("9090"));
    }

    #[test]
    fn test_flag_value_not_found() {
        let args: Vec<String> = vec!["prog".into(), "--host".into(), "localhost".into()];
        assert_eq!(flag_value(&args, "--port"), None);
    }

    #[test]
    fn test_flag_value_empty_args() {
        let args: Vec<String> = vec![];
        assert_eq!(flag_value(&args, "--port"), None);
    }
}
