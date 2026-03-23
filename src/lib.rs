//! # tokio-prompt-orchestrator
//!
//! Async LLM prompt orchestration layer for the Special Relativity in Financial
//! Modeling (SRFM) pipeline.
//!
//! ## Overview
//!
//! This crate provides the coordination, deduplication, cost-tracking, and
//! circuit-breaker subsystems for running multiple LLM inference workers
//! concurrently. It is the Rust runtime layer that sits beneath the C++ SRFM
//! signal-processing library.
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `web-api` | HTTP REST/SSE/WebSocket API via Axum |
//! | `mcp`     | Model Context Protocol server |
//! | `tui`     | Terminal UI dashboard via Ratatui |
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use tokio_prompt_orchestrator::{EchoWorker, ModelWorker};
//!
//! #[tokio::main]
//! async fn main() {
//!     let worker = EchoWorker::new();
//!     let tokens = worker.infer("Hello, world!").await.unwrap();
//!     println!("{}", tokens.join(""));
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod config;
pub mod workers;
pub mod worker;
pub mod relativistic_options;
pub mod twin_paradox;
pub mod doppler_effect;
pub mod holography;
pub mod string_theory;
pub mod superluminal;
pub mod dark_energy;

#[cfg(feature = "tui")]
pub mod tui;

#[cfg(feature = "web-api")]
pub mod web_api;

/// Top-level error type for all orchestrator operations.
///
/// Individual subsystem errors (coordination, routing, etc.) convert into
/// this type automatically via the `From` implementations on each module's
/// error enum.
#[derive(Debug, thiserror::Error)]
pub enum OrchestratorError {
    /// A network or HTTP request failed.
    #[error("request failed: {0}")]
    RequestFailed(String),

    /// The worker timed out before returning a result.
    #[error("worker timeout after {0:?}")]
    Timeout(std::time::Duration),

    /// The worker returned a non-UTF-8 or otherwise invalid response.
    #[error("invalid response: {0}")]
    InvalidResponse(String),

    /// An IO error occurred (e.g. when spawning a subprocess).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// A configuration error, such as a missing required environment variable.
    #[error("configuration error: {0}")]
    ConfigError(String),

    /// An inference error returned by a model worker backend.
    #[error("inference error: {0}")]
    Inference(String),

    /// A subsystem-specific error with a human-readable message.
    #[error("{0}")]
    Other(String),
}

pub mod metrics;
pub mod quantum;
pub mod frame_transformation;
pub mod hawking_radiation;
pub mod geodesic_signals;
pub mod proper_time;
pub mod gravitational_waves;
pub mod penrose;
pub mod holographic;
pub mod entanglement;
pub mod information_geometry;
pub mod causal_inference;
pub mod causal_graph;
pub mod tensor;
pub mod light_cone;
pub mod relativistic_momentum_factor;
pub mod spacetime_diagram;
pub mod relativistic_portfolio;
pub mod quantum_finance;
pub mod entropy_finance;
pub mod curved_spacetime;
pub mod wormhole;
pub mod uncertainty_principle;
pub mod decoherence;
pub mod many_worlds;
pub mod quantum_tunneling;
pub mod path_integral;
pub mod spin_network;
pub mod quantum_field;
pub mod entropy_dynamics;
pub mod gauge_field;
pub mod renormalization;
pub mod causal_structure;
pub mod quantum_gravity;
pub mod thermodynamic_market;

#[cfg(feature = "viz")]
pub mod viz;

// Re-export the most commonly used types at the crate root.
pub use worker::{AnthropicWorker, EchoWorker, LlamaCppWorker, ModelWorker, OpenAiWorker, VllmWorker};

/// A stable identifier for an agent/session.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct SessionId(String);

impl SessionId {
    /// Create a new `SessionId` from any string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Return the inner string value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A prompt request submitted to the pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PromptRequest {
    /// Session this request belongs to.
    pub session: SessionId,
    /// Unique request identifier.
    pub request_id: String,
    /// The prompt text to run inference on.
    pub input: String,
    /// Optional key-value metadata attached to the request.
    pub meta: Option<std::collections::HashMap<String, String>>,
}
