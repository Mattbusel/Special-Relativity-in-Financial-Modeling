//! llama.cpp HTTP server worker.

use super::ModelWorker;
use crate::OrchestratorError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// llama.cpp server request payload.
#[derive(Debug, Serialize)]
struct LlamaCppRequest {
    prompt: String,
    n_predict: i32,
    temperature: f32,
    stop: Vec<String>,
}

/// llama.cpp server response.
#[derive(Debug, Deserialize)]
struct LlamaCppResponse {
    content: String,
}

/// llama.cpp HTTP server worker.
///
/// Connects to a llama.cpp server instance.
/// Server URL can be set via `LLAMA_CPP_URL` environment variable
/// or defaults to `http://localhost:8080`.
///
/// ## Example
///
/// ```no_run
/// use tokio_prompt_orchestrator::LlamaCppWorker;
/// use std::sync::Arc;
///
/// let worker = Arc::new(
///     LlamaCppWorker::new()
///         .with_url("http://localhost:8080")
///         .with_max_tokens(512)
/// );
/// ```
pub struct LlamaCppWorker {
    pub(crate) client: reqwest::Client,
    pub(crate) url: String,
    pub(crate) max_tokens: i32,
    pub(crate) temperature: f32,
    pub(crate) timeout: Duration,
}

impl LlamaCppWorker {
    /// Create a new llama.cpp worker.
    ///
    /// Reads server URL from `LLAMA_CPP_URL` environment variable,
    /// or defaults to `http://localhost:8080`.
    pub fn new() -> Self {
        let url = std::env::var("LLAMA_CPP_URL")
            .unwrap_or_else(|_| "http://localhost:8080".to_string());

        Self {
            client: reqwest::Client::new(),
            url,
            max_tokens: 256,
            temperature: 0.8,
            timeout: Duration::from_secs(30),
        }
    }

    /// Set server URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Set maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: i32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for LlamaCppWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelWorker for LlamaCppWorker {
    async fn infer(&self, prompt: &str) -> Result<Vec<String>, OrchestratorError> {
        let request = LlamaCppRequest {
            prompt: prompt.to_string(),
            n_predict: self.max_tokens,
            temperature: self.temperature,
            stop: vec!["</s>".to_string(), "Human:".to_string()],
        };

        let response = self
            .client
            .post(format!("{}/completion", self.url))
            .timeout(self.timeout)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                OrchestratorError::Inference(format!("llama.cpp request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(OrchestratorError::Inference(format!(
                "llama.cpp error {}: {}",
                status, error_text
            )));
        }

        let api_response: LlamaCppResponse = response.json().await.map_err(|e| {
            OrchestratorError::Inference(format!("Failed to parse response: {}", e))
        })?;

        // Split response into tokens.
        let tokens: Vec<String> = api_response
            .content
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        Ok(tokens)
    }
}
