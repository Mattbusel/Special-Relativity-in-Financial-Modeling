//! vLLM inference server worker.

use super::ModelWorker;
use crate::OrchestratorError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// vLLM server request payload.
#[derive(Debug, Serialize)]
struct VllmRequest {
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
}

/// vLLM server response.
#[derive(Debug, Deserialize)]
struct VllmResponse {
    text: Vec<String>,
}

/// vLLM inference server worker.
///
/// Connects to a vLLM server instance.
/// Server URL can be set via `VLLM_URL` environment variable
/// or defaults to `http://localhost:8000`.
///
/// ## Example
///
/// ```no_run
/// use tokio_prompt_orchestrator::VllmWorker;
/// use std::sync::Arc;
///
/// let worker = Arc::new(
///     VllmWorker::new()
///         .with_url("http://localhost:8000")
///         .with_max_tokens(1024)
/// );
/// ```
pub struct VllmWorker {
    pub(crate) client: reqwest::Client,
    pub(crate) url: String,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) timeout: Duration,
}

impl VllmWorker {
    /// Create a new vLLM worker.
    ///
    /// Reads server URL from `VLLM_URL` environment variable,
    /// or defaults to `http://localhost:8000`.
    pub fn new() -> Self {
        let url =
            std::env::var("VLLM_URL").unwrap_or_else(|_| "http://localhost:8000".to_string());

        Self {
            client: reqwest::Client::new(),
            url,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            timeout: Duration::from_secs(60),
        }
    }

    /// Set server URL.
    pub fn with_url(mut self, url: impl Into<String>) -> Self {
        self.url = url.into();
        self
    }

    /// Set maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top_p sampling parameter.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for VllmWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelWorker for VllmWorker {
    async fn infer(&self, prompt: &str) -> Result<Vec<String>, OrchestratorError> {
        let request = VllmRequest {
            prompt: prompt.to_string(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
        };

        let response = self
            .client
            .post(format!("{}/generate", self.url))
            .timeout(self.timeout)
            .json(&request)
            .send()
            .await
            .map_err(|e| OrchestratorError::Inference(format!("vLLM request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(OrchestratorError::Inference(format!(
                "vLLM error {}: {}",
                status, error_text
            )));
        }

        let api_response: VllmResponse = response.json().await.map_err(|e| {
            OrchestratorError::Inference(format!("Failed to parse response: {}", e))
        })?;

        if api_response.text.is_empty() {
            return Err(OrchestratorError::Inference(
                "Empty response from vLLM".to_string(),
            ));
        }

        // Split response into tokens.
        let tokens: Vec<String> = api_response.text[0]
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        Ok(tokens)
    }
}
