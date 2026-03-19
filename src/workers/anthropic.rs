//! Anthropic Claude API worker.

use super::ModelWorker;
use crate::OrchestratorError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Anthropic API request payload.
#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    prompt: String,
    max_tokens_to_sample: u32,
    temperature: f32,
}

/// Anthropic API response.
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    completion: String,
}

/// Anthropic Claude API worker.
///
/// Requires `ANTHROPIC_API_KEY` environment variable.
///
/// ## Example
///
/// ```no_run
/// # use tokio_prompt_orchestrator::{AnthropicWorker, OrchestratorError};
/// # use std::sync::Arc;
/// # fn example() -> Result<(), OrchestratorError> {
/// let worker = Arc::new(
///     AnthropicWorker::new("claude-3-5-sonnet-20241022")?
///         .with_max_tokens(1024)
///         .with_temperature(1.0)
/// );
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct AnthropicWorker {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: f32,
    pub(crate) timeout: Duration,
    /// API base URL — override for Anthropic-compatible endpoints or testing.
    pub(crate) base_url: String,
}

impl AnthropicWorker {
    /// Create a new Anthropic worker.
    ///
    /// Reads the API key from the `ANTHROPIC_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns `Err(OrchestratorError::ConfigError)` if `ANTHROPIC_API_KEY` is not set.
    pub fn new(model: impl Into<String>) -> Result<Self, OrchestratorError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            OrchestratorError::ConfigError("ANTHROPIC_API_KEY environment variable not set".into())
        })?;

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.into(),
            max_tokens: 1024,
            temperature: 1.0,
            timeout: Duration::from_secs(60),
            base_url: "https://api.anthropic.com/v1".to_string(),
        })
    }

    /// Set maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature (0.0–1.0).
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the API base URL.
    ///
    /// Useful for Anthropic-compatible endpoints or for pointing at a mock server
    /// in tests. Default: `"https://api.anthropic.com/v1"`.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait]
impl ModelWorker for AnthropicWorker {
    async fn infer(&self, prompt: &str) -> Result<Vec<String>, OrchestratorError> {
        // Format prompt with Claude's expected format.
        let formatted_prompt = format!("\n\nHuman: {}\n\nAssistant:", prompt);

        let request = AnthropicRequest {
            model: self.model.clone(),
            prompt: formatted_prompt,
            max_tokens_to_sample: self.max_tokens,
            temperature: self.temperature,
        };

        let response = self
            .client
            .post(format!("{}/complete", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .timeout(self.timeout)
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                OrchestratorError::Inference(format!("Anthropic request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(OrchestratorError::Inference(format!(
                "Anthropic API error {}: {}",
                status, error_text
            )));
        }

        let api_response: AnthropicResponse = response.json().await.map_err(|e| {
            OrchestratorError::Inference(format!("Failed to parse response: {}", e))
        })?;

        // Split response into tokens.
        let tokens: Vec<String> = api_response
            .completion
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        Ok(tokens)
    }
}
