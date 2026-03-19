//! OpenAI API worker (GPT-4, GPT-3.5-turbo-instruct, etc.)

use super::{retry_with_backoff, ModelWorker, RetryPolicy};
use crate::OrchestratorError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// OpenAI API request payload.
#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

/// OpenAI API response.
#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    text: String,
}

/// OpenAI API worker (GPT-4, GPT-3.5-turbo-instruct, etc.)
///
/// Requires `OPENAI_API_KEY` environment variable.
///
/// ## Example
///
/// ```no_run
/// # use tokio_prompt_orchestrator::{OpenAiWorker, OrchestratorError};
/// # use std::sync::Arc;
/// # fn example() -> Result<(), OrchestratorError> {
/// let worker = Arc::new(
///     OpenAiWorker::new("gpt-3.5-turbo-instruct")?
///         .with_max_tokens(512)
///         .with_temperature(0.7)
/// );
/// # Ok(()) }
/// ```
#[derive(Debug)]
pub struct OpenAiWorker {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) max_tokens: u32,
    pub(crate) temperature: f32,
    pub(crate) timeout: Duration,
    /// API base URL — override for OpenAI-compatible endpoints or testing.
    pub(crate) base_url: String,
    /// Retry policy for transient errors.
    pub(crate) retry_policy: RetryPolicy,
}

impl OpenAiWorker {
    /// Create a new OpenAI worker.
    ///
    /// Reads the API key from the `OPENAI_API_KEY` environment variable.
    ///
    /// # Errors
    ///
    /// Returns `Err(OrchestratorError::ConfigError)` if `OPENAI_API_KEY` is not set.
    pub fn new(model: impl Into<String>) -> Result<Self, OrchestratorError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            OrchestratorError::ConfigError("OPENAI_API_KEY environment variable not set".into())
        })?;

        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
            model: model.into(),
            max_tokens: 256,
            temperature: 0.7,
            timeout: Duration::from_secs(30),
            base_url: "https://api.openai.com/v1".to_string(),
            retry_policy: RetryPolicy::default(),
        })
    }

    /// Set maximum tokens to generate.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature (0.0–2.0).
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
    /// Useful for OpenAI-compatible endpoints (Azure OpenAI, Groq, local proxies)
    /// and for pointing at a mock server in tests.
    /// Default: `"https://api.openai.com/v1"`.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[async_trait]
impl ModelWorker for OpenAiWorker {
    async fn infer(&self, prompt: &str) -> Result<Vec<String>, OrchestratorError> {
        let prompt = prompt.to_string();

        // Use retry_with_backoff for transient (non-4xx) HTTP errors.
        retry_with_backoff(&self.retry_policy, || {
            let prompt = prompt.clone();
            let client = self.client.clone();
            let base_url = self.base_url.clone();
            let api_key = self.api_key.clone();
            let model = self.model.clone();
            let max_tokens = self.max_tokens;
            let temperature = self.temperature;
            let timeout = self.timeout;

            async move {
                let request = OpenAiRequest {
                    model,
                    prompt,
                    max_tokens,
                    temperature,
                    stop: None,
                };

                let response = client
                    .post(format!("{}/completions", base_url))
                    .header("Authorization", format!("Bearer {}", api_key))
                    .header("Content-Type", "application/json")
                    .timeout(timeout)
                    .json(&request)
                    .send()
                    .await
                    .map_err(|e| {
                        if e.is_timeout() {
                            tracing::warn!(error_code = "timeout", "worker timeout");
                        } else {
                            tracing::warn!(error_code = "network", "worker network error");
                        }
                        OrchestratorError::Inference(format!("OpenAI request failed: {}", e))
                    })?;

                let status = response.status();

                // 4xx errors are permanent (client errors); return immediately
                // without retrying.
                if status.is_client_error() {
                    let error_text = response.text().await.unwrap_or_default();
                    if status.as_u16() == 429 {
                        tracing::warn!(error_code = "rate_limit", "worker rate limited");
                    }
                    return Err(OrchestratorError::Inference(format!(
                        "OpenAI API error {}: {}",
                        status, error_text
                    )));
                }

                if !status.is_success() {
                    let error_text = response.text().await.unwrap_or_default();
                    if status.is_server_error() {
                        tracing::warn!(error_code = "server_error", "worker server error");
                    }
                    return Err(OrchestratorError::Inference(format!(
                        "OpenAI API error {}: {}",
                        status, error_text
                    )));
                }

                let api_response: OpenAiResponse = response.json().await.map_err(|e| {
                    OrchestratorError::Inference(format!("Failed to parse response: {}", e))
                })?;

                if api_response.choices.is_empty() {
                    return Err(OrchestratorError::Inference(
                        "No choices in OpenAI response".to_string(),
                    ));
                }

                // Split response into tokens (simple whitespace split).
                let tokens: Vec<String> = api_response.choices[0]
                    .text
                    .split_whitespace()
                    .map(|s| s.to_string())
                    .collect();

                Ok(tokens)
            }
        })
        .await
    }
}
