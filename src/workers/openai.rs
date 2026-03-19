//! OpenAI API worker (GPT-4, GPT-3.5-turbo-instruct, etc.)

use super::ModelWorker;
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
        let request = OpenAiRequest {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stop: None,
        };

        let response = self
            .client
            .post(format!("{}/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .timeout(self.timeout)
            .json(&request)
            .send()
            .await
            .map_err(|e| OrchestratorError::Inference(format!("OpenAI request failed: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
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
}
