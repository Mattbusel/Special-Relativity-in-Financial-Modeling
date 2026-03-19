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
    #[cfg(feature = "web-api")]
    async fn infer_stream(
        &self,
        prompt: &str,
    ) -> Result<
        std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<String, OrchestratorError>> + Send>>,
        OrchestratorError,
    > {
        use futures::StreamExt;

        let body = serde_json::json!({
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": true
        });

        let response = self
            .client
            .post(format!("{}/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| OrchestratorError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(OrchestratorError::RequestFailed(format!(
                "OpenAI streaming error: {}",
                response.status()
            )));
        }

        // Parse SSE stream: each line is "data: {json}" or "data: [DONE]"
        let byte_stream = response.bytes_stream();
        let token_stream = byte_stream
            .filter_map(|chunk| async move {
                let bytes = chunk.ok()?;
                let text = String::from_utf8_lossy(&bytes).to_string();
                // Each chunk may contain multiple SSE lines
                let tokens: Vec<String> = text
                    .lines()
                    .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
                    .filter_map(|l| {
                        let json_str = &l["data: ".len()..];
                        serde_json::from_str::<serde_json::Value>(json_str).ok()
                    })
                    .filter_map(|v| {
                        v["choices"][0]["text"]
                            .as_str()
                            .or_else(|| v["choices"][0]["delta"]["content"].as_str())
                            .map(String::from)
                    })
                    .filter(|s| !s.is_empty())
                    .collect();
                if tokens.is_empty() { None } else { Some(tokens) }
            })
            .flat_map(|tokens| {
                futures::stream::iter(tokens.into_iter().map(Ok))
            });

        Ok(Box::pin(token_stream))
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    fn make_worker_for(base_url: &str) -> OpenAiWorker {
        std::env::set_var("OPENAI_API_KEY", "test-key-openai");
        let w = OpenAiWorker::new("gpt-3.5-turbo-instruct")
            .unwrap()
            .with_base_url(base_url);
        std::env::remove_var("OPENAI_API_KEY");
        w
    }

    #[cfg(feature = "web-api")]
    #[tokio::test]
    async fn test_openai_worker_infer_stream_parses_sse() {
        use futures::StreamExt;
        // ModelWorker is in scope via `use super::*` above.

        let server = MockServer::start().await;

        // Build a mock SSE response with two token chunks followed by [DONE].
        let sse_body = concat!(
            "data: {\"choices\":[{\"text\":\"Hello\"}]}\n\n",
            "data: {\"choices\":[{\"text\":\" world\"}]}\n\n",
            "data: [DONE]\n\n",
        );

        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse_body),
            )
            .mount(&server)
            .await;

        let worker = {
            let _g = ENV_MUTEX.lock().unwrap();
            make_worker_for(&server.uri())
        };

        let stream = worker.infer_stream("test prompt").await.unwrap();
        let tokens: Vec<String> = stream
            .filter_map(|r| async move { r.ok() })
            .collect()
            .await;

        assert!(!tokens.is_empty(), "stream should yield at least one token");
        let joined = tokens.join("");
        assert!(
            joined.contains("Hello"),
            "stream output should contain 'Hello', got {:?}",
            joined
        );
    }
}
