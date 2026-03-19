//! EchoWorker — testing/demo worker that echoes the prompt back as tokens.

use super::ModelWorker;
use crate::OrchestratorError;
use async_trait::async_trait;

#[cfg(feature = "web-api")]
use futures::stream::Stream;
#[cfg(feature = "web-api")]
use std::pin::Pin;

/// Dummy echo worker for testing.
///
/// Simply splits the prompt into words and returns them as tokens.
/// Useful for pipeline smoke tests without real model dependencies.
pub struct EchoWorker {
    /// Simulated inference delay in milliseconds.
    pub delay_ms: u64,
}

impl EchoWorker {
    /// Create a new `EchoWorker` with a default 10 ms simulated delay.
    pub fn new() -> Self {
        Self { delay_ms: 10 }
    }

    /// Create a new `EchoWorker` with a custom simulated inference delay in milliseconds.
    pub fn with_delay(delay_ms: u64) -> Self {
        Self { delay_ms }
    }
}

impl Default for EchoWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ModelWorker for EchoWorker {
    async fn infer(&self, prompt: &str) -> Result<Vec<String>, OrchestratorError> {
        // Simulate inference latency.
        tokio::time::sleep(tokio::time::Duration::from_millis(self.delay_ms)).await;

        // Echo back the prompt as tokens.
        let tokens: Vec<String> = prompt.split_whitespace().map(|s| s.to_string()).collect();

        Ok(tokens)
    }

    /// Real word-by-word streaming implementation for `EchoWorker`.
    ///
    /// Yields one word at a time with a small delay between each word, simulating
    /// a model that produces tokens incrementally.
    #[cfg(feature = "web-api")]
    async fn infer_stream(
        &self,
        prompt: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<String, OrchestratorError>> + Send>>,
        OrchestratorError,
    > {
        use futures::StreamExt;

        let words: Vec<String> = prompt.split_whitespace().map(|s| s.to_string()).collect();
        let delay_ms = self.delay_ms;

        // Build a stream that yields one word per iteration, each after a small delay.
        let stream = futures::stream::iter(words).then(move |word| async move {
            if delay_ms > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(delay_ms)).await;
            }
            Ok::<String, OrchestratorError>(word)
        });

        Ok(Box::pin(stream))
    }
}
