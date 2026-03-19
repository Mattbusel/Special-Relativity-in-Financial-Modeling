//! Worker trait and per-worker implementations.
//!
//! Re-exports all worker types so callers can use `crate::workers::EchoWorker`
//! or (via `src/worker.rs`) the crate-level `crate::EchoWorker`.

pub mod anthropic;
pub mod echo;
pub mod llama;
pub mod openai;
pub mod vllm;

pub use anthropic::AnthropicWorker;
pub use echo::EchoWorker;
pub use llama::LlamaCppWorker;
pub use openai::OpenAiWorker;
pub use vllm::VllmWorker;

use crate::OrchestratorError;
use async_trait::async_trait;

#[cfg(feature = "web-api")]
use futures::stream::Stream;
#[cfg(feature = "web-api")]
use std::pin::Pin;

/// Retry policy for transient worker errors.
///
/// Used by [`retry_with_backoff`] to control retry behaviour.
pub struct RetryPolicy {
    /// Maximum number of additional attempts after the first failure.
    pub max_retries: u32,
    /// Base delay in milliseconds between retries (doubles each attempt).
    pub base_delay_ms: u64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
        }
    }
}

/// Retry `f` with exponential backoff + jitter on error.
///
/// The closure is called up to `policy.max_retries + 1` times total.
/// Between attempts the helper sleeps for
/// `base_delay_ms * 2^attempt + jitter(0..base_delay_ms)` milliseconds.
///
/// Transient errors (network failures, HTTP 5xx) should be retried;
/// client errors (HTTP 4xx) are considered permanent and are returned
/// immediately without retrying.  The caller is responsible for
/// classifying errors before passing them to this helper — pass only
/// `is_transient == true` errors into the retry loop.
pub async fn retry_with_backoff<F, Fut, T>(
    policy: &RetryPolicy,
    mut f: F,
) -> Result<T, OrchestratorError>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, OrchestratorError>>,
{
    let mut last_err;
    let mut attempt = 0u32;
    loop {
        match f().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                last_err = e;
                if attempt >= policy.max_retries {
                    break;
                }
                // Exponential backoff: base * 2^attempt, plus a small jitter.
                let backoff_ms = policy.base_delay_ms * (1u64 << attempt.min(10));
                // Simple deterministic jitter: add half the base delay.
                let jitter_ms = policy.base_delay_ms / 2;
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms + jitter_ms)).await;
                attempt += 1;
            }
        }
    }
    Err(last_err)
}

/// Trait for model inference workers.
///
/// Implementations must be thread-safe (`Send + Sync`) for use across tasks.
/// The trait is object-safe to allow dynamic dispatch via `Arc<dyn ModelWorker>`.
#[async_trait]
pub trait ModelWorker: Send + Sync {
    /// Perform inference on the given prompt.
    ///
    /// Returns tokens as a vector of strings.
    /// For streaming implementations, this should be the final token set.
    async fn infer(&self, prompt: &str) -> Result<Vec<String>, OrchestratorError>;

    /// Streaming variant of [`infer`].
    ///
    /// Yields `Result<String, OrchestratorError>` items as they become available.
    /// The default implementation buffers the full response from [`infer`] and
    /// emits it as a single stream item, so existing workers remain compatible
    /// without any changes.
    ///
    /// Override this method to provide true token-by-token streaming.
    ///
    /// Requires the `web-api` feature (which activates the `futures` crate).
    #[cfg(feature = "web-api")]
    async fn infer_stream(
        &self,
        prompt: &str,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<String, OrchestratorError>> + Send>>,
        OrchestratorError,
    > {
        let result = self.infer(prompt).await?;
        let joined = result.join(" ");
        Ok(Box::pin(futures::stream::once(async move { Ok(joined) })))
    }
}
