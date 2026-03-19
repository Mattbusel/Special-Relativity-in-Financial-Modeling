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
        let tokens = self.infer(prompt).await?;
        Ok(Box::pin(futures::stream::iter(tokens.into_iter().map(Ok))))
    }
}
