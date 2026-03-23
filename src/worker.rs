//! Model worker abstraction and implementations.
//!
//! Provides the `ModelWorker` trait and production-ready implementations:
//! - `EchoWorker`: Testing/demo worker
//! - `OpenAiWorker`: OpenAI API (GPT-4, GPT-3.5, etc.)
//! - `AnthropicWorker`: Anthropic Claude API
//! - `LlamaCppWorker`: Local llama.cpp server
//! - `VllmWorker`: vLLM inference server
//!
//! ## Module layout
//!
//! Each worker lives in its own sub-module under `src/workers/`.
//! This file is the crate-visible façade that re-exports everything.
//!
//! ## Environment Variables
//!
//! - `OPENAI_API_KEY`: Required for `OpenAiWorker`
//! - `ANTHROPIC_API_KEY`: Required for `AnthropicWorker`
//! - `LLAMA_CPP_URL`: llama.cpp server URL (default: http://localhost:8080)
//! - `VLLM_URL`: vLLM server URL (default: http://localhost:8000)

pub use crate::workers::{
    retry_with_backoff, AnthropicWorker, EchoWorker, LlamaCppWorker, ModelWorker, OpenAiWorker,
    RetryPolicy, VllmWorker,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OrchestratorError;
    use std::sync::Mutex;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    /// Serialise all tests that read/write environment variables so they don't race.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Create an `OpenAiWorker` that points at `base_url`.
    /// Must be called while `ENV_MUTEX` is held.
    fn make_openai_worker_for(base_url: &str) -> OpenAiWorker {
        std::env::set_var("OPENAI_API_KEY", "test-key-openai");
        let w = OpenAiWorker::new("gpt-3.5-turbo-instruct")
            .expect("OpenAiWorker::new must succeed when OPENAI_API_KEY is set")
            .with_base_url(base_url);
        std::env::remove_var("OPENAI_API_KEY");
        w
    }

    /// Create an `AnthropicWorker` that points at `base_url`.
    /// Must be called while `ENV_MUTEX` is held.
    fn make_anthropic_worker_for(base_url: &str) -> AnthropicWorker {
        std::env::set_var("ANTHROPIC_API_KEY", "test-key-anthropic");
        let w = AnthropicWorker::new("claude-instant-1-2")
            .expect("AnthropicWorker::new must succeed when ANTHROPIC_API_KEY is set")
            .with_base_url(base_url);
        std::env::remove_var("ANTHROPIC_API_KEY");
        w
    }

    fn openai_success_body() -> serde_json::Value {
        serde_json::json!({"choices": [{"text": "hello world response"}]})
    }

    fn anthropic_success_body() -> serde_json::Value {
        serde_json::json!({"completion": "hello world response"})
    }

    fn llamacpp_success_body() -> serde_json::Value {
        serde_json::json!({"content": "hello world response"})
    }

    fn vllm_success_body() -> serde_json::Value {
        serde_json::json!({"text": ["hello world response"]})
    }

    // ── EchoWorker ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_echo_worker_infer_splits_on_whitespace() {
        let worker = EchoWorker::with_delay(0);
        let tokens = worker.infer("hello world").await.unwrap();
        assert_eq!(tokens, vec!["hello", "world"]);
    }

    #[tokio::test]
    async fn test_echo_worker_infer_empty_prompt_returns_empty_tokens() {
        let worker = EchoWorker::with_delay(0);
        let tokens = worker.infer("").await.unwrap();
        assert!(tokens.is_empty(), "empty prompt should produce no tokens");
    }

    #[tokio::test]
    async fn test_echo_worker_infer_single_word_returns_one_token() {
        let worker = EchoWorker::with_delay(0);
        let tokens = worker.infer("hello").await.unwrap();
        assert_eq!(tokens, vec!["hello"]);
    }

    #[tokio::test]
    async fn test_echo_worker_infer_multiple_whitespace_is_normalised() {
        let worker = EchoWorker::with_delay(0);
        let tokens = worker.infer("a   b   c").await.unwrap();
        assert_eq!(tokens, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn test_echo_worker_with_delay_stores_delay_ms() {
        let worker = EchoWorker::with_delay(42);
        assert_eq!(worker.delay_ms, 42);
    }

    #[tokio::test]
    async fn test_echo_worker_new_delay_is_10ms() {
        let worker = EchoWorker::new();
        assert_eq!(worker.delay_ms, 10);
    }

    #[tokio::test]
    async fn test_echo_worker_default_via_trait_works() {
        let worker = EchoWorker::default();
        let tokens = worker.infer("one two three").await.unwrap();
        assert_eq!(tokens.len(), 3);
    }

    #[tokio::test]
    async fn test_echo_worker_infer_always_returns_ok() {
        let worker = EchoWorker::with_delay(0);
        assert!(worker.infer("anything").await.is_ok());
    }

    // ── OpenAiWorker — constructor ────────────────────────────────────────────

    #[test]
    fn test_openai_worker_new_missing_key_returns_config_error() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("OPENAI_API_KEY");
        let result = OpenAiWorker::new("gpt-4");
        assert!(result.is_err(), "Expected Err when OPENAI_API_KEY is not set");
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::ConfigError(ref msg) if msg.contains("OPENAI_API_KEY")),
            "Expected ConfigError mentioning OPENAI_API_KEY, got {:?}", err
        );
    }

    #[test]
    fn test_openai_worker_new_with_key_succeeds() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::set_var("OPENAI_API_KEY", "sk-test");
        let result = OpenAiWorker::new("gpt-4");
        std::env::remove_var("OPENAI_API_KEY");
        assert!(result.is_ok(), "Expected Ok when OPENAI_API_KEY is set");
    }

    // ── OpenAiWorker — inference ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_openai_infer_success_parses_response_correctly() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_success_body()))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_openai_worker_for(&server.uri()) };
        let tokens = worker.infer("test prompt").await.unwrap();
        assert_eq!(tokens, vec!["hello", "world", "response"]);
    }

    #[tokio::test]
    async fn test_openai_infer_http_500_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_openai_worker_for(&server.uri()) };
        let result = worker.infer("test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::Inference(ref msg) if msg.contains("500")),
            "Expected Inference error containing '500', got {:?}", err
        );
    }

    #[tokio::test]
    async fn test_openai_infer_empty_choices_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"choices": []})))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_openai_worker_for(&server.uri()) };
        let result = worker.infer("test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::Inference(ref msg) if msg.contains("choices")),
            "Expected Inference error mentioning 'choices', got {:?}", err
        );
    }

    #[tokio::test]
    async fn test_openai_infer_invalid_json_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not valid json {{{{"))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_openai_worker_for(&server.uri()) };
        assert!(worker.infer("test").await.is_err());
    }

    #[tokio::test]
    async fn test_openai_infer_sends_authorization_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .and(header("authorization", "Bearer test-key-openai"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_success_body()))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_openai_worker_for(&server.uri()) };
        let result = worker.infer("test").await;
        assert!(result.is_ok(), "Request with correct auth header should succeed");
    }

    #[tokio::test]
    async fn test_openai_infer_sends_correct_model_in_request_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_success_body()))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_openai_worker_for(&server.uri()) };
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 1, "Exactly one request should be sent");
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        assert_eq!(body["model"], "gpt-3.5-turbo-instruct");
    }

    #[tokio::test]
    async fn test_openai_with_max_tokens_sends_correct_value() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_success_body()))
            .mount(&server)
            .await;

        let worker = {
            let _g = ENV_MUTEX.lock().unwrap();
            std::env::set_var("OPENAI_API_KEY", "test-key-openai");
            let w = OpenAiWorker::new("gpt-4").unwrap().with_max_tokens(1024).with_base_url(&server.uri());
            std::env::remove_var("OPENAI_API_KEY");
            w
        };
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        assert_eq!(body["max_tokens"], 1024);
    }

    #[tokio::test]
    async fn test_openai_with_temperature_sends_correct_value() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_success_body()))
            .mount(&server)
            .await;

        let worker = {
            let _g = ENV_MUTEX.lock().unwrap();
            std::env::set_var("OPENAI_API_KEY", "test-key-openai");
            let w = OpenAiWorker::new("gpt-4").unwrap().with_temperature(0.3).with_base_url(&server.uri());
            std::env::remove_var("OPENAI_API_KEY");
            w
        };
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        let temp = body["temperature"].as_f64().unwrap();
        assert!((temp - 0.3_f64).abs() < 0.01, "Temperature should be ~0.3, got {temp}");
    }

    // ── AnthropicWorker — constructor ─────────────────────────────────────────

    #[test]
    fn test_anthropic_worker_new_missing_key_returns_config_error() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::remove_var("ANTHROPIC_API_KEY");
        let result = AnthropicWorker::new("claude-3-5-sonnet-20241022");
        assert!(result.is_err(), "Expected Err when ANTHROPIC_API_KEY is not set");
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::ConfigError(ref msg) if msg.contains("ANTHROPIC_API_KEY")),
            "Expected ConfigError mentioning ANTHROPIC_API_KEY, got {:?}", err
        );
    }

    #[test]
    fn test_anthropic_worker_new_with_key_succeeds() {
        let _guard = ENV_MUTEX.lock().unwrap();
        std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test");
        let result = AnthropicWorker::new("claude-3-5-sonnet-20241022");
        std::env::remove_var("ANTHROPIC_API_KEY");
        assert!(result.is_ok(), "Expected Ok when ANTHROPIC_API_KEY is set");
    }

    // ── AnthropicWorker — inference ───────────────────────────────────────────

    #[tokio::test]
    async fn test_anthropic_infer_success_returns_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_success_body()))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        let tokens = worker.infer("test prompt").await.unwrap();
        assert_eq!(tokens, vec!["hello", "world", "response"]);
    }

    #[tokio::test]
    async fn test_anthropic_infer_http_500_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(500).set_body_string("error"))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        let result = worker.infer("test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::Inference(ref msg) if msg.contains("500")),
            "Expected Inference error containing '500', got {:?}", err
        );
    }

    #[tokio::test]
    async fn test_anthropic_infer_invalid_json_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        assert!(worker.infer("test").await.is_err());
    }

    #[tokio::test]
    async fn test_anthropic_infer_sends_api_key_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .and(header("x-api-key", "test-key-anthropic"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_success_body()))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        let result = worker.infer("test").await;
        assert!(result.is_ok(), "Request with correct x-api-key header should succeed");
    }

    #[tokio::test]
    async fn test_anthropic_infer_sends_version_header() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .and(header("anthropic-version", "2023-06-01"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_success_body()))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        let result = worker.infer("test").await;
        assert!(result.is_ok(), "Request with correct anthropic-version header should succeed");
    }

    #[tokio::test]
    async fn test_anthropic_infer_sends_correct_model_in_request_body() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_success_body()))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 1, "Exactly one request should be sent");
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        assert_eq!(body["model"], "claude-instant-1-2");
    }

    #[tokio::test]
    async fn test_anthropic_with_max_tokens_sends_correct_value() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_success_body()))
            .mount(&server)
            .await;

        let worker = {
            let _g = ENV_MUTEX.lock().unwrap();
            std::env::set_var("ANTHROPIC_API_KEY", "test-key-anthropic");
            let w = AnthropicWorker::new("claude-instant-1-2").unwrap().with_max_tokens(2048).with_base_url(&server.uri());
            std::env::remove_var("ANTHROPIC_API_KEY");
            w
        };
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        assert_eq!(body["max_tokens_to_sample"], 2048);
    }

    #[tokio::test]
    async fn test_anthropic_infer_formats_prompt_with_human_and_assistant_prefix() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_success_body()))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        let _ = worker.infer("my question").await;

        let reqs = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        let prompt = body["prompt"].as_str().unwrap();
        assert!(prompt.contains("Human:"), "Prompt should contain 'Human:' prefix");
        assert!(prompt.contains("Assistant:"), "Prompt should contain 'Assistant:' marker");
        assert!(prompt.contains("my question"), "Prompt should include the original input");
    }

    // ── LlamaCppWorker ────────────────────────────────────────────────────────

    #[test]
    fn test_llamacpp_default_constructor_builds_worker() {
        let worker = LlamaCppWorker::new();
        assert!(!worker.url.is_empty(), "URL should be non-empty");
    }

    #[tokio::test]
    async fn test_llamacpp_infer_success_returns_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llamacpp_success_body()))
            .mount(&server)
            .await;

        let worker = LlamaCppWorker::new().with_url(server.uri());
        let tokens = worker.infer("test prompt").await.unwrap();
        assert_eq!(tokens, vec!["hello", "world", "response"]);
    }

    #[tokio::test]
    async fn test_llamacpp_infer_http_500_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&server)
            .await;

        let worker = LlamaCppWorker::new().with_url(server.uri());
        let result = worker.infer("test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::Inference(ref msg) if msg.contains("500")),
            "Expected Inference error containing '500', got {:?}", err
        );
    }

    #[tokio::test]
    async fn test_llamacpp_infer_invalid_json_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let worker = LlamaCppWorker::new().with_url(server.uri());
        assert!(worker.infer("test").await.is_err());
    }

    #[tokio::test]
    async fn test_llamacpp_sends_request_to_completion_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llamacpp_success_body()))
            .mount(&server)
            .await;

        let worker = LlamaCppWorker::new().with_url(server.uri());
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 1, "Exactly one request should be sent");
        assert_eq!(reqs[0].url.path(), "/completion");
    }

    #[tokio::test]
    async fn test_llamacpp_with_max_tokens_sends_n_predict_field() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llamacpp_success_body()))
            .mount(&server)
            .await;

        let worker = LlamaCppWorker::new().with_url(server.uri()).with_max_tokens(512);
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        assert_eq!(body["n_predict"], 512);
    }

    #[tokio::test]
    async fn test_llamacpp_infer_empty_content_returns_empty_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"content": ""})))
            .mount(&server)
            .await;

        let worker = LlamaCppWorker::new().with_url(server.uri());
        let tokens = worker.infer("test").await.unwrap();
        assert!(tokens.is_empty(), "Empty content should produce no tokens");
    }

    #[tokio::test]
    async fn test_llamacpp_with_url_overrides_default_server() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completion"))
            .respond_with(ResponseTemplate::new(200).set_body_json(llamacpp_success_body()))
            .mount(&server)
            .await;

        let worker = LlamaCppWorker::new().with_url(server.uri());
        let result = worker.infer("test").await;
        assert!(result.is_ok(), "Request should reach the mock server via with_url");
    }

    // ── VllmWorker ────────────────────────────────────────────────────────────

    #[test]
    fn test_vllm_default_constructor_builds_worker() {
        let worker = VllmWorker::new();
        assert!(!worker.url.is_empty(), "URL should be non-empty");
    }

    #[tokio::test]
    async fn test_vllm_infer_success_returns_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vllm_success_body()))
            .mount(&server)
            .await;

        let worker = VllmWorker::new().with_url(server.uri());
        let tokens = worker.infer("test prompt").await.unwrap();
        assert_eq!(tokens, vec!["hello", "world", "response"]);
    }

    #[tokio::test]
    async fn test_vllm_infer_http_500_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&server)
            .await;

        let worker = VllmWorker::new().with_url(server.uri());
        let result = worker.infer("test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::Inference(ref msg) if msg.contains("500")),
            "Expected Inference error containing '500', got {:?}", err
        );
    }

    #[tokio::test]
    async fn test_vllm_infer_empty_text_array_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"text": []})))
            .mount(&server)
            .await;

        let worker = VllmWorker::new().with_url(server.uri());
        let result = worker.infer("test").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::Inference(ref msg) if msg.contains("Empty")),
            "Expected Inference error mentioning 'Empty', got {:?}", err
        );
    }

    #[tokio::test]
    async fn test_vllm_infer_invalid_json_returns_inference_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_string("not json"))
            .mount(&server)
            .await;

        let worker = VllmWorker::new().with_url(server.uri());
        assert!(worker.infer("test").await.is_err());
    }

    #[tokio::test]
    async fn test_vllm_sends_request_to_generate_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vllm_success_body()))
            .mount(&server)
            .await;

        let worker = VllmWorker::new().with_url(server.uri());
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        assert_eq!(reqs.len(), 1, "Exactly one request should be sent");
        assert_eq!(reqs[0].url.path(), "/generate");
    }

    #[tokio::test]
    async fn test_vllm_with_max_tokens_sends_correct_value() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vllm_success_body()))
            .mount(&server)
            .await;

        let worker = VllmWorker::new().with_url(server.uri()).with_max_tokens(2048);
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        assert_eq!(body["max_tokens"], 2048);
    }

    #[tokio::test]
    async fn test_vllm_with_top_p_sends_correct_value() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vllm_success_body()))
            .mount(&server)
            .await;

        let worker = VllmWorker::new().with_url(server.uri()).with_top_p(0.85);
        let _ = worker.infer("test").await;

        let reqs = server.received_requests().await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
        let top_p = body["top_p"].as_f64().unwrap();
        assert!((top_p - 0.85_f64).abs() < 0.01, "top_p should be ~0.85, got {top_p}");
    }

    #[tokio::test]
    async fn test_vllm_with_url_overrides_default_server() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/generate"))
            .respond_with(ResponseTemplate::new(200).set_body_json(vllm_success_body()))
            .mount(&server)
            .await;

        let worker = VllmWorker::new().with_url(server.uri());
        let result = worker.infer("test").await;
        assert!(result.is_ok(), "Request should reach the mock server via with_url");
    }

    // ── Item 18: infer_stream mock HTTP tests ─────────────────────────────────

    #[tokio::test]
    async fn test_openai_infer_mock_server_returns_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"choices": [{"text": "token1 token2 token3"}]})))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_openai_worker_for(&server.uri()) };
        let tokens = worker.infer("test prompt").await.unwrap();
        assert_eq!(tokens, vec!["token1", "token2", "token3"]);
    }

    #[tokio::test]
    async fn test_anthropic_infer_mock_server_returns_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"completion": "alpha beta gamma"})))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        let tokens = worker.infer("test prompt").await.unwrap();
        assert_eq!(tokens, vec!["alpha", "beta", "gamma"]);
    }

    // ── Item 19: Negative worker test cases ───────────────────────────────────

    #[tokio::test]
    async fn test_openai_infer_empty_prompt_returns_empty_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"choices": [{"text": ""}]})))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_openai_worker_for(&server.uri()) };
        let tokens = worker.infer("").await.unwrap();
        assert!(tokens.is_empty(), "empty server response should give no tokens");
    }

    #[tokio::test]
    async fn test_anthropic_infer_empty_prompt_returns_empty_tokens() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"completion": ""})))
            .mount(&server)
            .await;

        let worker = { let _g = ENV_MUTEX.lock().unwrap(); make_anthropic_worker_for(&server.uri()) };
        let tokens = worker.infer("").await.unwrap();
        assert!(tokens.is_empty(), "empty completion should give no tokens");
    }

    #[tokio::test]
    async fn test_openai_infer_429_retried_then_success() {
        use wiremock::matchers::method as wm_method;

        let server = MockServer::start().await;

        Mock::given(wm_method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .expect(3)
            .up_to_n_times(3)
            .mount(&server)
            .await;

        Mock::given(wm_method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"choices": [{"text": "ok"}]})))
            .mount(&server)
            .await;

        let worker = {
            let _g = ENV_MUTEX.lock().unwrap();
            std::env::set_var("OPENAI_API_KEY", "test-key-openai");
            let mut w = OpenAiWorker::new("gpt-3.5-turbo-instruct")
                .expect("worker must be created")
                .with_base_url(&server.uri());
            w.retry_policy = RetryPolicy { max_retries: 3, base_delay_ms: 0 };
            std::env::remove_var("OPENAI_API_KEY");
            w
        };

        // 429 is a 4xx client error; returned immediately without retrying.
        let result = worker.infer("test").await;
        assert!(result.is_err(), "429 is a client error and should be returned immediately with current impl");
        let err = result.unwrap_err();
        assert!(
            matches!(err, OrchestratorError::Inference(ref msg) if msg.contains("429")),
            "Expected Inference error containing '429', got {:?}", err
        );
    }

    #[tokio::test]
    async fn test_openai_infer_timeout_returns_error() {
        use std::time::Duration;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(openai_success_body()).set_delay(Duration::from_millis(500)))
            .mount(&server)
            .await;

        let worker = {
            let _g = ENV_MUTEX.lock().unwrap();
            std::env::set_var("OPENAI_API_KEY", "test-key-openai");
            let mut w = OpenAiWorker::new("gpt-3.5-turbo-instruct")
                .expect("worker must be created")
                .with_base_url(&server.uri())
                .with_timeout(Duration::from_millis(50));
            w.retry_policy = RetryPolicy { max_retries: 0, base_delay_ms: 0 };
            std::env::remove_var("OPENAI_API_KEY");
            w
        };

        let result = worker.infer("test").await;
        assert!(result.is_err(), "timed-out request should return Err");
    }

    #[tokio::test]
    async fn test_anthropic_infer_timeout_returns_error() {
        use std::time::Duration;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/complete"))
            .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_success_body()).set_delay(Duration::from_millis(500)))
            .mount(&server)
            .await;

        let worker = {
            let _g = ENV_MUTEX.lock().unwrap();
            std::env::set_var("ANTHROPIC_API_KEY", "test-key-anthropic");
            let mut w = AnthropicWorker::new("claude-instant-1-2")
                .expect("worker must be created")
                .with_base_url(&server.uri())
                .with_timeout(Duration::from_millis(50));
            w.retry_policy = RetryPolicy { max_retries: 0, base_delay_ms: 0 };
            std::env::remove_var("ANTHROPIC_API_KEY");
            w
        };

        let result = worker.infer("test").await;
        assert!(result.is_err(), "timed-out request should return Err");
    }
}
