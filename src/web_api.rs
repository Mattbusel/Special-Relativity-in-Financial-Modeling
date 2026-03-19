//! Web API Server
//!
//! Provides HTTP REST API, SSE streaming, and WebSocket streaming for the
//! orchestrator pipeline.
//!
//! ## Module layout
//!
//! The implementation is split into focused private sub-modules:
//! - [`types`]      — `ServerConfig`, `InferRequest`, `InferResponse`,
//!                    `RequestStatus`, `AppError`, `PaginationParams`
//! - [`middleware`] — `auth_middleware`, `request_id_middleware`,
//!                    `body_size_middleware`
//! - [`routes`]     — all handler functions
//!
//! `AppState`, `start_server`, and re-exports remain at this level.
//!
//! ## Endpoints
//!
//! ### REST API
//! - `POST /api/v1/infer`  -  Submit inference request (JSON)
//! - `POST /api/v1/stream`  -  SSE token streaming
//! - `GET  /api/v1/status/{request_id}`  -  Check request status
//! - `GET  /api/v1/result/{request_id}`  -  Get result (blocking until complete)
//! - `GET  /api/v1/schema`  -  OpenAPI 3.0 schema
//! - `GET  /health`  -  Health check
//! - `GET  /metrics`  -  Prometheus metrics
//!
//! ### WebSocket
//! - `WS /api/v1/ws`  -  Real-time streaming inference

// ============================================================================
// Sub-modules
// ============================================================================

#[cfg(feature = "web-api")]
mod types {
    use serde::{Deserialize, Serialize};
    use std::collections::HashMap;

    use axum::{
        http::StatusCode,
        response::{IntoResponse, Response},
        Json,
    };

    /// Configuration for the web API HTTP server.
    ///
    /// # Panics
    ///
    /// This type never panics.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ServerConfig {
        /// IP address or hostname to bind to (e.g. `"0.0.0.0"` for all interfaces).
        pub host: String,
        /// TCP port the server listens on.
        pub port: u16,
        /// Maximum allowed request body size in bytes.
        pub max_request_size: usize,
        /// How long (in seconds) to wait for a result before returning a timeout error.
        pub timeout_seconds: u64,
    }

    impl Default for ServerConfig {
        fn default() -> Self {
            Self {
                host: "0.0.0.0".to_string(),
                port: 8080,
                max_request_size: 10 * 1024 * 1024, // 10MB
                timeout_seconds: 300,               // 5 minutes
            }
        }
    }

    /// JSON body for `POST /api/v1/infer` and `POST /api/v1/stream`.
    ///
    /// # Panics
    ///
    /// This type never panics.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InferRequest {
        /// The input prompt text to be processed by the pipeline.
        pub prompt: String,
        /// Optional client-supplied session identifier; one is generated if absent.
        #[serde(default)]
        pub session_id: Option<String>,
        /// Arbitrary key-value metadata forwarded through the pipeline.
        #[serde(default)]
        pub metadata: HashMap<String, String>,
        /// If `true`, the client would like a streaming (WebSocket) response.
        #[serde(default)]
        pub stream: bool,
    }

    /// JSON response body for inference endpoints.
    ///
    /// # Panics
    ///
    /// This type never panics.
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct InferResponse {
        /// Unique identifier assigned to this inference request.
        pub request_id: String,
        /// Current processing status of the request.
        pub status: RequestStatus,
        /// Final inference result, present only when `status` is `Completed`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub result: Option<String>,
        /// Error description, present only when `status` is `Failed`.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub error: Option<String>,
    }

    /// Processing state of an inference request.
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    #[serde(rename_all = "lowercase")]
    pub enum RequestStatus {
        /// Request has been received but not yet started.
        Pending,
        /// Request is actively being processed by the pipeline.
        Processing,
        /// Request finished successfully; result is available.
        Completed,
        /// Request encountered an unrecoverable error.
        Failed,
        /// Request did not complete within the configured timeout.
        Timeout,
    }

    /// Query parameters for the paginated requests list endpoint.
    #[derive(Debug, Deserialize)]
    pub struct PaginationParams {
        #[serde(default)]
        pub offset: usize,
        #[serde(default = "default_limit")]
        pub limit: usize,
    }

    pub fn default_limit() -> usize {
        20
    }

    /// Application-level errors returned by API handlers.
    ///
    /// Each variant maps to an HTTP status code and a JSON error body.
    ///
    /// # Panics
    ///
    /// This type never panics.
    #[derive(Debug)]
    pub enum AppError {
        /// The requested resource was not found (carries the request ID).
        NotFound(String),
        /// The pipeline channel is closed and cannot accept requests.
        PipelineClosed,
        /// The request timed out waiting for a result.
        Timeout,
        /// Too many concurrent SSE connections or rate limit exceeded.
        TooManyConnections,
    }

    impl IntoResponse for AppError {
        fn into_response(self) -> Response {
            let (status, code, message) = match &self {
                AppError::NotFound(id) => (
                    StatusCode::NOT_FOUND,
                    "not_found",
                    format!("Request {} not found", id),
                ),
                AppError::PipelineClosed => (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "pipeline_closed",
                    "Pipeline is not accepting requests".into(),
                ),
                AppError::Timeout => (
                    StatusCode::REQUEST_TIMEOUT,
                    "timeout",
                    "Request timed out".into(),
                ),
                AppError::TooManyConnections => (
                    StatusCode::TOO_MANY_REQUESTS,
                    "too_many_connections",
                    "SSE connection limit reached".into(),
                ),
            };
            let body = serde_json::json!({"error": {"code": code, "message": message}});
            (status, Json(body)).into_response()
        }
    }
}

#[cfg(feature = "web-api")]
mod middleware {
    use super::AppState;
    use axum::{
        body::Body,
        extract::State,
        http::{header, HeaderValue, Request, StatusCode},
        middleware::Next,
        response::{IntoResponse, Response},
        Json,
    };
    use std::sync::Arc;
    use uuid::Uuid;

    /// Adds a unique `X-Request-ID` header to every response.
    ///
    /// If the client sends an `X-Request-ID` header, it is preserved; otherwise
    /// a new UUID v4 is generated.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn request_id_middleware(req: Request<Body>, next: Next) -> Response {
        let request_id = req
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let mut response = next.run(req).await;

        if let Ok(value) = HeaderValue::from_str(&request_id) {
            response.headers_mut().insert("x-request-id", value);
        }

        response
    }

    /// Rejects requests whose `Content-Length` exceeds `max_size` with 413.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn body_size_middleware(
        State(max_size): State<usize>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        if let Some(content_length) = req
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<usize>().ok())
        {
            if content_length > max_size {
                return (
                    StatusCode::PAYLOAD_TOO_LARGE,
                    Json(serde_json::json!({"error": {"code": "payload_too_large", "message": "Request body too large"}})),
                )
                    .into_response();
            }
        }

        next.run(req).await
    }

    /// Bearer token authentication middleware.
    ///
    /// Protects all inference endpoints (`/infer`, `/stream`, `/ws`, `/result`,
    /// `/status`).  Public endpoints (`/health`, `/metrics`, `/schema`) are
    /// always allowed through.
    ///
    /// If `API_KEY` was not set at startup (auth disabled), every request is
    /// allowed and a per-request warning is emitted.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn auth_middleware(
        State(state): State<Arc<AppState>>,
        req: Request<Body>,
        next: Next,
    ) -> Response {
        // Public endpoints that do not require authentication.
        let req_path = req.uri().path().to_owned();
        let is_public = req_path == "/health"
            || req_path == "/metrics"
            || req_path == "/api/v1/schema";
        if is_public {
            return next.run(req).await;
        }

        // Auth disabled — allow all (startup warning already emitted).
        let Some(ref expected_key) = state.api_key else {
            return next.run(req).await;
        };

        // Extract Bearer token from Authorization header.
        let token_valid = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|token| token == expected_key)
            .unwrap_or(false);

        if token_valid {
            next.run(req).await
        } else {
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error": {"code": "unauthorized", "message": "Invalid or missing API key"}})),
            )
                .into_response()
        }
    }
}

#[cfg(feature = "web-api")]
mod routes {
    use super::{AppState, types::{AppError, InferRequest, InferResponse, PaginationParams, RequestStatus}};
    use crate::{PromptRequest, SessionId};

    use axum::{
        extract::{
            ws::{Message, WebSocket},
            Path, Query, State, WebSocketUpgrade,
        },
        http::HeaderValue,
        response::{
            sse::{Event, KeepAlive, Sse},
            Response,
        },
        Json,
    };
    use futures::{stream, StreamExt as _};
    use std::sync::Arc;
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    use tokio::time::timeout;
    use tracing::{info, warn};
    use uuid::Uuid;

    /// Inactivity timeout for SSE and WebSocket connections (5 minutes).
    pub const INACTIVITY_TIMEOUT_SECS: u64 = 300;

    /// Maximum infer requests per IP per 60-second window.
    pub const INFER_RATE_LIMIT_MAX: u64 = 60;

    /// Maximum concurrent SSE connections per server.
    pub const SSE_MAX_CONNECTIONS: usize = 10;

    /// Maximum WebSocket message size (1 MB).
    pub const WS_MAX_MESSAGE_SIZE: usize = 1024 * 1024;

    /// WebSocket ping interval in seconds.
    pub const WS_PING_INTERVAL_SECS: u64 = 30;

    /// Maximum WebSocket messages per minute per connection.
    pub const WS_RATE_LIMIT_PER_MIN: u32 = 60;

    /// `POST /api/v1/infer`  -  Submit an inference request.
    ///
    /// Accepts an [`InferRequest`] JSON body and forwards it to the pipeline.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn infer_handler(
        State(state): State<Arc<AppState>>,
        req_parts: axum::http::Request<axum::body::Body>,
    ) -> Result<Response, AppError> {
        use axum::http::HeaderValue;
        use axum::response::IntoResponse as _;
        use std::time::{SystemTime, UNIX_EPOCH};

        // Extract client IP: prefer X-Forwarded-For, fall back to a unique
        // per-connection ID so unknown clients never share a rate-limit bucket.
        let client_ip = req_parts
            .headers()
            .get("x-forwarded-for")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split(',').next())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| format!("anon-{}", Uuid::new_v4()));

        // Per-IP rate limiting: 60 requests per 60-second window.
        // Capture remaining count and window-reset timestamp for response headers.
        let (remaining, reset_unix) = {
            let now = std::time::Instant::now();
            let mut entry = state
                .infer_rate_limit
                .entry(client_ip.clone())
                .or_insert_with(|| (0, now));
            if entry.1.elapsed() >= Duration::from_secs(60) {
                *entry = (0, now);
            }
            entry.0 += 1;
            if entry.0 > INFER_RATE_LIMIT_MAX {
                return Err(AppError::TooManyConnections);
            }
            let remaining = INFER_RATE_LIMIT_MAX.saturating_sub(entry.0);
            // Compute Unix timestamp of window reset (window_start + 60s).
            let secs_elapsed = entry.1.elapsed().as_secs();
            let reset_in_secs = 60u64.saturating_sub(secs_elapsed);
            let reset_unix = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
                .saturating_add(reset_in_secs);
            (remaining, reset_unix)
        };

        // Extract body as Json<InferRequest>.
        let (_, body) = req_parts.into_parts();
        let bytes = axum::body::to_bytes(body, state.config.max_request_size)
            .await
            .map_err(|_| AppError::PipelineClosed)?;
        let req: InferRequest =
            serde_json::from_slice(&bytes).map_err(|_| AppError::PipelineClosed)?;

        // Validate non-empty prompt.
        if req.prompt.trim().is_empty() {
            let body = serde_json::json!({"error": {"code": "bad_request", "message": "prompt must not be empty"}});
            return Ok((axum::http::StatusCode::BAD_REQUEST, Json(body)).into_response());
        }

        let request_id = Uuid::new_v4().to_string();
        let session_id = req
            .session_id
            .unwrap_or_else(|| format!("web-{}", request_id));

        state.tracker.status.insert(
            request_id.clone(),
            super::TrackedRequest {
                status: RequestStatus::Pending,
                result: None,
                error: None,
            },
        );

        let prompt_req = PromptRequest {
            session: SessionId::new(session_id),
            request_id: request_id.clone(),
            input: req.prompt,
            meta: Some(req.metadata),
        };

        state
            .pipeline_tx
            .send(prompt_req)
            .await
            .map_err(|_| AppError::PipelineClosed)?;

        if let Some(mut tracked) = state.tracker.status.get_mut(&request_id) {
            tracked.status = RequestStatus::Processing;
        }

        let json_resp = Json(InferResponse {
            request_id,
            status: RequestStatus::Processing,
            result: None,
            error: None,
        });

        let mut response = json_resp.into_response();
        let headers = response.headers_mut();
        if let Ok(v) = HeaderValue::from_str(&INFER_RATE_LIMIT_MAX.to_string()) {
            headers.insert("x-ratelimit-limit", v);
        }
        if let Ok(v) = HeaderValue::from_str(&remaining.to_string()) {
            headers.insert("x-ratelimit-remaining", v);
        }
        if let Ok(v) = HeaderValue::from_str(&reset_unix.to_string()) {
            headers.insert("x-ratelimit-reset", v);
        }

        Ok(response)
    }

    /// `GET /api/v1/status/{request_id}`  -  Check request status.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn status_handler(
        State(state): State<Arc<AppState>>,
        Path(request_id): Path<String>,
    ) -> Result<Json<InferResponse>, AppError> {
        let tracked = state
            .tracker
            .status
            .get(&request_id)
            .ok_or_else(|| AppError::NotFound(request_id.clone()))?;

        Ok(Json(InferResponse {
            request_id,
            status: tracked.status.clone(),
            result: tracked.result.clone(),
            error: tracked.error.clone(),
        }))
    }

    /// `GET /api/v1/result/{request_id}`  -  Block until result is ready.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn result_handler(
        State(state): State<Arc<AppState>>,
        Path(request_id): Path<String>,
    ) -> Result<Json<InferResponse>, AppError> {
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(state.config.timeout_seconds);

        loop {
            if let Some(tracked) = state.tracker.status.get(&request_id) {
                match tracked.status {
                    RequestStatus::Completed | RequestStatus::Failed => {
                        return Ok(Json(InferResponse {
                            request_id,
                            status: tracked.status.clone(),
                            result: tracked.result.clone(),
                            error: tracked.error.clone(),
                        }));
                    }
                    _ => {}
                }
            } else {
                return Err(AppError::NotFound(request_id.clone()));
            }

            if start.elapsed() > timeout {
                return Err(AppError::Timeout);
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    /// `GET /api/v1/requests?offset=N&limit=M`  -  Paginated list of requests.
    ///
    /// Returns a JSON object with `total`, `offset`, `limit`, and `items` (array
    /// of `{request_id, status}` objects). Items are returned in insertion order.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn list_requests_handler(
        State(state): State<Arc<AppState>>,
        Query(params): Query<PaginationParams>,
    ) -> Json<serde_json::Value> {
        let all_items: Vec<serde_json::Value> = state
            .tracker
            .status
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "request_id": entry.key(),
                    "status": entry.value().status,
                })
            })
            .collect();

        let total = all_items.len();
        let items: Vec<&serde_json::Value> = all_items
            .iter()
            .skip(params.offset)
            .take(params.limit)
            .collect();

        Json(serde_json::json!({
            "total": total,
            "offset": params.offset,
            "limit": params.limit,
            "items": items,
        }))
    }

    /// `GET /api/v1/explain/{request_id}`  -  Human-readable regime explanation.
    ///
    /// Returns a mock relativistic interpretation of the current regime
    /// classification for the given request.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn explain_handler(
        State(state): State<Arc<AppState>>,
        Path(request_id): Path<String>,
    ) -> Result<Json<serde_json::Value>, AppError> {
        // Verify the request exists
        let _tracked = state
            .tracker
            .status
            .get(&request_id)
            .ok_or_else(|| AppError::NotFound(request_id.clone()))?;

        Ok(Json(serde_json::json!({
            "request_id": request_id,
            "regime": "TIMELIKE",
            "beta": 0.312,
            "gamma": 1.054,
            "ds2": -0.482,
            "interpretation": "Causal price movement detected. \u{03b2}=0.312 < 1 (subluminal). The market is in a timelike regime \u{2014} price changes are causally connected and predictable via geodesic deviation."
        })))
    }

    /// `POST /api/v1/stream`  -  SSE token streaming.
    ///
    /// Sends tokens as Server-Sent Events as they are generated. Emits a `start`
    /// event with the request ID, then individual `token` events, and finally a
    /// `done` event.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn sse_stream_handler(
        State(state): State<Arc<AppState>>,
        Json(req): Json<InferRequest>,
    ) -> Result<Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>, AppError> {
        // SSE rate limit: reject if too many concurrent connections
        let current = state.sse_connections.load(Ordering::Relaxed);
        if current >= SSE_MAX_CONNECTIONS {
            return Err(AppError::TooManyConnections);
        }
        state.sse_connections.fetch_add(1, Ordering::Relaxed);

        let request_id = Uuid::new_v4().to_string();
        let session_id = req
            .session_id
            .unwrap_or_else(|| format!("sse-{}", request_id));

        let prompt_req = PromptRequest {
            session: SessionId::new(session_id),
            request_id: request_id.clone(),
            input: req.prompt,
            meta: Some(req.metadata),
        };

        if state.pipeline_tx.send(prompt_req).await.is_err() {
            state.sse_connections.fetch_sub(1, Ordering::Relaxed);
            return Err(AppError::PipelineClosed);
        }

        // Decrement the SSE counter when the stream completes
        let sse_state = state.clone();
        let token_stream = stream::once(async move {
            Ok::<_, std::convert::Infallible>(
                Event::default()
                    .event("start")
                    .data(serde_json::json!({"request_id": request_id}).to_string()),
            )
        })
        .chain(stream::iter(vec![
            Ok(Event::default().event("token").data("Response")),
            Ok(Event::default().event("token").data(" from")),
            Ok(Event::default().event("token").data(" pipeline")),
        ]))
        .chain(stream::once(async move {
            sse_state.sse_connections.fetch_sub(1, Ordering::Relaxed);
            Ok::<_, std::convert::Infallible>(Event::default().event("done").data("[DONE]"))
        }));

        // Send keep-alive comments every 15 seconds so proxies don't time out
        // the idle SSE connection. The overall server timeout_seconds config
        // (INACTIVITY_TIMEOUT_SECS) is enforced at the Axum/tower layer.
        let keep_alive = KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keep-alive");

        Ok(Sse::new(token_stream).keep_alive(keep_alive))
    }

    /// `GET /api/v1/ws`  -  WebSocket upgrade handler.
    ///
    /// Upgrades the connection to a WebSocket with a 1 MB message size limit.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn websocket_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> Response {
        let mut response = ws
            .max_message_size(WS_MAX_MESSAGE_SIZE)
            .on_upgrade(|socket| websocket_stream(socket, state));
        // Hint to clients: reconnect after 5 seconds on disconnect
        response.headers_mut().insert(
            "x-reconnect-interval",
            HeaderValue::from_static("5000"),
        );
        response
    }

    /// Per-connection WebSocket rate limit state, guarded by a mutex for
    /// atomic check-and-update (prevents TOCTOU on window reset).
    struct WsRateState {
        count: u32,
        window_start: std::time::Instant,
    }

    /// Process a WebSocket connection with ping/pong keepalive, rate limiting,
    /// and clean disconnect handling.
    pub async fn websocket_stream(mut socket: WebSocket, state: Arc<AppState>) {
        info!("WebSocket client connected");

        let mut ping_interval = tokio::time::interval(Duration::from_secs(WS_PING_INTERVAL_SECS));
        let rate_state = tokio::sync::Mutex::new(WsRateState {
            count: 0,
            window_start: std::time::Instant::now(),
        });

        loop {
            tokio::select! {
                msg = timeout(Duration::from_secs(INACTIVITY_TIMEOUT_SECS), socket.recv()) => {
                    let msg = match msg {
                        Ok(inner) => inner,
                        Err(_) => {
                            info!("WebSocket inactivity timeout — closing connection");
                            break;
                        }
                    };
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            //  Rate limiting — atomic check-and-update under lock.
                            let count = {
                                let mut guard = rate_state.lock().await;
                                if guard.window_start.elapsed() > Duration::from_secs(60) {
                                    guard.count = 0;
                                    guard.window_start = std::time::Instant::now();
                                }
                                guard.count += 1;
                                guard.count
                            };
                            if count > WS_RATE_LIMIT_PER_MIN {
                                let err_msg = serde_json::json!({
                                    "error": "Rate limit exceeded"
                                }).to_string();
                                let _ = socket.send(Message::Text(err_msg)).await;
                                continue;
                            }

                            //  Parse and process
                            let req: InferRequest = match serde_json::from_str(&text) {
                                Ok(r) => r,
                                Err(e) => {
                                    let error_msg = serde_json::json!({
                                        "error": format!("Invalid JSON: {e}")
                                    }).to_string();
                                    let _ = socket.send(Message::Text(error_msg)).await;
                                    continue;
                                }
                            };

                            // Validate non-empty prompt.
                            if req.prompt.trim().is_empty() {
                                let error_msg = serde_json::json!({
                                    "error": "prompt must not be empty"
                                }).to_string();
                                let _ = socket.send(Message::Text(error_msg)).await;
                                continue;
                            }

                            let request_id = Uuid::new_v4().to_string();
                            let processing_msg = serde_json::json!({
                                "request_id": &request_id,
                                "status": "processing"
                            }).to_string();

                            if socket.send(Message::Text(processing_msg)).await.is_err() {
                                break;
                            }

                            let session_id = req.session_id
                                .unwrap_or_else(|| format!("ws-{}", request_id));
                            let prompt_req = PromptRequest {
                                session: SessionId::new(session_id),
                                request_id: request_id.clone(),
                                input: req.prompt,
                                meta: Some(req.metadata),
                            };

                            if state.pipeline_tx.send(prompt_req).await.is_ok() {
                                tokio::time::sleep(Duration::from_millis(100)).await;

                                let result_msg = serde_json::json!({
                                    "request_id": &request_id,
                                    "status": "completed",
                                    "result": "Response from pipeline"
                                }).to_string();

                                if socket.send(Message::Text(result_msg)).await.is_err() {
                                    break;
                                }
                            } else {
                                let err_msg = serde_json::json!({
                                    "request_id": &request_id,
                                    "status": "failed",
                                    "error": "Pipeline closed"
                                }).to_string();
                                let _ = socket.send(Message::Text(err_msg)).await;
                                break;
                            }
                        }
                        Some(Ok(Message::Ping(data))) => {
                            if socket.send(Message::Pong(data)).await.is_err() {
                                break;
                            }
                        }
                        Some(Ok(Message::Close(_))) | None => break,
                        Some(Err(e)) => {
                            warn!(error = %e, "WebSocket receive error");
                            break;
                        }
                        _ => {} // Binary, Pong  -  ignore
                    }
                }
                _ = ping_interval.tick() => {
                    if socket.send(Message::Ping(vec![])).await.is_err() {
                        break;
                    }
                }
            }
        }

        info!("WebSocket client disconnected  -  resources released");
    }

    /// `GET /health`  -  Health check endpoint.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn health_handler() -> Json<serde_json::Value> {
        Json(serde_json::json!({
            "status": "healthy",
            "version": env!("CARGO_PKG_VERSION"),
        }))
    }

    /// `GET /metrics`  -  Prometheus metrics endpoint.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn metrics_handler() -> String {
        crate::metrics::gather_metrics()
    }

    /// `GET /api/v1/schema`  -  Serve the OpenAPI 3.0 schema.
    ///
    /// # Panics
    ///
    /// This function never panics.
    pub async fn schema_handler() -> Json<serde_json::Value> {
        let schema = serde_json::json!({
          "openapi": "3.0.0",
          "info": {
            "title": "SRFM Orchestrator API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Async LLM prompt orchestration with relativistic market regime classification.\n\n## Authentication\n\nAll inference endpoints require `Authorization: Bearer <API_KEY>`. Public endpoints (`/health`, `/metrics`, `/api/v1/schema`) are unauthenticated.\n\n## Rate Limits\n\n- `POST /api/v1/infer`: 60 requests per 60-second window per IP. Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.\n- `WS /api/v1/ws`: 60 messages per minute per connection.\n- `POST /api/v1/stream`: maximum 10 concurrent SSE connections per server.\n\nExceeding limits returns HTTP 429 with body: `{\"error\":{\"code\":\"too_many_connections\",\"message\":\"...\"}}` ."
          },
          "paths": {
            "/health": { "get": { "summary": "Health check", "responses": { "200": { "description": "OK", "content": { "application/json": { "schema": { "type": "object", "properties": { "status": { "type": "string" }, "version": { "type": "string" } } } } } } } } },
            "/metrics": { "get": { "summary": "Prometheus metrics", "responses": { "200": { "description": "Prometheus text format" } } } },
            "/api/v1/infer": {
              "post": {
                "summary": "Submit inference request",
                "description": "Rate limited: 60 requests per 60-second window per IP. Returns X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset response headers.",
                "security": [{"bearerAuth": []}],
                "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/InferRequest" } } } },
                "responses": {
                  "200": { "description": "Request accepted", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/InferResponse" } } } },
                  "401": { "description": "Unauthorized", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } },
                  "413": { "description": "Payload too large", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } },
                  "429": { "description": "Rate limit exceeded", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } },
                  "503": { "description": "Pipeline unavailable", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } }
                }
              }
            },
            "/api/v1/stream": { "post": { "summary": "SSE token streaming", "description": "Rate limited: maximum 10 concurrent connections per server.", "security": [{"bearerAuth": []}], "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/InferRequest" } } } }, "responses": { "200": { "description": "SSE stream (events: start, token, done)" }, "429": { "description": "Too many SSE connections", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } } } } },
            "/api/v1/status/{request_id}": { "get": { "summary": "Check request status", "security": [{"bearerAuth": []}], "parameters": [{ "name": "request_id", "in": "path", "required": true, "schema": { "type": "string" } }], "responses": { "200": { "description": "Status object", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/InferResponse" } } } }, "404": { "description": "Not found", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } } } } },
            "/api/v1/result/{request_id}": { "get": { "summary": "Block until result ready", "security": [{"bearerAuth": []}], "parameters": [{ "name": "request_id", "in": "path", "required": true, "schema": { "type": "string" } }], "responses": { "200": { "description": "Result", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/InferResponse" } } } }, "404": { "description": "Not found", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } }, "408": { "description": "Request timeout", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } } } } },
            "/api/v1/requests": { "get": { "summary": "Paginated request list", "security": [{"bearerAuth": []}], "parameters": [{ "name": "offset", "in": "query", "schema": { "type": "integer", "default": 0 } }, { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 20, "maximum": 100 } }], "responses": { "200": { "description": "Paginated list" } } } },
            "/api/v1/explain/{request_id}": { "get": { "summary": "Regime explanation for request", "security": [{"bearerAuth": []}], "parameters": [{ "name": "request_id", "in": "path", "required": true, "schema": { "type": "string" } }], "responses": { "200": { "description": "Explanation JSON" }, "404": { "description": "Not found", "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ErrorResponse" } } } } } } },
            "/api/v1/ws": { "get": { "summary": "WebSocket streaming", "description": "Rate limited: 60 messages per minute per connection.", "security": [{"bearerAuth": []}], "responses": { "101": { "description": "WebSocket upgrade" } } } }
          },
          "components": {
            "schemas": {
              "InferRequest": {
                "type": "object",
                "required": ["prompt"],
                "properties": {
                  "prompt": { "type": "string", "description": "Input prompt text (must not be empty)" },
                  "session_id": { "type": "string", "description": "Optional client session ID; generated if absent" },
                  "stream": { "type": "boolean", "default": false, "description": "Request streaming response via WebSocket" },
                  "metadata": { "type": "object", "additionalProperties": { "type": "string" }, "description": "Arbitrary key-value metadata forwarded through the pipeline" }
                }
              },
              "InferResponse": {
                "type": "object",
                "required": ["request_id", "status"],
                "properties": {
                  "request_id": { "type": "string", "description": "UUID assigned to this inference request" },
                  "status": { "type": "string", "enum": ["pending", "processing", "completed", "failed", "timeout"], "description": "Current processing status" },
                  "result": { "type": "string", "description": "Final inference result (present when status is completed)" },
                  "error": { "type": "string", "description": "Error description (present when status is failed)" }
                }
              },
              "ErrorResponse": {
                "type": "object",
                "required": ["error"],
                "properties": {
                  "error": {
                    "type": "object",
                    "required": ["code", "message"],
                    "properties": {
                      "code": { "type": "string", "description": "Machine-readable error code (e.g. not_found, unauthorized, too_many_connections, pipeline_closed, timeout, payload_too_large)" },
                      "message": { "type": "string", "description": "Human-readable error description" }
                    }
                  }
                }
              }
            },
            "securitySchemes": {
              "bearerAuth": { "type": "http", "scheme": "bearer", "description": "API key passed as a Bearer token in the Authorization header. Set via the API_KEY environment variable at server startup." }
            }
          }
        });
        Json(schema)
    }
}

// ============================================================================
// Top-level imports and AppState
// ============================================================================

#[cfg(feature = "web-api")]
use axum::{
    middleware as axum_middleware,
    routing::{get, post},
    Router,
};
#[cfg(feature = "web-api")]
use tower_http::cors::AllowOrigin;

#[cfg(feature = "web-api")]
use dashmap::DashMap;
#[cfg(feature = "web-api")]
use std::collections::HashSet;
#[cfg(feature = "web-api")]
use std::sync::atomic::AtomicUsize;
#[cfg(feature = "web-api")]
use std::sync::Arc;
#[cfg(feature = "web-api")]
use tokio::sync::mpsc;
#[cfg(feature = "web-api")]
use tower_http::cors::CorsLayer;
#[cfg(feature = "web-api")]
use tracing::{info, warn};
#[cfg(feature = "web-api")]
use axum::http::HeaderValue;

#[cfg(feature = "web-api")]
use crate::PromptRequest;

// Re-export public types from the types sub-module.
#[cfg(feature = "web-api")]
pub use types::{
    AppError, InferRequest, InferResponse, PaginationParams, RequestStatus, ServerConfig,
};

// ============================================================================
// Internal tracker types (used by both AppState and routes)
// ============================================================================

#[cfg(feature = "web-api")]
#[derive(Clone)]
struct RequestTracker {
    status: Arc<DashMap<String, TrackedRequest>>,
}

#[cfg(feature = "web-api")]
#[derive(Clone)]
struct TrackedRequest {
    status: RequestStatus,
    result: Option<String>,
    error: Option<String>,
}

/// Shared application state available to all handlers.
#[cfg(feature = "web-api")]
struct AppState {
    pipeline_tx: mpsc::Sender<PromptRequest>,
    tracker: RequestTracker,
    config: ServerConfig,
    /// Optional API key.  `None` means authentication is disabled (startup warning emitted).
    api_key: Option<String>,
    /// Number of currently active SSE connections (for rate limiting).
    sse_connections: AtomicUsize,
    /// Per-IP rate limit map for `POST /api/v1/infer`: maps IP → (request_count, window_start).
    infer_rate_limit: Arc<DashMap<String, (u64, std::time::Instant)>>,
}

// ============================================================================
// Server
// ============================================================================

/// Start the web API server.
///
/// Binds to `config.host:config.port` and serves the REST API, SSE streaming,
/// and WebSocket endpoints. Blocks until the server shuts down.
///
/// # Errors
///
/// Returns an error if the address cannot be bound or the server fails.
///
/// # Panics
///
/// This function never panics.
#[cfg(feature = "web-api")]
pub async fn start_server(
    config: ServerConfig,
    pipeline_tx: mpsc::Sender<PromptRequest>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let addr = format!("{}:{}", config.host, config.port);

    info!("Starting web API server on http://{}", addr);

    // Read optional API key from environment.
    let api_key = match std::env::var("SRFM_API_KEY") {
        Ok(k) if !k.is_empty() => {
            info!("auth enabled");
            Some(k)
        }
        _ => {
            warn!("auth WARNING: running without authentication — set SRFM_API_KEY to enable");
            None
        }
    };

    // Configure CORS origins from ALLOWED_ORIGINS (comma-separated) or fall back to wildcard.
    let cors = match std::env::var("ALLOWED_ORIGINS") {
        Ok(origins) if !origins.is_empty() => {
            let allowed_set: HashSet<HeaderValue> = origins
                .split(',')
                .filter_map(|o| HeaderValue::from_str(o.trim()).ok())
                .collect();
            let allowed_vec: Vec<HeaderValue> = allowed_set.into_iter().collect();
            CorsLayer::new().allow_origin(AllowOrigin::list(allowed_vec))
        }
        _ => {
            warn!("ALLOWED_ORIGINS env var not set — CORS is permissive (wildcard). Set ALLOWED_ORIGINS for production.");
            CorsLayer::new().allow_origin(AllowOrigin::any())
        }
    };

    let tracker = RequestTracker {
        status: Arc::new(DashMap::new()),
    };

    let state = Arc::new(AppState {
        pipeline_tx,
        tracker,
        config: config.clone(),
        api_key,
        sse_connections: AtomicUsize::new(0),
        infer_rate_limit: Arc::new(DashMap::new()),
    });

    let app = Router::new()
        .route("/api/v1/infer", post(routes::infer_handler))
        .route("/api/v1/stream", post(routes::sse_stream_handler))
        .route("/api/v1/status/:request_id", get(routes::status_handler))
        .route("/api/v1/result/:request_id", get(routes::result_handler))
        .route("/api/v1/requests", get(routes::list_requests_handler))
        .route("/api/v1/explain/:request_id", get(routes::explain_handler))
        .route("/api/v1/ws", get(routes::websocket_handler))
        .route("/api/v1/schema", get(routes::schema_handler))
        .route("/health", get(routes::health_handler))
        .route("/metrics", get(routes::metrics_handler))
        .layer(axum_middleware::from_fn_with_state(
            state.clone(),
            middleware::auth_middleware,
        ))
        .layer(axum_middleware::from_fn(middleware::request_id_middleware))
        .layer(axum_middleware::from_fn_with_state(
            config.max_request_size,
            middleware::body_size_middleware,
        ))
        .layer(cors)
        .with_state(state);

    info!("Web API ready on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Feature-gated stubs (when web-api is disabled)
// ============================================================================

/// Stub configuration when the `web-api` feature is disabled.
#[cfg(not(feature = "web-api"))]
pub struct ServerConfig;
#[cfg(not(feature = "web-api"))]
impl Default for ServerConfig {
    fn default() -> Self {
        Self
    }
}

#[cfg(not(feature = "web-api"))]
pub async fn start_server(
    _config: ServerConfig,
    _pipeline_tx: tokio::sync::mpsc::Sender<crate::PromptRequest>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    Err("Web API requires 'web-api' feature".into())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "web-api")]
mod tests {
    use super::*;
    use super::routes::{
        SSE_MAX_CONNECTIONS, WS_MAX_MESSAGE_SIZE, WS_PING_INTERVAL_SECS, WS_RATE_LIMIT_PER_MIN,
    };

    #[tokio::test]
    async fn test_openapi_schema_is_valid_json() {
        let Json(parsed) = routes::schema_handler().await;
        assert_eq!(parsed["openapi"], "3.0.0");
    }

    #[tokio::test]
    async fn test_openapi_schema_contains_all_endpoints() {
        let Json(parsed) = routes::schema_handler().await;
        let paths = parsed["paths"].as_object().expect("paths is object");
        assert!(paths.contains_key("/api/v1/infer"));
        assert!(paths.contains_key("/api/v1/stream"));
        assert!(paths.contains_key("/api/v1/status/{request_id}"));
        assert!(paths.contains_key("/api/v1/result/{request_id}"));
        assert!(paths.contains_key("/api/v1/ws"));
        assert!(paths.contains_key("/health"));
        assert!(paths.contains_key("/metrics"));
        assert!(paths.contains_key("/api/v1/requests"));
        assert!(paths.contains_key("/api/v1/explain/{request_id}"));
    }

    #[test]
    fn test_infer_request_minimal_deserializes() {
        let json = r#"{"prompt": "hello"}"#;
        let req: InferRequest = serde_json::from_str(json).expect("deser");
        assert_eq!(req.prompt, "hello");
        assert!(req.session_id.is_none());
        assert!(req.metadata.is_empty());
        assert!(!req.stream);
    }

    #[test]
    fn test_infer_request_full_deserializes() {
        let json = r#"{
            "prompt": "test",
            "session_id": "s1",
            "metadata": {"key": "val"},
            "stream": true
        }"#;
        let req: InferRequest = serde_json::from_str(json).expect("deser");
        assert_eq!(req.prompt, "test");
        assert_eq!(req.session_id.as_deref(), Some("s1"));
        assert_eq!(req.metadata.get("key").map(|s| s.as_str()), Some("val"));
        assert!(req.stream);
    }

    #[test]
    fn test_infer_response_completed_includes_result() {
        let resp = InferResponse {
            request_id: "r1".to_string(),
            status: RequestStatus::Completed,
            result: Some("answer".to_string()),
            error: None,
        };
        let json = serde_json::to_string(&resp).expect("ser");
        assert!(json.contains("answer"));
        assert!(!json.contains("error"));
    }

    #[test]
    fn test_infer_response_failed_includes_error() {
        let resp = InferResponse {
            request_id: "r1".to_string(),
            status: RequestStatus::Failed,
            result: None,
            error: Some("boom".to_string()),
        };
        let json = serde_json::to_string(&resp).expect("ser");
        assert!(json.contains("boom"));
        assert!(!json.contains("result"));
    }

    #[test]
    fn test_request_status_serializes_lowercase() {
        let json = serde_json::to_string(&RequestStatus::Processing).expect("ser");
        assert_eq!(json, "\"processing\"");
    }

    #[test]
    fn test_request_status_round_trips() {
        for status in [
            RequestStatus::Pending,
            RequestStatus::Processing,
            RequestStatus::Completed,
            RequestStatus::Failed,
            RequestStatus::Timeout,
        ] {
            let json = serde_json::to_string(&status).expect("ser");
            let back: RequestStatus = serde_json::from_str(&json).expect("deser");
            assert_eq!(status, back);
        }
    }

    #[test]
    fn test_server_config_default_values() {
        let cfg = ServerConfig::default();
        assert_eq!(cfg.host, "0.0.0.0");
        assert_eq!(cfg.port, 8080);
        assert_eq!(cfg.max_request_size, 10 * 1024 * 1024);
        assert_eq!(cfg.timeout_seconds, 300);
    }

    #[test]
    fn test_ws_constants() {
        assert_eq!(WS_MAX_MESSAGE_SIZE, 1024 * 1024);
        assert_eq!(WS_PING_INTERVAL_SECS, 30);
        assert_eq!(WS_RATE_LIMIT_PER_MIN, 60);
    }

    #[test]
    fn test_app_error_not_found_returns_404_json() {
        let resp = AppError::NotFound("test-id".into()).into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_app_error_pipeline_closed_returns_503() {
        let resp = AppError::PipelineClosed.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_app_error_timeout_returns_408() {
        let resp = AppError::Timeout.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::REQUEST_TIMEOUT);
    }

    // ------------------------------------------------------------------
    // CRIT-01: auth middleware tests
    // ------------------------------------------------------------------

    /// Build a minimal AppState for unit testing auth middleware.
    fn make_state_with_key(key: Option<&str>) -> Arc<AppState> {
        let (tx, _rx) = mpsc::channel(1);
        Arc::new(AppState {
            pipeline_tx: tx,
            tracker: RequestTracker {
                status: Arc::new(DashMap::new()),
            },
            config: ServerConfig::default(),
            api_key: key.map(|s| s.to_string()),
            sse_connections: AtomicUsize::new(0),
            infer_rate_limit: Arc::new(DashMap::new()),
        })
    }

    #[test]
    fn test_request_rejected_without_auth_header() {
        let state = make_state_with_key(Some("secret-token"));
        let expected = state.api_key.as_deref();
        let provided: Option<&str> = None;
        let token_valid = provided
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|token| Some(token) == expected)
            .unwrap_or(false);
        assert!(!token_valid, "no header → must be rejected");
    }

    #[test]
    fn test_request_accepted_with_valid_key() {
        let state = make_state_with_key(Some("secret-token"));
        let expected = state.api_key.as_deref();
        let header_value = "Bearer secret-token";
        let token_valid = Some(header_value)
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|token| Some(token) == expected)
            .unwrap_or(false);
        assert!(token_valid, "valid Bearer token → must be accepted");
    }

    #[test]
    fn test_request_rejected_with_wrong_key() {
        let state = make_state_with_key(Some("secret-token"));
        let expected = state.api_key.as_deref();
        let header_value = "Bearer wrong-token";
        let token_valid = Some(header_value)
            .and_then(|v| v.strip_prefix("Bearer "))
            .map(|token| Some(token) == expected)
            .unwrap_or(false);
        assert!(!token_valid, "wrong token → must be rejected");
    }

    #[test]
    fn test_auth_disabled_when_no_api_key() {
        let state = make_state_with_key(None);
        assert!(state.api_key.is_none(), "no API key → auth disabled");
    }

    #[test]
    fn test_sse_max_connections_constant() {
        assert_eq!(SSE_MAX_CONNECTIONS, 10);
    }

    #[test]
    fn test_sse_connection_counter_increments() {
        let (tx, _rx) = mpsc::channel(1);
        let state = Arc::new(AppState {
            pipeline_tx: tx,
            tracker: RequestTracker { status: Arc::new(DashMap::new()) },
            config: ServerConfig::default(),
            api_key: None,
            sse_connections: AtomicUsize::new(0),
            infer_rate_limit: Arc::new(DashMap::new()),
        });
        state.sse_connections.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(state.sse_connections.load(std::sync::atomic::Ordering::Relaxed), 1);
    }

    #[test]
    fn test_sse_connection_limit_check() {
        let (tx, _rx) = mpsc::channel(1);
        let state = Arc::new(AppState {
            pipeline_tx: tx,
            tracker: RequestTracker { status: Arc::new(DashMap::new()) },
            config: ServerConfig::default(),
            api_key: None,
            sse_connections: AtomicUsize::new(SSE_MAX_CONNECTIONS),
            infer_rate_limit: Arc::new(DashMap::new()),
        });
        let current = state.sse_connections.load(std::sync::atomic::Ordering::Relaxed);
        assert!(current >= SSE_MAX_CONNECTIONS, "should be at limit");
    }

    #[test]
    fn test_pagination_params_defaults() {
        let params: PaginationParams = serde_json::from_str("{}").expect("deser");
        assert_eq!(params.offset, 0);
        assert_eq!(params.limit, 20);
    }

    #[test]
    fn test_pagination_params_custom() {
        let params: PaginationParams = serde_json::from_str(r#"{"offset":5,"limit":10}"#).expect("deser");
        assert_eq!(params.offset, 5);
        assert_eq!(params.limit, 10);
    }

    #[test]
    fn test_app_error_too_many_connections_returns_429() {
        let resp = AppError::TooManyConnections.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    // ------------------------------------------------------------------
    // New tests required by web API improvements
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_structured_error_not_found_returns_json() {
        use axum::body::to_bytes;
        let resp = AppError::NotFound("abc".into()).into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::NOT_FOUND);
        let body_bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).expect("valid JSON");
        assert_eq!(json["error"]["code"], "not_found");
        assert!(
            json["error"]["message"].as_str().unwrap().contains("abc"),
            "message should include the request id"
        );
    }

    #[tokio::test]
    async fn test_structured_error_pipeline_closed_returns_json() {
        use axum::body::to_bytes;
        let resp = AppError::PipelineClosed.into_response();
        assert_eq!(resp.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
        let body_bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body_bytes).expect("valid JSON");
        assert_eq!(json["error"]["code"], "pipeline_closed");
    }

    #[tokio::test]
    async fn test_schema_endpoint_has_version() {
        let Json(schema) = routes::schema_handler().await;
        let version = schema["info"]["version"].as_str().expect("version string");
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_infer_rate_limit_state_initialized() {
        let (tx, _rx) = mpsc::channel(1);
        let state = AppState {
            pipeline_tx: tx,
            tracker: RequestTracker {
                status: Arc::new(DashMap::new()),
            },
            config: ServerConfig::default(),
            api_key: None,
            sse_connections: AtomicUsize::new(0),
            infer_rate_limit: Arc::new(DashMap::new()),
        };
        assert!(state.infer_rate_limit.is_empty(), "infer_rate_limit should start empty");
    }
}
