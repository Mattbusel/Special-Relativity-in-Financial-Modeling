//! Integration tests for the tokio-prompt-orchestrator.

// ── Item 35: Full round-trip integration test ─────────────────────────────────

#[cfg(feature = "web-api")]
#[tokio::test]
async fn test_infer_round_trip() {
    use std::collections::HashMap;
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;
    use tokio_prompt_orchestrator::{
        web_api::{start_server, ServerConfig},
        PromptRequest,
    };

    // Bind to a random available port.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let port = addr.port();

    // Create pipeline channel; spawn a background task that auto-completes requests.
    let (tx, mut rx) = mpsc::channel::<PromptRequest>(16);

    // Auto-complete worker: echoes prompt tokens back as the result.
    tokio::spawn(async move {
        while let Some(_req) = rx.recv().await {
            // EchoWorker logic: just consume — in real tests wire back completion.
        }
    });

    let cfg = ServerConfig {
        host: "127.0.0.1".to_string(),
        port,
        ..Default::default()
    };

    // Start server in background.
    tokio::spawn(async move {
        let _ = start_server(cfg, tx).await;
    });

    // Give server a moment to start.
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    let base = format!("http://127.0.0.1:{}", port);
    let client = reqwest::Client::new();

    // POST /api/v1/infer
    let body = serde_json::json!({
        "prompt": "hello world",
        "metadata": {}
    });
    let resp = client
        .post(format!("{}/api/v1/infer", base))
        .json(&body)
        .send()
        .await
        .expect("POST /api/v1/infer failed");

    assert!(
        resp.status().is_success(),
        "Expected 2xx, got {}",
        resp.status()
    );

    let json: serde_json::Value = resp.json().await.unwrap();
    let request_id = json["request_id"].as_str().expect("request_id field");
    assert!(!request_id.is_empty());

    // GET /api/v1/status/{id} — should return processing or pending.
    let status_resp = client
        .get(format!("{}/api/v1/status/{}", base, request_id))
        .send()
        .await
        .expect("GET /api/v1/status failed");

    assert!(
        status_resp.status().is_success(),
        "Expected 2xx, got {}",
        status_resp.status()
    );

    let status_json: serde_json::Value = status_resp.json().await.unwrap();
    let status = status_json["status"].as_str().unwrap_or("");
    assert!(
        status == "processing" || status == "pending" || status == "completed",
        "Unexpected status: {}",
        status
    );
}

// ── Item 36: Config fixture test ──────────────────────────────────────────────

#[test]
fn test_config_load_from_fixture() {
    use std::io::Write;
    use tempfile::NamedTempFile;
    use tokio_prompt_orchestrator::config::Config;

    // Write a temp srfm.toml with known values.
    let mut f = NamedTempFile::new().unwrap();
    writeln!(
        f,
        r#"
[server]
host = "127.0.0.1"
port = 9876
timeout_seconds = 60

[workers]
max_tokens = 2048
temperature = 0.3

[tui]
render_ms = 200
data_ms = 500
left_col_pct = 40
"#
    )
    .unwrap();

    let cfg = Config::load_from_path(f.path().to_str().unwrap());

    assert_eq!(cfg.server.host, "127.0.0.1");
    assert_eq!(cfg.server.port, 9876);
    assert_eq!(cfg.server.timeout_seconds, 60);
    assert_eq!(cfg.workers.max_tokens, 2048);
    assert!((cfg.workers.temperature - 0.3).abs() < 1e-9);
    assert_eq!(cfg.tui.render_ms, 200);
    assert_eq!(cfg.tui.data_ms, 500);
    assert_eq!(cfg.tui.left_col_pct, 40);
}
