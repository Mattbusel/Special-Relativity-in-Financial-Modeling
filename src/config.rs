//! # Configuration
//!
//! Loads `srfm.toml` from the current directory or `~/.config/srfm/srfm.toml`.
//! All fields are optional and fall back to environment variables or defaults.

use serde::{Deserialize, Serialize};

/// Top-level configuration parsed from `srfm.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// Web API server settings.
    #[serde(default)]
    pub server: ServerConfig,
    /// LLM worker settings.
    #[serde(default)]
    pub workers: WorkersConfig,
    /// TUI dashboard settings.
    #[serde(default)]
    pub tui: TuiConfig,
}

/// Web API server settings parsed from `[server]` in `srfm.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind host address (default: `"0.0.0.0"`).
    #[serde(default = "default_host")]
    pub host: String,
    /// Bind port (default: `8080`).
    #[serde(default = "default_port")]
    pub port: u16,
    /// Request timeout in seconds (default: `300`).
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            timeout_seconds: default_timeout_seconds(),
        }
    }
}

fn default_host() -> String { "0.0.0.0".into() }
fn default_port() -> u16 { 8080 }
fn default_timeout_seconds() -> u64 { 300 }

/// LLM worker settings parsed from `[workers]` in `srfm.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkersConfig {
    /// Override OpenAI base URL (default: https://api.openai.com).
    pub openai_base_url: Option<String>,
    /// Override Anthropic base URL.
    pub anthropic_base_url: Option<String>,
    /// llama.cpp server URL.
    pub llama_cpp_url: Option<String>,
    /// vLLM server URL.
    pub vllm_url: Option<String>,
    /// Default max tokens for inference.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    /// Default temperature.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
}

fn default_max_tokens() -> u32 { 512 }
fn default_temperature() -> f64 { 0.7 }

/// TUI dashboard settings parsed from `[tui]` in `srfm.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiConfig {
    /// Render rate in milliseconds (default: 100 = 10fps).
    #[serde(default = "default_render_ms")]
    pub render_ms: u64,
    /// Data update interval in milliseconds (default: 1000 = 1hz).
    #[serde(default = "default_data_ms")]
    pub data_ms: u64,
    /// Initial left-column width percentage (30–70, default 50).
    #[serde(default = "default_left_col_pct")]
    pub left_col_pct: u16,
}

impl Default for TuiConfig {
    fn default() -> Self {
        Self {
            render_ms: default_render_ms(),
            data_ms: default_data_ms(),
            left_col_pct: default_left_col_pct(),
        }
    }
}

fn default_render_ms() -> u64 { 100 }
fn default_data_ms() -> u64 { 1000 }
fn default_left_col_pct() -> u16 { 50 }

impl Config {
    /// Load config from `srfm.toml` in the current directory, or return defaults.
    ///
    /// Never fails — missing or malformed config falls back to defaults with a warning.
    pub fn load() -> Self {
        Self::load_from_path("srfm.toml")
    }

    /// Load config from a specific path.
    pub fn load_from_path(path: &str) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => match toml::from_str(&contents) {
                Ok(cfg) => cfg,
                Err(e) => {
                    eprintln!("Warning: failed to parse {}: {}", path, e);
                    Self::default()
                }
            },
            Err(_) => Self::default(), // file not found is normal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_values() {
        let cfg = Config::default();
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert_eq!(cfg.server.port, 8080);
        assert_eq!(cfg.workers.max_tokens, 512);
        assert!((cfg.workers.temperature - 0.7).abs() < 1e-9);
        assert_eq!(cfg.tui.render_ms, 100);
        assert_eq!(cfg.tui.left_col_pct, 50);
    }

    #[test]
    fn test_config_load_missing_file_returns_defaults() {
        let cfg = Config::load_from_path("nonexistent_file_xyz.toml");
        assert_eq!(cfg.server.port, 8080);
    }

    #[test]
    fn test_config_load_from_toml_string() {
        let toml_str = r#"
[server]
port = 9090
host = "127.0.0.1"

[workers]
max_tokens = 1024
temperature = 0.5
"#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.server.port, 9090);
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.workers.max_tokens, 1024);
        assert!((cfg.workers.temperature - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_config_partial_toml_uses_defaults_for_missing() {
        let toml_str = "[server]\nport = 7777\n";
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.server.port, 7777);
        assert_eq!(cfg.server.host, "0.0.0.0"); // default preserved
        assert_eq!(cfg.workers.max_tokens, 512); // default preserved
    }

    #[test]
    fn test_config_malformed_toml_returns_defaults() {
        use tempfile::NamedTempFile;
        use std::io::Write;
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "this is not valid toml !!!").unwrap();
        let cfg = Config::load_from_path(f.path().to_str().unwrap());
        assert_eq!(cfg.server.port, 8080);
    }
}
