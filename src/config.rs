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
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for WorkersConfig {
    fn default() -> Self {
        Self {
            openai_base_url: None,
            anthropic_base_url: None,
            llama_cpp_url: None,
            vllm_url: None,
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
        }
    }
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
    /// Load config from the first `srfm.toml` found in the standard search paths.
    ///
    /// Search order:
    /// 1. `./srfm.toml` (current directory)
    /// 2. `{config_dir}/srfm/srfm.toml` (platform config dir via `dirs`)
    /// 3. `~/.config/srfm/srfm.toml` (XDG-style fallback)
    ///
    /// Never fails — missing or malformed config falls back to defaults.
    pub fn load() -> Self {
        let candidates: Vec<std::path::PathBuf> = {
            let mut paths = vec![std::path::PathBuf::from("srfm.toml")];
            if let Some(cfg_dir) = dirs::config_dir() {
                paths.push(cfg_dir.join("srfm").join("srfm.toml"));
            }
            if let Some(home_dir) = dirs::home_dir() {
                paths.push(home_dir.join(".config").join("srfm").join("srfm.toml"));
            }
            paths
        };
        for candidate in &candidates {
            if candidate.exists() {
                return Self::load_from_path(candidate.to_str().unwrap_or("srfm.toml"));
            }
        }
        Self::default()
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

    /// Validate all config fields are within acceptable ranges.
    ///
    /// Returns `Ok(())` if all fields are valid, or `Err(Vec<String>)` listing
    /// every validation error found.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.server.port == 0 {
            errors.push("server.port must be > 0".into());
        }
        if self.tui.render_ms == 0 {
            errors.push("tui.render_ms must be > 0".into());
        }
        if self.tui.left_col_pct < 30 || self.tui.left_col_pct > 70 {
            errors.push(format!(
                "tui.left_col_pct must be 30-70, got {}",
                self.tui.left_col_pct
            ));
        }
        if self.workers.temperature < 0.0 || self.workers.temperature > 2.0 {
            errors.push(format!(
                "workers.temperature must be 0.0-2.0, got {}",
                self.workers.temperature
            ));
        }
        if self.server.timeout_seconds == 0 {
            errors.push("server.timeout_seconds must be > 0".into());
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

/// Watch `path` for changes and call `on_change` with the newly loaded [`Config`]
/// whenever the file is modified.
///
/// Returns a [`notify::RecommendedWatcher`] that must be kept alive for watching
/// to continue. Drop it to stop watching.
///
/// # Errors
///
/// Returns `Err` if the watcher cannot be created or the path cannot be watched.
#[cfg(feature = "watch-config")]
pub fn watch_config(
    path: &str,
    on_change: impl Fn(Config) + Send + 'static,
) -> notify::Result<notify::RecommendedWatcher> {
    use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
    use std::path::PathBuf;

    let watch_path = PathBuf::from(path);
    let config_path = path.to_string();

    let mut watcher = RecommendedWatcher::new(
        move |result: notify::Result<notify::Event>| {
            if let Ok(event) = result {
                let relevant = matches!(
                    event.kind,
                    EventKind::Modify(_) | EventKind::Create(_)
                );
                if relevant {
                    let cfg = Config::load_from_path(&config_path);
                    on_change(cfg);
                }
            }
        },
        notify::Config::default(),
    )?;

    watcher.watch(&watch_path, RecursiveMode::NonRecursive)?;
    Ok(watcher)
}

/// Watch `path` for changes — no-op stub when `watch-config` feature is disabled.
#[cfg(not(feature = "watch-config"))]
pub fn watch_config(
    path: &str,
    on_change: impl Fn(Config) + Send + 'static,
) -> Result<(), String> {
    let _ = (path, on_change);
    Err("watch_config requires the `watch-config` feature or the `notify` crate as a regular dependency".into())
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
    fn test_config_validate_valid() {
        let cfg = Config::default();
        assert!(cfg.validate().is_ok(), "default config should be valid");
    }

    #[test]
    fn test_config_validate_invalid_port() {
        let mut cfg = Config::default();
        cfg.server.port = 0;
        let result = cfg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("server.port")),
            "errors should mention server.port, got {:?}",
            errors
        );
    }

    #[test]
    fn test_config_validate_invalid_col_pct() {
        let mut cfg = Config::default();
        cfg.tui.left_col_pct = 20; // below minimum of 30
        let result = cfg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("left_col_pct")),
            "errors should mention left_col_pct, got {:?}",
            errors
        );

        cfg.tui.left_col_pct = 80; // above maximum of 70
        let result = cfg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("left_col_pct")),
            "errors should mention left_col_pct, got {:?}",
            errors
        );
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

    #[test]
    fn test_config_validate_multiple_simultaneous_errors() {
        let mut cfg = Config::default();
        cfg.server.port = 0;                // triggers port error
        cfg.tui.left_col_pct = 10;          // triggers left_col_pct error
        let result = cfg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.len() >= 2, "expected at least 2 errors, got {:?}", errors);
        assert!(errors.iter().any(|e| e.contains("server.port")));
        assert!(errors.iter().any(|e| e.contains("left_col_pct")));
    }

    #[test]
    fn test_config_validate_temperature_out_of_range() {
        let mut cfg = Config::default();
        cfg.workers.temperature = -0.1; // below minimum 0.0
        let result = cfg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("workers.temperature")),
            "errors should mention workers.temperature, got {:?}",
            errors
        );

        cfg.workers.temperature = 2.5; // above maximum 2.0
        let result = cfg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("workers.temperature")),
            "errors should mention workers.temperature, got {:?}",
            errors
        );
    }

    #[test]
    fn test_config_validate_timeout_seconds_zero() {
        let mut cfg = Config::default();
        cfg.server.timeout_seconds = 0;
        let result = cfg.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains("server.timeout_seconds")),
            "errors should mention server.timeout_seconds, got {:?}",
            errors
        );
    }

    #[test]
    fn test_config_load_malformed_toml_falls_back_to_defaults() {
        use tempfile::NamedTempFile;
        use std::io::Write;
        let mut f = NamedTempFile::new().unwrap();
        // Write clearly malformed TOML
        writeln!(f, "[server]\nport = !!!invalid").unwrap();
        let cfg = Config::load_from_path(f.path().to_str().unwrap());
        // Must fall back to defaults gracefully
        assert_eq!(cfg.server.port, 8080, "malformed TOML should yield default port");
        assert_eq!(cfg.server.host, "0.0.0.0");
        assert!(cfg.validate().is_ok(), "default config should be valid");
    }
}
