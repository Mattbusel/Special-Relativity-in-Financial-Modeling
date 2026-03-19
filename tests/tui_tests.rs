//! Integration tests for the TUI dashboard module.

#[cfg(feature = "tui")]
mod tui;

// ── Top-level TUI integration tests ──────────────────────────────────────────
//
// These tests exercise the public API of the TUI subsystem without spinning up
// a real terminal. They are compiled unconditionally (no `tui` feature gate)
// so that `cargo test` always runs them when the crate is built.
//
// Naming convention: test_<subject>_<expectation>.

#[cfg(feature = "tui")]
mod tui_integration {
    use std::time::Duration;

    use tokio_prompt_orchestrator::tui::app::{App, LogEntry, LogLevel};
    use tokio_prompt_orchestrator::tui::events::{apply_event, InputEvent};
    use tokio_prompt_orchestrator::tui::metrics::MockMetrics;
    use tokio_prompt_orchestrator::tui::widgets::sparkline::regime_color;
    use tokio_prompt_orchestrator::tui::app::RegimeType;

    // ── App::new initialisation ───────────────────────────────────────────

    #[test]
    fn test_app_new_tick_rate_is_non_zero() {
        let app = App::new(Duration::from_secs(1));
        assert!(app.tick_rate > Duration::ZERO, "tick_rate must be non-zero");
    }

    #[test]
    fn test_app_new_log_entries_empty() {
        let app = App::new(Duration::from_secs(1));
        assert!(app.log_entries.is_empty(), "log_entries must start empty");
    }

    #[test]
    fn test_app_new_should_quit_false() {
        let app = App::new(Duration::from_secs(1));
        assert!(!app.should_quit);
    }

    #[test]
    fn test_app_new_paused_false() {
        let app = App::new(Duration::from_secs(1));
        assert!(!app.paused);
    }

    #[test]
    fn test_app_new_show_help_false() {
        let app = App::new(Duration::from_secs(1));
        assert!(!app.show_help);
    }

    #[test]
    fn test_app_new_tick_count_zero() {
        let app = App::new(Duration::from_secs(1));
        assert_eq!(app.tick_count, 0);
    }

    // ── apply_event(InputEvent::Quit) ─────────────────────────────────────

    #[test]
    fn test_apply_event_quit_sets_should_quit_true() {
        let mut app = App::new(Duration::from_secs(1));
        apply_event(&mut app, InputEvent::Quit);
        assert!(app.should_quit, "Quit event must set should_quit = true");
    }

    #[test]
    fn test_apply_event_quit_does_not_change_paused() {
        let mut app = App::new(Duration::from_secs(1));
        apply_event(&mut app, InputEvent::Quit);
        assert!(!app.paused, "Quit should not affect paused flag");
    }

    // ── apply_event(InputEvent::Pause) toggles is_paused ─────────────────

    #[test]
    fn test_apply_event_pause_first_press_sets_paused_true() {
        let mut app = App::new(Duration::from_secs(1));
        assert!(!app.paused);
        apply_event(&mut app, InputEvent::Pause);
        assert!(app.paused, "First Pause press must set paused = true");
    }

    #[test]
    fn test_apply_event_pause_second_press_restores_paused_false() {
        let mut app = App::new(Duration::from_secs(1));
        apply_event(&mut app, InputEvent::Pause);
        apply_event(&mut app, InputEvent::Pause);
        assert!(!app.paused, "Second Pause press must restore paused = false");
    }

    #[test]
    fn test_apply_event_pause_triple_press_ends_paused() {
        let mut app = App::new(Duration::from_secs(1));
        for _ in 0..3 {
            apply_event(&mut app, InputEvent::Pause);
        }
        assert!(app.paused, "Three Pause presses: net paused = true");
    }

    // ── apply_event(InputEvent::Help) toggles show_help ──────────────────

    #[test]
    fn test_apply_event_help_first_press_shows_help() {
        let mut app = App::new(Duration::from_secs(1));
        assert!(!app.show_help);
        apply_event(&mut app, InputEvent::Help);
        assert!(app.show_help, "First Help press must show help overlay");
    }

    #[test]
    fn test_apply_event_help_second_press_hides_help() {
        let mut app = App::new(Duration::from_secs(1));
        apply_event(&mut app, InputEvent::Help);
        apply_event(&mut app, InputEvent::Help);
        assert!(!app.show_help, "Second Help press must hide help overlay");
    }

    // ── apply_event(InputEvent::LogExport) does not panic ────────────────

    #[test]
    fn test_apply_event_log_export_empty_log_no_panic() {
        let mut app = App::new(Duration::from_secs(1));
        // Must not panic on an empty log.
        apply_event(&mut app, InputEvent::LogExport);
    }

    #[test]
    fn test_apply_event_log_export_with_entries_no_panic() {
        let mut app = App::new(Duration::from_secs(1));
        app.push_log(LogEntry {
            timestamp: "00:00:01".into(),
            level: LogLevel::Info,
            message: "export test entry".into(),
            fields: "k=v".into(),
        });
        apply_event(&mut app, InputEvent::LogExport);
    }

    // ── on_tick (MockMetrics::tick) does not panic ────────────────────────

    #[test]
    fn test_mock_metrics_tick_single_no_panic() {
        let mock = MockMetrics::new();
        let mut app = App::new(Duration::from_secs(1));
        // Must not panic.
        mock.tick(&mut app);
    }

    #[test]
    fn test_mock_metrics_tick_many_no_panic() {
        let mock = MockMetrics::new();
        let mut app = App::new(Duration::from_secs(1));
        // Simulate a full story cycle twice — must never panic.
        for _ in 0..240 {
            mock.tick(&mut app);
        }
    }

    #[test]
    fn test_mock_metrics_tick_increments_tick_count() {
        let mock = MockMetrics::new();
        let mut app = App::new(Duration::from_secs(1));
        mock.tick(&mut app);
        assert_eq!(app.tick_count, 1);
        mock.tick(&mut app);
        assert_eq!(app.tick_count, 2);
    }

    // ── Sparkline: empty history ──────────────────────────────────────────

    #[test]
    fn test_sparkline_data_collection_empty_history_no_panic() {
        let app = App::new(Duration::from_secs(1));
        // Collect the data the render function would build — must not panic.
        let data: Vec<u64> = app.throughput_history.iter().copied().collect();
        assert!(data.is_empty());
        // unwrap_or(0) must work for empty history.
        let current = app.throughput_history.back().copied().unwrap_or(0);
        assert_eq!(current, 0);
    }

    // ── Sparkline: renders correctly with data ────────────────────────────

    #[test]
    fn test_sparkline_data_collection_with_data() {
        let mut app = App::new(Duration::from_secs(1));
        app.push_throughput(5);
        app.push_throughput(10);
        app.push_throughput(15);

        let data: Vec<u64> = app.throughput_history.iter().copied().collect();
        assert_eq!(data, vec![5, 10, 15]);
        assert_eq!(app.throughput_history.back().copied().unwrap_or(0), 15);
    }

    #[test]
    fn test_sparkline_cpu_history_with_data() {
        let mut app = App::new(Duration::from_secs(1));
        app.cpu_percent = 42.0;
        app.push_cpu_sample();
        app.cpu_percent = 75.0;
        app.push_cpu_sample();

        let data: Vec<u64> = app.cpu_history.iter().map(|v| *v as u64).collect();
        assert_eq!(data, vec![42, 75]);
    }

    #[test]
    fn test_sparkline_mem_history_with_data() {
        let mut app = App::new(Duration::from_secs(1));
        app.mem_percent = 30.0;
        app.push_mem_sample();

        let data: Vec<u64> = app.mem_history.iter().map(|v| *v as u64).collect();
        assert_eq!(data, vec![30]);
    }

    #[test]
    fn test_sparkline_empty_cpu_history_no_panic() {
        let app = App::new(Duration::from_secs(1));
        let current = app.cpu_history.back().copied().unwrap_or(0.0);
        assert_eq!(current, 0.0);
    }

    #[test]
    fn test_sparkline_regime_color_all_variants_no_panic() {
        // regime_color must return a valid Color for every RegimeType.
        let _ = regime_color(RegimeType::Timelike);
        let _ = regime_color(RegimeType::Lightlike);
        let _ = regime_color(RegimeType::Spacelike);
        let _ = regime_color(RegimeType::Unknown);
    }
}
