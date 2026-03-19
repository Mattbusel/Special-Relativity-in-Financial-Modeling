//! # Module: TUI Dashboard
//!
//! ## Responsibility
//! Provides a production-grade ASCII terminal dashboard for the tokio-prompt-orchestrator
//! using Ratatui. This is the primary demo and monitoring interface, rendering pipeline
//! flow, channel depths, circuit breaker states, deduplication stats, throughput
//! sparklines, system health, and a log tail in real-time.
//!
//! ## Guarantees
//! - No panics in any rendering or update path
//! - Clean terminal restore on exit, including on panic
//! - Graceful resize handling down to 100x40 minimum
//! - 10fps rendering with 1hz data updates
//!
//! ## NOT Responsible For
//! - Orchestrator pipeline execution (read-only monitoring)
//! - Persistent storage of metrics (ephemeral display only)
//! - Network protocol implementation (delegates to metrics module)

pub mod app;
pub mod events;
pub mod metrics;
pub mod ui;
pub mod widgets;

/// Run the TUI dashboard in mock-data mode.
///
/// Sets up the terminal, runs the event loop at 10 fps with 1 Hz data updates
/// driven by [`metrics::MockMetrics`], then restores the terminal on exit.
///
/// # Errors
/// Returns an error if the terminal cannot be initialised or if rendering fails.
pub async fn run_mock() -> Result<(), Box<dyn std::error::Error>> {
    use crossterm::{
        execute,
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    };
    use ratatui::{backend::CrosstermBackend, Terminal};
    use std::io;
    use std::time::{Duration, Instant};

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Restore terminal on panic.
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        default_panic(info);
    }));

    const RENDER_RATE: Duration = Duration::from_millis(100); // 10 fps
    const DATA_RATE: Duration = Duration::from_secs(1);

    let mut app_state = app::App::new(DATA_RATE);
    let mock = metrics::MockMetrics::new();
    let mut last_data_tick = Instant::now();

    // Startup animation: 0 → 100 % over ~15 render frames (~1.5 s).
    app_state.loading_progress = Some(0);

    loop {
        // Advance startup progress bar.
        if let Some(ref mut p) = app_state.loading_progress {
            *p = (*p).saturating_add(8).min(100);
            if *p >= 100 {
                app_state.loading_progress = None;
            }
        }

        terminal.draw(|f| ui::draw(f, &app_state))?;

        let event = events::poll_event(RENDER_RATE);
        events::apply_event(&mut app_state, event);

        if app_state.should_quit {
            break;
        }

        if app_state.loading_progress.is_none()
            && !app_state.paused
            && last_data_tick.elapsed() >= DATA_RATE
        {
            mock.tick(&mut app_state);
            last_data_tick = Instant::now();
        }
    }

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    Ok(())
}
