//! # Widget: Throughput Sparkline
//!
//! ## Responsibility
//! Renders a throughput sparkline chart showing requests per second over the
//! last 60 seconds using Ratatui's built-in Sparkline widget. Also provides a
//! multi-sparkline layout showing CPU, memory, and throughput side by side.
//!
//! ## Guarantees
//! - Handles empty data gracefully (shows empty chart)
//! - Y-axis adapts to current max value
//! - Never panics on any data range

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, Sparkline as RatatuiSparkline};
use ratatui::Frame;

use crate::tui::app::{App, RegimeType};

/// Maps a regime to a display color.
pub fn regime_color(regime: RegimeType) -> Color {
    match regime {
        RegimeType::Timelike => Color::Green,
        RegimeType::Lightlike => Color::Yellow,
        RegimeType::Spacelike => Color::Red,
        RegimeType::Unknown => Color::Cyan,
    }
}

/// Renders the throughput sparkline widget.
///
/// # Arguments
/// * `f` - Ratatui frame to render into.
/// * `area` - Rectangular area allocated for this widget.
/// * `app` - Application state containing throughput history.
pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let current_rps = app.throughput_history.back().copied().unwrap_or(0);
    let symbol = app.current_symbol();
    let color = regime_color(app.regime);
    let title = format!(
        " THROUGHPUT ({} req/s) | {} | \u{03b2}={:.3} \u{03b3}={:.2} [{}] ",
        current_rps, symbol, app.beta_display, app.gamma_display, app.regime.label()
    );

    let block = Block::default()
        .title(Span::styled(
            title,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let data: Vec<u64> = app.throughput_history.iter().copied().collect();

    let sparkline = RatatuiSparkline::default()
        .block(block)
        .data(&data)
        .style(Style::default().fg(color));

    f.render_widget(sparkline, area);
}

/// Renders three mini sparklines side by side: CPU%, MEM%, and throughput.
///
/// # Arguments
/// * `f` - Ratatui frame to render into.
/// * `area` - Rectangular area allocated for this widget. Will be split into thirds.
/// * `app` - Application state containing history data.
pub fn render_multi_sparkline(f: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(area);

    // CPU sparkline
    let cpu_current = app.cpu_history.back().copied().unwrap_or(0.0);
    let cpu_block = Block::default()
        .title(Span::styled(
            format!(" CPU ({:.0}%) ", cpu_current),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let cpu_data: Vec<u64> = app.cpu_history.iter().map(|v| *v as u64).collect();
    let cpu_sparkline = RatatuiSparkline::default()
        .block(cpu_block)
        .data(&cpu_data)
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(cpu_sparkline, chunks[0]);

    // Memory sparkline
    let mem_current = app.mem_history.back().copied().unwrap_or(0.0);
    let mem_block = Block::default()
        .title(Span::styled(
            format!(" MEM ({:.0}%) ", mem_current),
            Style::default()
                .fg(Color::Magenta)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let mem_data: Vec<u64> = app.mem_history.iter().map(|v| *v as u64).collect();
    let mem_sparkline = RatatuiSparkline::default()
        .block(mem_block)
        .data(&mem_data)
        .style(Style::default().fg(Color::Magenta));
    f.render_widget(mem_sparkline, chunks[1]);

    // Throughput sparkline
    let current_rps = app.throughput_history.back().copied().unwrap_or(0);
    let tput_block = Block::default()
        .title(Span::styled(
            format!(" RPS ({}) ", current_rps),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let tput_data: Vec<u64> = app.throughput_history.iter().copied().collect();
    let tput_sparkline = RatatuiSparkline::default()
        .block(tput_block)
        .data(&tput_data)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(tput_sparkline, chunks[2]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_empty_throughput_no_panic() {
        let app = App::new(Duration::from_secs(1));
        assert!(app.throughput_history.is_empty());
        // The render function uses unwrap_or(0) for empty data
        let current = app.throughput_history.back().copied().unwrap_or(0);
        assert_eq!(current, 0);
    }

    #[test]
    fn test_throughput_data_collection() {
        let mut app = App::new(Duration::from_secs(1));
        app.push_throughput(10);
        app.push_throughput(20);
        app.push_throughput(15);
        let data: Vec<u64> = app.throughput_history.iter().copied().collect();
        assert_eq!(data, vec![10, 20, 15]);
    }

    #[test]
    fn test_regime_color_timelike_green() {
        assert_eq!(regime_color(RegimeType::Timelike), Color::Green);
    }

    #[test]
    fn test_regime_color_lightlike_yellow() {
        assert_eq!(regime_color(RegimeType::Lightlike), Color::Yellow);
    }

    #[test]
    fn test_regime_color_spacelike_red() {
        assert_eq!(regime_color(RegimeType::Spacelike), Color::Red);
    }

    #[test]
    fn test_regime_color_unknown_cyan() {
        assert_eq!(regime_color(RegimeType::Unknown), Color::Cyan);
    }

    #[test]
    fn test_current_rps_is_last_value() {
        let mut app = App::new(Duration::from_secs(1));
        app.push_throughput(10);
        app.push_throughput(25);
        assert_eq!(app.throughput_history.back().copied().unwrap_or(0), 25);
    }
}
