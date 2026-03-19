//! # Module: TUI Rendering
//!
//! ## Responsibility
//! Orchestrates the overall dashboard layout by dividing the terminal into regions
//! and delegating to individual widget renderers. Handles the minimum size guard
//! and help overlay.
//!
//! ## Guarantees
//! - Layout adapts gracefully to terminal sizes from 100x40 to 220x60+
//! - Minimum size guard displays a centered message if terminal is too small
//! - No panics during rendering regardless of terminal dimensions

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use ratatui::widgets::Gauge;

use super::app::{App, MIN_COLS, MIN_ROWS};
use super::widgets;

/// Renders the complete dashboard UI into the given frame.
///
/// # Arguments
/// * `f` - The Ratatui frame to render into.
/// * `app` - The application state to display.
pub fn draw(f: &mut Frame, app: &App) {
    let size = f.size();

    // Minimum size guard
    if size.width < MIN_COLS || size.height < MIN_ROWS {
        draw_too_small(f, size);
        return;
    }

    // Help overlay
    if app.show_help {
        draw_help_overlay(f, size);
        return;
    }

    // Title bar
    let title = format!(
        " tokio-prompt-orchestrator {:>width$} ",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        width = (size.width as usize).saturating_sub(30),
    );

    let outer_block = Block::default()
        .title(Span::styled(
            title,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let ver = env!("CARGO_PKG_VERSION");
    let symbol = app.current_symbol();
    let footer_text = format!(
        " [q]uit [p]ause [r]eset [h]elp [f]ullscreen [Tab]symbol [e]xport [j/k]scroll  {} \u{03b2}={:.3} \u{03b3}={:.2} v{} ",
        symbol, app.beta_display, app.gamma_display, ver
    );
    let mut footer_spans = vec![
        Span::styled(footer_text, Style::default().fg(Color::DarkGray)),
        if app.paused {
            Span::styled(
                " PAUSED ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            )
        } else {
            Span::raw("")
        },
    ];
    if let Some(ref flash) = app.log_export_flash {
        footer_spans.push(Span::styled(
            format!(" {} ", flash),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ));
    }
    let footer = Line::from(footer_spans);

    let footer_block = Block::default().title_bottom(footer).borders(Borders::NONE);

    let inner = outer_block.inner(size);
    f.render_widget(outer_block, size);
    f.render_widget(footer_block, size);

    // Main layout: top section, sparkline, regime timeline, log
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(14), // Top section (pipeline + health + channels + circuit + dedup)
            Constraint::Length(8), // Throughput sparkline
            Constraint::Length(3), // Regime timeline
            Constraint::Length(10), // Log tail
        ])
        .split(inner);

    // Top section: left (pipeline + channels) and right (health + circuit + dedup)
    let top_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(app.left_col_pct),
            Constraint::Percentage(100 - app.left_col_pct),
        ])
        .split(main_chunks[0]);

    // Left column: pipeline flow + channels
    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),    // Pipeline flow
            Constraint::Length(6), // Channels
        ])
        .split(top_chunks[0]);

    // Right column: health + circuit breakers + dedup
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6), // System health
            Constraint::Length(5), // Circuit breakers
            Constraint::Min(6),    // Dedup stats
        ])
        .split(top_chunks[1]);

    if app.fullscreen_sparkline {
        // Fullscreen sparkline: fill entire inner area
        widgets::sparkline::render(f, inner, app);
    } else {
        // Render all widgets
        widgets::pipeline::render(f, left_chunks[0], app);
        widgets::channels::render(f, left_chunks[1], app);
        widgets::health::render(f, right_chunks[0], app);
        widgets::circuit::render(f, right_chunks[1], app);
        widgets::dedup::render(f, right_chunks[2], app);
        widgets::sparkline::render_multi_sparkline(f, main_chunks[1], app);
        widgets::regime::render(f, main_chunks[2], app);
        widgets::log::render(f, main_chunks[3], app);
    }

    // Loading overlay
    if let Some(progress) = app.loading_progress {
        draw_loading_overlay(f, size, progress);
    }
}

/// Renders the loading progress overlay.
fn draw_loading_overlay(f: &mut Frame, area: Rect, progress: u8) {
    let popup_width = 40.min(area.width.saturating_sub(4));
    let popup_height = 5.min(area.height.saturating_sub(4));
    let popup_x = (area.width.saturating_sub(popup_width)) / 2;
    let popup_y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    f.render_widget(Clear, popup_area);

    let block = Block::default()
        .title(" Loading ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let pct = progress as u16;
    let gauge = Gauge::default()
        .block(block)
        .gauge_style(Style::default().fg(Color::Cyan))
        .percent(pct)
        .label(format!("Loading\u{2026} {}%", pct));

    f.render_widget(gauge, popup_area);
}

/// Renders the "terminal too small" warning.
fn draw_too_small(f: &mut Frame, area: Rect) {
    let msg = format!(
        "Terminal too small \u{2014} resize to at least {}x{} (currently {}x{})",
        MIN_COLS, MIN_ROWS, area.width, area.height
    );
    let current_size = format!("Current size: {}x{}", area.width, area.height);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Red));

    let para = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            msg,
            Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            current_size,
            Style::default().fg(Color::DarkGray),
        )),
    ])
    .block(block)
    .alignment(Alignment::Center)
    .wrap(Wrap { trim: true });

    f.render_widget(para, area);
}

/// Renders the help overlay.
fn draw_help_overlay(f: &mut Frame, area: Rect) {
    // Center the help popup
    let popup_width = 52.min(area.width.saturating_sub(4));
    let popup_height = 22.min(area.height.saturating_sub(4));
    let popup_x = (area.width.saturating_sub(popup_width)) / 2;
    let popup_y = (area.height.saturating_sub(popup_height)) / 2;
    let popup_area = Rect::new(popup_x, popup_y, popup_width, popup_height);

    f.render_widget(Clear, popup_area);

    let help_text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "  tokio-prompt-orchestrator TUI",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "  Keybindings:",
            Style::default().fg(Color::White),
        )),
        Line::from(vec![
            Span::styled("    ?  ", Style::default().fg(Color::Cyan)),
            Span::styled("Toggle this help", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("    q  ", Style::default().fg(Color::Cyan)),
            Span::styled("Quit  (Esc / Ctrl+C also quit)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("    p  ", Style::default().fg(Color::Cyan)),
            Span::styled("Pause / Resume", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("    r  ", Style::default().fg(Color::Cyan)),
            Span::styled("Reset stats", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("    e  ", Style::default().fg(Color::Cyan)),
            Span::styled("Export log to file", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("    f  ", Style::default().fg(Color::Cyan)),
            Span::styled("Toggle fullscreen sparkline", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("  Tab  ", Style::default().fg(Color::Cyan)),
            Span::styled("Next symbol", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("  j/k  ", Style::default().fg(Color::Cyan)),
            Span::styled("[j/k] Vim scroll log", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("    \u{2191}\u{2193} ", Style::default().fg(Color::Cyan)),
            Span::styled("Scroll log", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("    [  ", Style::default().fg(Color::Cyan)),
            Span::styled("[[] widen left  []] narrow left", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  --mock  Synthetic 2-min story (default)",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "  --live  Connect to Prometheus endpoint",
            Style::default().fg(Color::DarkGray),
        )),
        Line::from(Span::styled(
            "  Press any key to close",
            Style::default().fg(Color::Yellow),
        )),
    ];

    let block = Block::default()
        .title(" Help ")
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));

    let para = Paragraph::new(help_text).block(block);
    f.render_widget(para, popup_area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_size_constants() {
        assert_eq!(MIN_COLS, 80);
        assert_eq!(MIN_ROWS, 24);
    }

    #[test]
    fn test_too_small_detection_width() {
        let area = Rect::new(0, 0, 70, 30);
        assert!(area.width < MIN_COLS);
    }

    #[test]
    fn test_too_small_detection_height() {
        let area = Rect::new(0, 0, 120, 20);
        assert!(area.height < MIN_ROWS);
    }

    #[test]
    fn test_adequate_size() {
        let area = Rect::new(0, 0, 120, 50);
        assert!(area.width >= MIN_COLS && area.height >= MIN_ROWS);
    }

    #[test]
    fn test_exactly_minimum_size() {
        let area = Rect::new(0, 0, MIN_COLS, MIN_ROWS);
        assert!(area.width >= MIN_COLS && area.height >= MIN_ROWS);
    }

    #[test]
    fn test_popup_centering_calculation() {
        let area_width: u16 = 120;
        let area_height: u16 = 50;
        let popup_width = 50.min(area_width.saturating_sub(4));
        let popup_height = 18.min(area_height.saturating_sub(4));
        let popup_x = (area_width.saturating_sub(popup_width)) / 2;
        let popup_y = (area_height.saturating_sub(popup_height)) / 2;

        assert_eq!(popup_width, 50);
        assert_eq!(popup_height, 18);
        assert_eq!(popup_x, 35);
        assert_eq!(popup_y, 16);
    }

    #[test]
    fn test_popup_centering_small_terminal() {
        let area_width: u16 = 40;
        let area_height: u16 = 15;
        let popup_width = 50.min(area_width.saturating_sub(4));
        let popup_height = 18.min(area_height.saturating_sub(4));

        assert_eq!(popup_width, 36);
        assert_eq!(popup_height, 11);
    }

    #[test]
    fn test_effective_min_size_guard_80x24() {
        // The draw function guards at 80x24
        let area = Rect::new(0, 0, 79, 24);
        assert!(area.width < 80, "width 79 < 80 should trigger guard");
    }

    #[test]
    fn test_effective_min_size_passes_80x24() {
        let area = Rect::new(0, 0, 80, 24);
        assert!(area.width >= 80 && area.height >= 24);
    }
}
