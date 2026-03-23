//! # Widget: Regime Classifier Panel
//!
//! ## Responsibility
//! Renders a dedicated panel showing real-time spacetime regime classification
//! for the current bar, rolling statistics over the last 100 bars, and a
//! Lorentz factor (gamma) sparkline.
//!
//! ## Layout
//! ```text
//! ┌─ REGIME CLASSIFIER ──────────────────────────────────────────────────────┐
//! │ Current:  [TIMELIKE]   β=0.412   γ=1.093   ds²=-3.21                   │
//! │ Last 100: T:62%  L:8%  S:30%                                            │
//! │ γ sparkline: ▁▂▃▄▄▃▅▆▇█▇▅▃▂▁▂▃▄▅▅▄▃▂▁▁▂▃▄▄▃▅▆▇█                      │
//! └──────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Color coding
//! - TIMELIKE  = Green  (ds² < 0: causal, subluminal movement)
//! - LIGHTLIKE = Yellow (ds² ≈ 0: at the speed of information)
//! - SPACELIKE = Red    (ds² > 0: acausal, stochastic movement)

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Sparkline as RatatuiSparkline};
use ratatui::Frame;

use crate::tui::app::{App, RegimeType};

/// Returns the display colour for a regime type.
pub fn regime_color(regime: RegimeType) -> Color {
    match regime {
        RegimeType::Timelike  => Color::Green,
        RegimeType::Lightlike => Color::Yellow,
        RegimeType::Spacelike => Color::Red,
        RegimeType::Unknown   => Color::DarkGray,
    }
}

/// Renders the full regime classifier panel including current classification,
/// rolling statistics, and the Lorentz factor gamma sparkline.
///
/// # Arguments
/// * `f`    - Ratatui frame to render into.
/// * `area` - Rectangular area allocated for this widget.
/// * `app`  - Application state containing regime and gamma data.
pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let total_window = app.regime_history.len().max(1);
    let t_pct = app.regime_stats[0] as f64 / total_window as f64 * 100.0;
    let l_pct = app.regime_stats[1] as f64 / total_window as f64 * 100.0;
    let s_pct = app.regime_stats[2] as f64 / total_window as f64 * 100.0;

    let title = format!(
        " REGIME CLASSIFIER  |  Last {}bars: T:{:.0}%  L:{:.0}%  S:{:.0}% ",
        total_window,
        t_pct,
        l_pct,
        s_pct,
    );

    let outer_block = Block::default()
        .title(Span::styled(
            title,
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner = outer_block.inner(area);
    f.render_widget(outer_block, area);

    // Split inner area into: top info row + gamma sparkline.
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2), // Current regime + stats row
            Constraint::Min(1),    // Gamma sparkline
        ])
        .split(inner);

    // ── Top row: current classification ──────────────────────────────────────
    let regime_color = regime_color(app.regime);
    let regime_label = app.regime.label();

    let current_line = Line::from(vec![
        Span::styled("Current: ", Style::default().fg(Color::White)),
        Span::styled(
            format!("[{:<9}]", regime_label),
            Style::default()
                .fg(regime_color)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!("\u{03b2}={:.4}", app.beta_display),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw("  "),
        Span::styled(
            format!("\u{03b3}={:.4}", app.gamma_display),
            Style::default().fg(Color::Magenta),
        ),
        Span::raw("  "),
        Span::styled(
            format!("ds\u{00b2}={:.4}", app.ds2_display),
            Style::default().fg(if app.ds2_display < 0.0 {
                Color::Green
            } else if app.ds2_display.abs() < 0.01 {
                Color::Yellow
            } else {
                Color::Red
            }),
        ),
        Span::raw("  "),
        // Compact rolling counts
        Span::styled(
            format!(
                "T:{:.0}%",
                t_pct
            ),
            Style::default().fg(Color::Green),
        ),
        Span::raw(" "),
        Span::styled(
            format!("L:{:.0}%", l_pct),
            Style::default().fg(Color::Yellow),
        ),
        Span::raw(" "),
        Span::styled(
            format!("S:{:.0}%", s_pct),
            Style::default().fg(Color::Red),
        ),
    ]);

    let info_para = Paragraph::new(current_line);
    f.render_widget(info_para, chunks[0]);

    // ── Gamma sparkline ────────────────────────────────────────────────────────
    // The sparkline uses u64 values (gamma * 1000) from app.gamma_history.
    let gamma_data: Vec<u64> = app.gamma_history.iter().copied().collect();

    let sparkline_color = regime_color;
    let gamma_title = format!(
        " \u{03b3} sparkline (last {} samples, max={:.3}) ",
        gamma_data.len(),
        app.gamma_history
            .iter()
            .copied()
            .max()
            .unwrap_or(1000) as f64
            / 1000.0,
    );
    let sparkline = RatatuiSparkline::default()
        .block(
            Block::default()
                .title(Span::styled(
                    gamma_title,
                    Style::default().fg(Color::DarkGray),
                ))
                .borders(Borders::NONE),
        )
        .data(&gamma_data)
        .style(Style::default().fg(sparkline_color));
    f.render_widget(sparkline, chunks[1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_color_timelike() {
        assert_eq!(regime_color(RegimeType::Timelike), Color::Green);
    }

    #[test]
    fn test_regime_color_lightlike() {
        assert_eq!(regime_color(RegimeType::Lightlike), Color::Yellow);
    }

    #[test]
    fn test_regime_color_spacelike() {
        assert_eq!(regime_color(RegimeType::Spacelike), Color::Red);
    }

    #[test]
    fn test_regime_color_unknown() {
        assert_eq!(regime_color(RegimeType::Unknown), Color::DarkGray);
    }
}
