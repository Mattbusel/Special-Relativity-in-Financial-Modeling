//! # Widget: Regime Timeline
//!
//! ## Responsibility
//! Renders the last 60 bars colour-coded by spacetime regime type as a
//! horizontal bar chart. Green=TIMELIKE, Yellow=LIGHTLIKE, Red=SPACELIKE,
//! DarkGray=UNKNOWN.

use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
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

/// Returns the single-character glyph for a regime bar.
pub fn regime_glyph(regime: RegimeType) -> &'static str {
    match regime {
        RegimeType::Timelike  => "▓",
        RegimeType::Lightlike => "▒",
        RegimeType::Spacelike => "░",
        RegimeType::Unknown   => "·",
    }
}

/// Renders the regime timeline widget.
pub fn render(f: &mut Frame, area: Rect, app: &App) {
    let timelike_count  = app.regime_history.iter().filter(|&&r| r == RegimeType::Timelike).count();
    let spacelike_count = app.regime_history.iter().filter(|&&r| r == RegimeType::Spacelike).count();
    let total = app.regime_history.len().max(1);
    let title = format!(
        " REGIME TIMELINE  T:{:.0}%  S:{:.0}% ",
        timelike_count  as f64 / total as f64 * 100.0,
        spacelike_count as f64 / total as f64 * 100.0,
    );

    let block = Block::default()
        .title(ratatui::text::Span::styled(
            title,
            Style::default().fg(Color::Cyan).add_modifier(ratatui::style::Modifier::BOLD),
        ))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    f.render_widget(block, area);

    // Build a single line of coloured glyphs, one per history sample.
    let spans: Vec<Span> = app
        .regime_history
        .iter()
        .map(|&r| Span::styled(regime_glyph(r), Style::default().fg(regime_color(r))))
        .collect();

    let para = Paragraph::new(Line::from(spans));
    f.render_widget(para, inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_color_timelike_green() {
        assert_eq!(regime_color(RegimeType::Timelike), Color::Green);
    }

    #[test]
    fn test_regime_color_spacelike_red() {
        assert_eq!(regime_color(RegimeType::Spacelike), Color::Red);
    }

    #[test]
    fn test_regime_color_lightlike_yellow() {
        assert_eq!(regime_color(RegimeType::Lightlike), Color::Yellow);
    }

    #[test]
    fn test_regime_color_unknown_gray() {
        assert_eq!(regime_color(RegimeType::Unknown), Color::DarkGray);
    }

    #[test]
    fn test_regime_glyph_nonempty() {
        assert!(!regime_glyph(RegimeType::Timelike).is_empty());
        assert!(!regime_glyph(RegimeType::Spacelike).is_empty());
        assert!(!regime_glyph(RegimeType::Lightlike).is_empty());
        assert!(!regime_glyph(RegimeType::Unknown).is_empty());
    }
}
