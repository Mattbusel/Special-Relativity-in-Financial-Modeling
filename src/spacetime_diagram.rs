//! ASCII spacetime diagram renderer for Minkowski spacetime.
//!
//! Renders worldlines, light cones, and Lorentz-boosted reference frames
//! on a 2D ASCII grid with ct (vertical) and x (horizontal) axes.

/// A single worldline in Minkowski spacetime.
/// Events are stored as `(ct, x)` pairs.
#[derive(Debug, Clone)]
pub struct WorldLine {
    pub label: String,
    pub events: Vec<(f64, f64)>,
    pub line_char: char,
}

/// A 2D Minkowski spacetime diagram that can be rendered to ASCII.
#[derive(Debug, Clone)]
pub struct SpacetimeDiagram {
    pub width: usize,
    pub height: usize,
    /// (ct_min, ct_max)
    pub ct_range: (f64, f64),
    /// (x_min, x_max)
    pub x_range: (f64, f64),
    pub worldlines: Vec<WorldLine>,
}

impl SpacetimeDiagram {
    /// Create a new diagram with the given dimensions and axis ranges.
    pub fn new(
        width: usize,
        height: usize,
        ct_range: (f64, f64),
        x_range: (f64, f64),
    ) -> Self {
        Self {
            width,
            height,
            ct_range,
            x_range,
            worldlines: Vec::new(),
        }
    }

    /// Map a (ct, x) coordinate to a grid (row, col) pair.
    /// Row 0 is the top of the grid (high ct).
    fn to_grid(&self, ct: f64, x: f64) -> Option<(usize, usize)> {
        let (ct_min, ct_max) = self.ct_range;
        let (x_min, x_max) = self.x_range;

        if ct_max <= ct_min || x_max <= x_min {
            return None;
        }

        let ct_frac = (ct - ct_min) / (ct_max - ct_min);
        let x_frac = (x - x_min) / (x_max - x_min);

        // ct increases upward → row 0 = ct_max
        let row = ((1.0 - ct_frac) * (self.height as f64 - 1.0)).round() as isize;
        let col = (x_frac * (self.width as f64 - 1.0)).round() as isize;

        if row < 0 || row >= self.height as isize || col < 0 || col >= self.width as isize {
            return None;
        }

        Some((row as usize, col as usize))
    }

    /// Find the grid column corresponding to x = 0 (origin column).
    fn origin_col(&self) -> Option<usize> {
        let (x_min, x_max) = self.x_range;
        if x_max <= x_min {
            return None;
        }
        let frac = (0.0 - x_min) / (x_max - x_min);
        if !(0.0..=1.0).contains(&frac) {
            return None;
        }
        Some((frac * (self.width as f64 - 1.0)).round() as usize)
    }

    /// Find the grid row corresponding to ct = 0 (origin row).
    fn origin_row(&self) -> Option<usize> {
        let (ct_min, ct_max) = self.ct_range;
        if ct_max <= ct_min {
            return None;
        }
        let frac = (0.0 - ct_min) / (ct_max - ct_min);
        if !(0.0..=1.0).contains(&frac) {
            return None;
        }
        Some(((1.0 - frac) * (self.height as f64 - 1.0)).round() as usize)
    }

    /// Add a worldline to the diagram.
    pub fn add_worldline(&mut self, label: &str, events: Vec<(f64, f64)>, ch: char) {
        self.worldlines.push(WorldLine {
            label: label.to_string(),
            events,
            line_char: ch,
        });
    }

    /// Add a particle worldline computed from relativistic velocity β (v/c).
    /// The worldline spans the full ct range of the diagram.
    pub fn add_particle(&mut self, label: &str, start_ct: f64, start_x: f64, beta: f64) {
        // x(ct) = start_x + beta * (ct - start_ct)
        let (ct_min, ct_max) = self.ct_range;
        let events = vec![
            (ct_min, start_x + beta * (ct_min - start_ct)),
            (ct_max, start_x + beta * (ct_max - start_ct)),
        ];
        self.add_worldline(label, events, '*');
    }

    /// Add a photon worldline travelling at c (slope ±1 in Minkowski units).
    /// `direction`: +1 for right-moving, -1 for left-moving.
    pub fn add_photon(&mut self, label: &str, start_ct: f64, start_x: f64, direction: i8) {
        let slope = direction.signum() as f64; // ±1
        let (ct_min, ct_max) = self.ct_range;
        let events = vec![
            (ct_min, start_x + slope * (ct_min - start_ct)),
            (ct_max, start_x + slope * (ct_max - start_ct)),
        ];
        self.add_worldline(label, events, 'p');
    }

    /// Render the diagram to an ASCII string.
    ///
    /// - Vertical ct axis drawn with '|'
    /// - Horizontal x axis drawn with '-'
    /// - Origin at '+'
    /// - Light cones from origin: '/' (right-moving) and '\' (left-moving)
    /// - Lorentz-boosted simultaneity lines for β=0.5 drawn as '--'
    /// - Worldline events plotted with each line's `line_char`
    pub fn render_ascii(&self) -> String {
        // Allocate grid as a flat Vec<char>.
        let mut grid = vec![vec![' '; self.width]; self.height];

        // Helper: draw a rasterised line between two grid points.
        let draw_line = |grid: &mut Vec<Vec<char>>, r0: usize, c0: usize,
                         r1: usize, c1: usize, ch: char| {
            let steps = ((r1 as isize - r0 as isize).abs())
                .max((c1 as isize - c0 as isize).abs())
                + 1;
            for i in 0..=steps {
                let t = if steps == 0 { 0.0 } else { i as f64 / steps as f64 };
                let r = (r0 as f64 + t * (r1 as f64 - r0 as f64)).round() as usize;
                let c = (c0 as f64 + t * (c1 as f64 - c0 as f64)).round() as usize;
                if r < grid.len() && c < grid[0].len() {
                    grid[r][c] = ch;
                }
            }
        };

        // Draw horizontal x-axis.
        if let Some(orow) = self.origin_row() {
            for c in 0..self.width {
                grid[orow][c] = '-';
            }
        }

        // Draw vertical ct-axis.
        if let Some(ocol) = self.origin_col() {
            for r in 0..self.height {
                grid[r][ocol] = '|';
            }
        }

        // Draw origin marker.
        if let (Some(orow), Some(ocol)) = (self.origin_row(), self.origin_col()) {
            grid[orow][ocol] = '+';

            // Draw light cones from origin (slopes ±1 in ct-x space).
            // Right-moving photon: x = ct → slope +1 → '/'
            // Left-moving photon:  x = -ct → slope -1 → '\'
            let (ct_min, ct_max) = self.ct_range;
            let (x_min, x_max) = self.x_range;

            // Forward light cone (ct > 0).
            for r in 0..self.height {
                let ct_frac = 1.0 - r as f64 / (self.height as f64 - 1.0);
                let ct = ct_min + ct_frac * (ct_max - ct_min);
                if ct >= 0.0 {
                    // Right: x = ct
                    let x_r = ct;
                    if x_r >= x_min && x_r <= x_max {
                        let col = ((x_r - x_min) / (x_max - x_min) * (self.width as f64 - 1.0))
                            .round() as usize;
                        if col < self.width {
                            grid[r][col] = '/';
                        }
                    }
                    // Left: x = -ct
                    let x_l = -ct;
                    if x_l >= x_min && x_l <= x_max {
                        let col = ((x_l - x_min) / (x_max - x_min) * (self.width as f64 - 1.0))
                            .round() as usize;
                        if col < self.width {
                            grid[r][col] = '\\';
                        }
                    }
                }
            }

            // Lorentz-boosted simultaneity lines for β=0.5.
            // In boosted frame, lines of constant t' have slope β in (x, ct) space:
            // ct' = const → ct = β * x + const (for β=0.5, slope=0.5)
            let beta = 0.5_f64;
            // Draw a few simultaneity lines through the origin neighbourhood.
            let n_lines = 5usize;
            for i in 0..n_lines {
                let offset = (i as f64 - (n_lines as f64 - 1.0) / 2.0)
                    * (ct_max - ct_min) / (n_lines as f64);
                // ct = beta * x + offset
                // Sample along x.
                let steps = self.width * 4;
                let mut prev: Option<(usize, usize)> = None;
                for j in 0..=steps {
                    let x = x_min + (j as f64 / steps as f64) * (x_max - x_min);
                    let ct = beta * x + offset;
                    if let Some(gc) = self.to_grid(ct, x) {
                        if let Some(p) = prev {
                            // Only draw as '-' if not already occupied by axis.
                            if grid[gc.0][gc.1] == ' ' {
                                grid[gc.0][gc.1] = '-';
                            }
                            let _ = p;
                        }
                        prev = Some(gc);
                    }
                }
            }
        }

        // Draw worldlines.
        for wl in &self.worldlines {
            if wl.events.len() < 2 {
                // Just plot the event.
                for &(ct, x) in &wl.events {
                    if let Some((r, c)) = self.to_grid(ct, x) {
                        grid[r][c] = wl.line_char;
                    }
                }
                continue;
            }

            // Rasterise between successive event pairs.
            for pair in wl.events.windows(2) {
                let (ct0, x0) = pair[0];
                let (ct1, x1) = pair[1];

                let steps = self.width.max(self.height) * 4;
                for i in 0..=steps {
                    let t = i as f64 / steps as f64;
                    let ct = ct0 + t * (ct1 - ct0);
                    let x = x0 + t * (x1 - x0);
                    if let Some((r, c)) = self.to_grid(ct, x) {
                        grid[r][c] = wl.line_char;
                    }
                }
            }
        }

        // Serialise grid to string.
        grid.iter()
            .map(|row| row.iter().collect::<String>())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diagram() -> SpacetimeDiagram {
        SpacetimeDiagram::new(41, 21, (-5.0, 5.0), (-5.0, 5.0))
    }

    #[test]
    fn render_produces_correct_dimensions() {
        let diag = make_diagram();
        let rendered = diag.render_ascii();
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines.len(), 21, "height should be 21");
        for line in &lines {
            assert_eq!(line.chars().count(), 41, "width should be 41");
        }
    }

    #[test]
    fn particle_worldline_has_correct_slope() {
        let mut diag = make_diagram();
        // β=0 → vertical worldline (stationary particle)
        diag.add_particle("rest", 0.0, 0.0, 0.0);
        let wl = &diag.worldlines[0];
        // Both events should have x ≈ 0.
        for &(_ct, x) in &wl.events {
            assert!((x - 0.0).abs() < 1e-10, "stationary particle x should be 0");
        }
    }

    #[test]
    fn photon_at_45_degrees() {
        // A photon travels at c; slope = 1 in ct-x plane.
        let mut diag = make_diagram();
        diag.add_photon("photon", 0.0, 0.0, 1);
        let wl = &diag.worldlines[0];
        assert_eq!(wl.events.len(), 2);
        let (ct0, x0) = wl.events[0];
        let (ct1, x1) = wl.events[1];
        let slope = (ct1 - ct0) / (x1 - x0 + 1e-15);
        assert!((slope - 1.0).abs() < 1e-9, "photon slope should be 1");
    }

    #[test]
    fn worldline_with_multiple_events_renders() {
        let mut diag = make_diagram();
        diag.add_worldline("path", vec![(0.0, 0.0), (2.0, 1.0), (4.0, 0.5)], '@');
        let rendered = diag.render_ascii();
        assert!(rendered.contains('@'), "worldline character should appear in output");
    }
}
