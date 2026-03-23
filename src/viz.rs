//! Interactive Spacetime Visualization (requires `viz` feature)
//!
//! ## Overview
//!
//! This module provides interactive egui-based visualizations for the SRFM
//! financial manifold:
//!
//! - [`SpacetimePlotter`] — 2D Minkowski diagram showing the price worldline,
//!   light cone boundaries, and per-event color coding by relativistic velocity.
//!
//! - [`PortfolioManifoldViewer`] — 3D scatter plot of multi-asset spacetime
//!   positions with click-to-inspect event details.
//!
//! ## Building
//!
//! ```text
//! cargo run --features viz -- --viz
//! ```
//!
//! ## Architecture
//!
//! The egui rendering loop is driven by [`eframe`]. Each viewer implements
//! [`eframe::App`] and manages its own state machine. Data is fed in via
//! shared buffers that can be updated from other threads (e.g. a live tick
//! stream).
//!
//! ## Spacetime Plotter Layout
//!
//! ```text
//!   ┌──────────────────────────────────────────────────────────┐
//!   │  SRFM Spacetime Plotter          [Zoom: 1.0x] [Reset]   │
//!   ├──────────────────┬───────────────────────────────────────┤
//!   │  Controls        │          Minkowski Diagram            │
//!   │  ─────────       │                                       │
//!   │  c_scale: 0.05   │   future light cone   ╱╲             │
//!   │  Show geodesic   │                       ╱  ╲            │
//!   │  Color by: β     │   worldline ●────────●────●          │
//!   │                  │                       ╲  ╱            │
//!   │  Inspect panel   │   past light cone      ╲╱             │
//!   │  ─────────       │                                       │
//!   │  t=42.0          │   ← log-price (x) →                  │
//!   │  P=105.2         │                                       │
//!   │  β=0.23          │                                       │
//!   │  γ=1.028         └───────────────────────────────────────┤
//!   │  ds²=-0.0023     │  Status: 1024 events | TIMELIKE 78%  │
//!   └──────────────────┴───────────────────────────────────────┘
//! ```

use std::collections::VecDeque;

/// One rendered event on the spacetime plotter.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpacetimeEvent {
    /// Coordinate time (x-axis of the plotter, treated as vertical in Minkowski).
    pub coord_time: f64,
    /// Log-price (x-axis of the Minkowski diagram).
    pub log_price: f64,
    /// Normalised velocity β = ΔP / (c·Δt).
    pub beta: f64,
    /// Lorentz factor γ.
    pub gamma: f64,
    /// Spacetime interval ds².
    pub ds2: f64,
    /// Human-readable label (e.g. ticker + timestamp).
    pub label: Option<String>,
}

impl SpacetimeEvent {
    /// Create a new event.
    pub fn new(coord_time: f64, log_price: f64, beta: f64) -> Self {
        let beta_safe = beta.clamp(-0.9999, 0.9999);
        let gamma = 1.0 / (1.0 - beta_safe * beta_safe).sqrt();
        let ds2 = -(1.0 - beta_safe * beta_safe); // simplified unit interval
        Self {
            coord_time,
            log_price,
            beta: beta_safe,
            gamma,
            ds2,
            label: None,
        }
    }

    /// Attach a human-readable label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Return the SRFM interval classification string.
    pub fn classify(&self) -> &'static str {
        const EPS: f64 = 1e-12;
        if self.ds2.abs() <= EPS {
            "LIGHTLIKE"
        } else if self.ds2 < 0.0 {
            "TIMELIKE"
        } else {
            "SPACELIKE"
        }
    }
}

/// Color encoding for a velocity value β ∈ [−1, 1].
///
/// Maps:
/// - β ≈ 0  → blue  (stationary / cold)
/// - β ≈ ±1 → red   (near-luminal / hot)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityColor {
    /// Red channel [0, 255].
    pub r: u8,
    /// Green channel [0, 255].
    pub g: u8,
    /// Blue channel [0, 255].
    pub b: u8,
}

impl VelocityColor {
    /// Compute the color for a given β.
    pub fn from_beta(beta: f64) -> Self {
        let t = beta.abs().clamp(0.0, 0.9999); // 0=slow, 1=near-luminal
        // Interpolate blue → cyan → yellow → red.
        let r = (255.0 * t.powi(2)) as u8;
        let g = (255.0 * (1.0 - (2.0 * t - 1.0).powi(2)).max(0.0)) as u8;
        let b = (255.0 * (1.0 - t)) as u8;
        Self { r, g, b }
    }
}

/// Configuration for the spacetime plotter.
#[derive(Debug, Clone)]
pub struct SpacetimePlotterConfig {
    /// Financial speed of light scale.
    pub c_scale: f64,
    /// Number of events to retain in the rolling buffer.
    pub max_events: usize,
    /// Whether to render the geodesic path (best-fit straight line).
    pub show_geodesic: bool,
    /// Initial zoom factor.
    pub initial_zoom: f64,
    /// Window title.
    pub title: String,
}

impl Default for SpacetimePlotterConfig {
    fn default() -> Self {
        Self {
            c_scale: 0.05,
            max_events: 2048,
            show_geodesic: true,
            initial_zoom: 1.0,
            title: "SRFM Spacetime Plotter".into(),
        }
    }
}

/// Interactive 2D Minkowski spacetime plotter.
///
/// Maintains a rolling buffer of [`SpacetimeEvent`]s and can render them
/// as an egui window when the `viz` feature is enabled.
///
/// The Minkowski diagram shows:
/// - **Horizontal axis**: log-price (spatial coordinate x).
/// - **Vertical axis**: coordinate time t.
/// - **Light cone lines**: drawn at slope ±c from the most recent event.
/// - **Worldline**: the sequence of events connected by line segments.
/// - **Color coding**: each event dot is colored by its velocity β.
/// - **Geodesic**: a best-fit constant-velocity path (straight line).
#[derive(Debug, Clone)]
pub struct SpacetimePlotter {
    cfg: SpacetimePlotterConfig,
    events: VecDeque<SpacetimeEvent>,
    /// Currently inspected event index (hover / click).
    inspected: Option<usize>,
    /// Current zoom factor.
    zoom: f64,
    /// Pan offset (x, y) in data coordinates.
    pan: (f64, f64),
    /// Statistics: fraction of TIMELIKE events in current buffer.
    timelike_fraction: f64,
}

impl SpacetimePlotter {
    /// Create a new empty plotter.
    pub fn new(cfg: SpacetimePlotterConfig) -> Self {
        Self {
            cfg,
            events: VecDeque::new(),
            inspected: None,
            zoom: 1.0,
            pan: (0.0, 0.0),
            timelike_fraction: 0.0,
        }
    }

    /// Push a new event into the rolling buffer.
    ///
    /// Old events are dropped when the buffer exceeds `max_events`.
    pub fn push_event(&mut self, event: SpacetimeEvent) {
        self.events.push_back(event);
        while self.events.len() > self.cfg.max_events {
            self.events.pop_front();
        }
        self.update_statistics();
    }

    /// Push a raw (coord_time, log_price, beta) triplet.
    pub fn push_raw(&mut self, coord_time: f64, log_price: f64, beta: f64) {
        self.push_event(SpacetimeEvent::new(coord_time, log_price, beta));
    }

    /// Number of events currently in the buffer.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Fraction of buffer events classified as TIMELIKE.
    pub fn timelike_fraction(&self) -> f64 {
        self.timelike_fraction
    }

    /// Return a snapshot of all events in chronological order.
    pub fn events(&self) -> impl Iterator<Item = &SpacetimeEvent> {
        self.events.iter()
    }

    /// Set the zoom factor (clamped to [0.1, 50]).
    pub fn set_zoom(&mut self, zoom: f64) {
        self.zoom = zoom.clamp(0.1, 50.0);
    }

    /// Reset zoom and pan to defaults.
    pub fn reset_view(&mut self) {
        self.zoom = self.cfg.cfg_zoom();
        self.pan = (0.0, 0.0);
    }

    /// Light-cone boundary slope in data coordinates: m = ±1/c.
    pub fn light_cone_slope(&self) -> f64 {
        if self.cfg.c_scale > 1e-12 { 1.0 / self.cfg.c_scale } else { 1e12 }
    }

    /// Compute the best-fit geodesic as (slope, intercept) in (t, x) space.
    ///
    /// Returns `None` if fewer than 2 events are available.
    pub fn geodesic_fit(&self) -> Option<(f64, f64)> {
        if self.events.len() < 2 {
            return None;
        }
        // Ordinary least-squares: x = m·t + b.
        let n = self.events.len() as f64;
        let sum_t: f64 = self.events.iter().map(|e| e.coord_time).sum();
        let sum_x: f64 = self.events.iter().map(|e| e.log_price).sum();
        let sum_tt: f64 = self.events.iter().map(|e| e.coord_time * e.coord_time).sum();
        let sum_tx: f64 = self.events.iter().map(|e| e.coord_time * e.log_price).sum();
        let denom = n * sum_tt - sum_t * sum_t;
        if denom.abs() < 1e-12 {
            return None;
        }
        let slope = (n * sum_tx - sum_t * sum_x) / denom;
        let intercept = (sum_x - slope * sum_t) / n;
        Some((slope, intercept))
    }

    /// Export all events as a JSON string for external rendering.
    pub fn export_json(&self) -> String {
        let events: Vec<&SpacetimeEvent> = self.events.iter().collect();
        serde_json::to_string(&events).unwrap_or_default()
    }

    // ── egui rendering ────────────────────────────────────────────────────────

    /// Render the plotter as an egui panel.
    ///
    /// Call this from within an `eframe::App::update` implementation.
    /// Requires the `viz` feature to be enabled.
    #[cfg(feature = "viz")]
    pub fn render(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("controls").show(ctx, |ui| {
            self.render_controls(ui);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_diagram(ui);
        });
    }

    #[cfg(feature = "viz")]
    fn render_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
        ui.separator();
        ui.label(format!("c_scale: {:.4}", self.cfg.c_scale));
        ui.label(format!("Events: {}", self.events.len()));
        ui.label(format!("TIMELIKE: {:.1}%", self.timelike_fraction * 100.0));
        ui.separator();
        ui.checkbox(&mut self.cfg.show_geodesic, "Show geodesic");
        ui.add(egui::Slider::new(&mut self.zoom, 0.1..=20.0).text("Zoom"));
        if ui.button("Reset View").clicked() {
            self.reset_view();
        }
        ui.separator();
        if let Some(idx) = self.inspected {
            if let Some(ev) = self.events.get(idx) {
                ui.strong("Inspect");
                ui.label(format!("t  = {:.4}", ev.coord_time));
                ui.label(format!("P  = {:.4}", ev.log_price));
                ui.label(format!("β  = {:.4}", ev.beta));
                ui.label(format!("γ  = {:.4}", ev.gamma));
                ui.label(format!("ds²= {:.4e}", ev.ds2));
                ui.label(format!("Class: {}", ev.classify()));
                if let Some(ref lbl) = ev.label {
                    ui.label(lbl.as_str());
                }
            }
        }
    }

    #[cfg(feature = "viz")]
    fn render_diagram(&mut self, ui: &mut egui::Ui) {
        use egui::{Color32, Stroke, pos2};

        let avail = ui.available_size();
        let (response, painter) = ui.allocate_painter(avail, egui::Sense::hover());
        let rect = response.rect;

        // Data bounds.
        let (t_min, t_max, x_min, x_max) = self.data_bounds();
        let t_range = (t_max - t_min).max(1.0);
        let x_range = (x_max - x_min).max(1.0);

        let to_screen = |t: f64, x: f64| -> egui::Pos2 {
            let nx = (x - x_min) / x_range;
            let ny = 1.0 - (t - t_min) / t_range; // flip y: high t at top
            pos2(
                rect.min.x + (nx * rect.width() as f64 * self.zoom) as f32,
                rect.min.y + (ny * rect.height() as f64 * self.zoom) as f32,
            )
        };

        // Draw background grid.
        painter.rect_filled(rect, 0.0, Color32::from_gray(20));

        // Draw light-cone lines from most recent event.
        if let Some(last) = self.events.back() {
            let slope = self.light_cone_slope();
            let dt = t_range;
            // Forward cone.
            let p0 = to_screen(last.coord_time, last.log_price);
            let pf1 = to_screen(last.coord_time + dt, last.log_price + dt / slope);
            let pf2 = to_screen(last.coord_time + dt, last.log_price - dt / slope);
            let pb1 = to_screen(last.coord_time - dt, last.log_price + dt / slope);
            let pb2 = to_screen(last.coord_time - dt, last.log_price - dt / slope);
            let cone_stroke = Stroke::new(1.0, Color32::from_rgba_unmultiplied(200, 200, 0, 120));
            painter.line_segment([p0, pf1], cone_stroke);
            painter.line_segment([p0, pf2], cone_stroke);
            painter.line_segment([p0, pb1], cone_stroke);
            painter.line_segment([p0, pb2], cone_stroke);
        }

        // Draw geodesic.
        if self.cfg.show_geodesic {
            if let Some((slope, intercept)) = self.geodesic_fit() {
                let geo_t0 = t_min;
                let geo_t1 = t_max;
                let gp0 = to_screen(geo_t0, slope * geo_t0 + intercept);
                let gp1 = to_screen(geo_t1, slope * geo_t1 + intercept);
                painter.line_segment(
                    [gp0, gp1],
                    Stroke::new(1.5, Color32::from_rgba_unmultiplied(100, 200, 255, 180)),
                );
            }
        }

        // Draw worldline segments and dots.
        let events: Vec<&SpacetimeEvent> = self.events.iter().collect();
        for i in 1..events.len() {
            let pa = to_screen(events[i - 1].coord_time, events[i - 1].log_price);
            let pb = to_screen(events[i].coord_time, events[i].log_price);
            let c = VelocityColor::from_beta(events[i].beta);
            let col = Color32::from_rgb(c.r, c.g, c.b);
            painter.line_segment([pa, pb], Stroke::new(1.2, col));
        }
        for (idx, ev) in events.iter().enumerate() {
            let p = to_screen(ev.coord_time, ev.log_price);
            let c = VelocityColor::from_beta(ev.beta);
            let col = Color32::from_rgb(c.r, c.g, c.b);
            let radius = if ev.classify() == "TIMELIKE" { 3.0 } else { 4.0 };
            painter.circle_filled(p, radius, col);

            // Hover detection.
            if let Some(cursor) = response.hover_pos() {
                let dist = ((p.x - cursor.x).powi(2) + (p.y - cursor.y).powi(2)).sqrt();
                if dist < 8.0 {
                    self.inspected = Some(idx);
                }
            }
        }

        // Status bar.
        painter.text(
            egui::pos2(rect.min.x + 4.0, rect.max.y - 16.0),
            egui::Align2::LEFT_BOTTOM,
            format!(
                "{} events | TIMELIKE {:.1}% | Zoom {:.1}x",
                self.events.len(),
                self.timelike_fraction * 100.0,
                self.zoom
            ),
            egui::FontId::monospace(11.0),
            Color32::from_gray(180),
        );
    }

    fn update_statistics(&mut self) {
        if self.events.is_empty() {
            self.timelike_fraction = 0.0;
            return;
        }
        let tl = self.events.iter().filter(|e| e.classify() == "TIMELIKE").count();
        self.timelike_fraction = tl as f64 / self.events.len() as f64;
    }

    fn data_bounds(&self) -> (f64, f64, f64, f64) {
        if self.events.is_empty() {
            return (0.0, 1.0, 0.0, 1.0);
        }
        let t_min = self.events.iter().map(|e| e.coord_time).fold(f64::INFINITY, f64::min);
        let t_max = self.events.iter().map(|e| e.coord_time).fold(f64::NEG_INFINITY, f64::max);
        let x_min = self.events.iter().map(|e| e.log_price).fold(f64::INFINITY, f64::min);
        let x_max = self.events.iter().map(|e| e.log_price).fold(f64::NEG_INFINITY, f64::max);
        (t_min, t_max, x_min, x_max)
    }
}

// Implement a helper so reset_view can access initial zoom without borrow issues.
impl SpacetimePlotterConfig {
    fn cfg_zoom(&self) -> f64 {
        self.initial_zoom
    }
}

// ── PortfolioManifoldViewer ───────────────────────────────────────────────────

/// A single asset's position in the portfolio manifold.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AssetPoint {
    /// Asset symbol / label.
    pub symbol: String,
    /// 3D projection of the spacetime position: (P_normalised, V_normalised, ds²).
    pub position: [f64; 3],
    /// Lorentz factor γ for this asset.
    pub gamma: f64,
    /// Interval class.
    pub class: String,
}

impl AssetPoint {
    /// Construct from raw SRFM fields.
    ///
    /// - `symbol`  — Asset label.
    /// - `price`   — Current price P.
    /// - `volume`  — Current volume V.
    /// - `ds2`     — Spacetime interval ds².
    /// - `beta`    — Normalised velocity β.
    pub fn new(symbol: impl Into<String>, price: f64, volume: f64, ds2: f64, beta: f64) -> Self {
        let beta_safe = beta.clamp(-0.9999, 0.9999);
        let gamma = 1.0 / (1.0 - beta_safe * beta_safe).sqrt();
        let class = if ds2.abs() <= 1e-12 {
            "LIGHTLIKE"
        } else if ds2 < 0.0 {
            "TIMELIKE"
        } else {
            "SPACELIKE"
        }
        .to_string();
        Self {
            symbol: symbol.into(),
            position: [price, volume.max(0.0).ln().max(0.0), ds2],
            gamma,
            class,
        }
    }
}

/// Configuration for the portfolio manifold viewer.
#[derive(Debug, Clone)]
pub struct ManifoldViewerConfig {
    /// Window title.
    pub title: String,
    /// Point radius (screen pixels).
    pub point_radius: f32,
    /// Whether to draw labels beside each point.
    pub show_labels: bool,
}

impl Default for ManifoldViewerConfig {
    fn default() -> Self {
        Self {
            title: "SRFM Portfolio Manifold".into(),
            point_radius: 6.0,
            show_labels: true,
        }
    }
}

/// 3D scatter-plot viewer for multi-asset spacetime positions.
///
/// Projects each asset's `(P, V, ds²)` position into a 2D view using
/// an azimuth/elevation camera model. The user can drag to rotate and
/// scroll to zoom. Clicking a dot shows the full event details.
///
/// ## Axis Mapping
///
/// | Screen axis | Data axis |
/// |-------------|-----------|
/// | Horizontal  | P (normalised price) |
/// | Vertical    | ds² (spacetime interval) |
/// | Depth       | V (log-volume) — encoded as point opacity |
#[derive(Debug, Clone)]
pub struct PortfolioManifoldViewer {
    cfg: ManifoldViewerConfig,
    points: Vec<AssetPoint>,
    /// Camera azimuth angle (radians).
    azimuth: f64,
    /// Camera elevation angle (radians).
    elevation: f64,
    /// Zoom factor.
    zoom: f64,
    /// Index of the currently selected point (click).
    selected: Option<usize>,
}

impl PortfolioManifoldViewer {
    /// Create a new empty viewer.
    pub fn new(cfg: ManifoldViewerConfig) -> Self {
        Self {
            cfg,
            points: Vec::new(),
            azimuth: 0.3,
            elevation: 0.4,
            zoom: 1.0,
            selected: None,
        }
    }

    /// Add or update a point in the manifold.
    ///
    /// If an asset with the same symbol already exists, its entry is replaced.
    pub fn upsert_point(&mut self, point: AssetPoint) {
        if let Some(existing) = self.points.iter_mut().find(|p| p.symbol == point.symbol) {
            *existing = point;
        } else {
            self.points.push(point);
        }
    }

    /// Remove all points.
    pub fn clear(&mut self) {
        self.points.clear();
        self.selected = None;
    }

    /// Number of assets currently displayed.
    pub fn asset_count(&self) -> usize {
        self.points.len()
    }

    /// Selected point, if any.
    pub fn selected_point(&self) -> Option<&AssetPoint> {
        self.selected.and_then(|i| self.points.get(i))
    }

    /// Project a 3D point to 2D using azimuth/elevation rotation.
    fn project(&self, pos: [f64; 3]) -> (f64, f64) {
        let az = self.azimuth;
        let el = self.elevation;
        // Rotate around Y axis by azimuth, then X axis by elevation.
        let x = pos[0];
        let y = pos[1];
        let z = pos[2];
        let x1 = x * az.cos() - z * az.sin();
        let z1 = x * az.sin() + z * az.cos();
        let y2 = y * el.cos() - z1 * el.sin();
        (x1 * self.zoom, y2 * self.zoom)
    }

    /// Export all points as a JSON string.
    pub fn export_json(&self) -> String {
        serde_json::to_string(&self.points).unwrap_or_default()
    }

    /// Render the viewer as an egui panel.
    #[cfg(feature = "viz")]
    pub fn render(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("manifold_inspect").show(ctx, |ui| {
            self.render_inspect_panel(ui);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_scatter(ui);
        });
    }

    #[cfg(feature = "viz")]
    fn render_inspect_panel(&self, ui: &mut egui::Ui) {
        ui.heading("Asset Details");
        ui.separator();
        if let Some(pt) = self.selected_point() {
            ui.strong(&pt.symbol);
            ui.label(format!("Price (norm): {:.4}", pt.position[0]));
            ui.label(format!("Log-vol:      {:.4}", pt.position[1]));
            ui.label(format!("ds²:          {:.4e}", pt.position[2]));
            ui.label(format!("γ:            {:.4}", pt.gamma));
            ui.label(format!("Class:        {}", pt.class));
        } else {
            ui.label("Click a dot to inspect.");
        }
        ui.separator();
        ui.label(format!("Assets: {}", self.points.len()));
        let tl = self.points.iter().filter(|p| p.class == "TIMELIKE").count();
        let sl = self.points.iter().filter(|p| p.class == "SPACELIKE").count();
        ui.label(format!("TIMELIKE:  {}", tl));
        ui.label(format!("SPACELIKE: {}", sl));
    }

    #[cfg(feature = "viz")]
    fn render_scatter(&mut self, ui: &mut egui::Ui) {
        use egui::{Color32, Stroke, pos2};

        let avail = ui.available_size();
        let (response, painter) = ui.allocate_painter(avail, egui::Sense::click_and_drag());
        let rect = response.rect;
        let cx = rect.center().x as f64;
        let cy = rect.center().y as f64;

        // Handle drag → rotate camera.
        if response.dragged() {
            let delta = response.drag_delta();
            self.azimuth += delta.x as f64 * 0.01;
            self.elevation += delta.y as f64 * 0.01;
        }

        // Handle scroll → zoom.
        let scroll = ui.input(|i| i.scroll_delta.y);
        self.zoom *= 1.0 + scroll as f64 * 0.001;
        self.zoom = self.zoom.clamp(0.1, 20.0);

        painter.rect_filled(rect, 0.0, Color32::from_gray(18));

        // Draw axes.
        let axis_col = Color32::from_gray(80);
        painter.line_segment(
            [pos2(rect.min.x, cy as f32), pos2(rect.max.x, cy as f32)],
            Stroke::new(0.5, axis_col),
        );
        painter.line_segment(
            [pos2(cx as f32, rect.min.y), pos2(cx as f32, rect.max.y)],
            Stroke::new(0.5, axis_col),
        );

        // Normalize positions for rendering.
        let scale = 150.0 * self.zoom;
        let mut closest_dist = f32::INFINITY;
        let mut closest_idx = None;

        for (idx, pt) in self.points.iter().enumerate() {
            let (sx, sy) = self.project(pt.position);
            let screen = pos2((cx + sx * scale) as f32, (cy - sy * scale) as f32);

            // Color by class.
            let col = match pt.class.as_str() {
                "TIMELIKE" => Color32::from_rgb(50, 200, 80),
                "SPACELIKE" => Color32::from_rgb(220, 60, 60),
                _ => Color32::from_rgb(220, 220, 50), // LIGHTLIKE
            };

            painter.circle_filled(screen, self.cfg.point_radius, col);
            painter.circle_stroke(screen, self.cfg.point_radius, Stroke::new(1.0, Color32::WHITE));

            if self.cfg.show_labels {
                painter.text(
                    egui::pos2(screen.x + self.cfg.point_radius + 2.0, screen.y),
                    egui::Align2::LEFT_CENTER,
                    &pt.symbol,
                    egui::FontId::monospace(10.0),
                    Color32::from_gray(200),
                );
            }

            // Track closest to cursor for click detection.
            if let Some(cursor) = response.interact_pointer_pos() {
                let dist = ((screen.x - cursor.x).powi(2) + (screen.y - cursor.y).powi(2)).sqrt();
                if dist < closest_dist {
                    closest_dist = dist;
                    closest_idx = Some(idx);
                }
            }
        }

        // Select on click.
        if response.clicked() {
            if closest_dist < 16.0 {
                self.selected = closest_idx;
            }
        }
    }
}

// ── Tests (non-viz) ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spacetime_event_classify_timelike() {
        let e = SpacetimeEvent::new(0.0, 10.0, 0.1);
        // ds2 = -(1 - β²) < 0 → TIMELIKE.
        assert_eq!(e.classify(), "TIMELIKE");
    }

    #[test]
    fn spacetime_event_classify_lightlike() {
        let mut e = SpacetimeEvent::new(0.0, 10.0, 0.0);
        e.ds2 = 0.0;
        assert_eq!(e.classify(), "LIGHTLIKE");
    }

    #[test]
    fn spacetime_event_classify_spacelike() {
        let mut e = SpacetimeEvent::new(0.0, 10.0, 0.5);
        e.ds2 = 1.0; // force positive
        assert_eq!(e.classify(), "SPACELIKE");
    }

    #[test]
    fn plotter_rolling_buffer_cap() {
        let cfg = SpacetimePlotterConfig {
            max_events: 5,
            ..SpacetimePlotterConfig::default()
        };
        let mut plotter = SpacetimePlotter::new(cfg);
        for i in 0..10 {
            plotter.push_raw(i as f64, (i as f64).ln().max(0.0), 0.1);
        }
        assert_eq!(plotter.event_count(), 5, "buffer should cap at max_events=5");
    }

    #[test]
    fn plotter_timelike_fraction_all_slow() {
        let cfg = SpacetimePlotterConfig::default();
        let mut plotter = SpacetimePlotter::new(cfg);
        for i in 0..20 {
            plotter.push_raw(i as f64, i as f64, 0.01); // β=0.01 → TIMELIKE
        }
        assert!(
            plotter.timelike_fraction() > 0.9,
            "all slow events should be mostly TIMELIKE"
        );
    }

    #[test]
    fn plotter_geodesic_fit_linear_data() {
        let cfg = SpacetimePlotterConfig::default();
        let mut plotter = SpacetimePlotter::new(cfg);
        // Perfect linear trajectory: x = 2t + 1.
        for i in 0..10 {
            let t = i as f64;
            plotter.push_raw(t, 2.0 * t + 1.0, 0.05);
        }
        let (slope, intercept) = plotter.geodesic_fit().expect("should have fit");
        assert!((slope - 2.0).abs() < 1e-6, "slope should be 2, got {}", slope);
        assert!((intercept - 1.0).abs() < 1e-6, "intercept should be 1, got {}", intercept);
    }

    #[test]
    fn plotter_geodesic_fit_none_for_single_event() {
        let cfg = SpacetimePlotterConfig::default();
        let mut plotter = SpacetimePlotter::new(cfg);
        plotter.push_raw(0.0, 10.0, 0.05);
        assert!(plotter.geodesic_fit().is_none(), "need ≥2 events for fit");
    }

    #[test]
    fn plotter_export_json_is_valid() {
        let cfg = SpacetimePlotterConfig::default();
        let mut plotter = SpacetimePlotter::new(cfg);
        plotter.push_raw(0.0, 10.0, 0.1);
        let json = plotter.export_json();
        assert!(json.starts_with('['), "export should be a JSON array");
    }

    #[test]
    fn velocity_color_slow_is_blue() {
        let c = VelocityColor::from_beta(0.0);
        assert!(c.b > c.r, "slow (β=0) should be more blue than red");
    }

    #[test]
    fn velocity_color_fast_is_red() {
        let c = VelocityColor::from_beta(0.99);
        assert!(c.r > c.b, "fast (β=0.99) should be more red than blue");
    }

    #[test]
    fn portfolio_viewer_upsert_replaces_symbol() {
        let cfg = ManifoldViewerConfig::default();
        let mut viewer = PortfolioManifoldViewer::new(cfg);
        viewer.upsert_point(AssetPoint::new("BTC", 50000.0, 1e9, -0.5, 0.2));
        viewer.upsert_point(AssetPoint::new("BTC", 60000.0, 1e9, -0.5, 0.3));
        assert_eq!(viewer.asset_count(), 1, "upsert should replace existing symbol");
    }

    #[test]
    fn portfolio_viewer_export_json() {
        let cfg = ManifoldViewerConfig::default();
        let mut viewer = PortfolioManifoldViewer::new(cfg);
        viewer.upsert_point(AssetPoint::new("ETH", 3000.0, 5e8, -0.1, 0.1));
        let json = viewer.export_json();
        assert!(json.contains("ETH"));
    }

    #[test]
    fn asset_point_timelike_for_negative_ds2() {
        let pt = AssetPoint::new("SPY", 400.0, 1e8, -0.5, 0.2);
        assert_eq!(pt.class, "TIMELIKE");
    }

    #[test]
    fn asset_point_spacelike_for_positive_ds2() {
        let pt = AssetPoint::new("SPY", 400.0, 1e8, 0.5, 0.2);
        assert_eq!(pt.class, "SPACELIKE");
    }
}
