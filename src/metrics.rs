//! Prometheus metrics helpers.

use prometheus::{Encoder, TextEncoder};

/// Gather all registered Prometheus metrics and return them as a UTF-8 string
/// in the standard Prometheus text exposition format.
pub fn gather_metrics() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder
        .encode(&metric_families, &mut buffer)
        .unwrap_or_default();
    String::from_utf8(buffer).unwrap_or_default()
}
