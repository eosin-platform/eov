#[derive(Clone, Debug, Default, PartialEq)]
pub struct ErrorStats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

pub fn summarize_errors(values: impl IntoIterator<Item = f64>) -> ErrorStats {
    let mut values = values.into_iter().filter(|value| value.is_finite()).collect::<Vec<_>>();
    if values.is_empty() {
        return ErrorStats::default();
    }

    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let median = percentile(&values, 0.5);
    let p95 = percentile(&values, 0.95);
    let max = *values.last().unwrap_or(&0.0);

    ErrorStats {
        count,
        mean,
        median,
        p95,
        max,
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let last_index = sorted.len().saturating_sub(1);
    let index = (percentile.clamp(0.0, 1.0) * last_index as f64).round() as usize;
    sorted[index]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_mean_median_p95_and_max() {
        let summary = summarize_errors([0.1, 0.4, 0.3, 0.2, 0.9]);
        assert_eq!(summary.count, 5);
        assert!((summary.mean - 0.38).abs() < 1e-6);
        assert!((summary.median - 0.3).abs() < 1e-6);
        assert!((summary.p95 - 0.9).abs() < 1e-6);
        assert!((summary.max - 0.9).abs() < 1e-6);
    }
}