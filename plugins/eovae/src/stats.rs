#[derive(Clone, Debug, Default, PartialEq)]
pub struct ErrorStats {
    pub count: usize,
    pub mean: f64,
    pub p05: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ErrorHistogramBin {
    pub label: String,
    pub count: usize,
    pub normalized: f32,
}

pub fn summarize_errors(values: impl IntoIterator<Item = f64>) -> ErrorStats {
    let mut values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return ErrorStats::default();
    }

    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let count = values.len();
    let mean = values.iter().sum::<f64>() / count as f64;
    let p05 = percentile(&values, 0.05);
    let median = percentile(&values, 0.5);
    let p95 = percentile(&values, 0.95);
    let max = *values.last().unwrap_or(&0.0);

    ErrorStats {
        count,
        mean,
        p05,
        median,
        p95,
        max,
    }
}

pub fn build_error_histogram(
    values: impl IntoIterator<Item = f64>,
    bin_count: usize,
) -> Vec<ErrorHistogramBin> {
    let values = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() || bin_count == 0 {
        return Vec::new();
    }

    let max_value = values
        .iter()
        .copied()
        .fold(0.0_f64, |current, value| current.max(value));
    let step = if max_value <= f64::EPSILON {
        1.0
    } else {
        max_value / bin_count as f64
    };
    let mut counts = vec![0usize; bin_count];

    for value in values {
        let raw_index = if max_value <= f64::EPSILON {
            0
        } else {
            ((value / max_value) * bin_count as f64).floor() as usize
        };
        let index = raw_index.min(bin_count.saturating_sub(1));
        counts[index] += 1;
    }

    let max_count = counts.iter().copied().max().unwrap_or(0).max(1);
    counts
        .into_iter()
        .enumerate()
        .map(|(index, count)| {
            let start = if max_value <= f64::EPSILON {
                0.0
            } else {
                step * index as f64
            };
            let end = if max_value <= f64::EPSILON {
                1.0
            } else {
                step * (index + 1) as f64
            };
            ErrorHistogramBin {
                label: format!("{start:.2}-{end:.2}"),
                count,
                normalized: count as f32 / max_count as f32,
            }
        })
        .collect()
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
        assert!((summary.p05 - 0.1).abs() < 1e-6);
        assert!((summary.median - 0.3).abs() < 1e-6);
        assert!((summary.p95 - 0.9).abs() < 1e-6);
        assert!((summary.max - 0.9).abs() < 1e-6);
    }

    #[test]
    fn builds_histogram_bins_with_normalized_heights() {
        let bins = build_error_histogram([0.1, 0.2, 0.2, 0.7, 0.9], 4);
        assert_eq!(bins.len(), 4);
        assert_eq!(bins.iter().map(|bin| bin.count).sum::<usize>(), 5);
        assert!(bins.iter().any(|bin| (bin.normalized - 1.0).abs() < 1e-6));
        assert!(bins.iter().all(|bin| (0.0..=1.0).contains(&bin.normalized)));
    }
}
