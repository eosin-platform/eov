//! Error statistics: summaries with interpolated percentiles, histograms
//! with readable labels, and the highest-error tiles (VAE-17).

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A summary of tile errors.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Summary {
    pub count: usize,
    pub mean: f64,
    pub min: f64,
    pub p05: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

/// The `q`-quantile of sorted values, linearly interpolated between closest
/// ranks (the default of R, NumPy and Excel).
pub fn percentile(sorted: &[f32], q: f64) -> f64 {
    match sorted.len() {
        0 => 0.0,
        1 => f64::from(sorted[0]),
        n => {
            let position = q.clamp(0.0, 1.0) * (n - 1) as f64;
            let lower = position.floor() as usize;
            let upper = (lower + 1).min(n - 1);
            let fraction = position - lower as f64;
            f64::from(sorted[lower])
                + (f64::from(sorted[upper]) - f64::from(sorted[lower])) * fraction
        }
    }
}

/// Summarises values; sorts them in place. Non-finite values are dropped.
pub fn summarize(values: &mut Vec<f32>) -> Option<Summary> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let sum: f64 = values.iter().map(|v| f64::from(*v)).sum();
    Some(Summary {
        count: values.len(),
        mean: sum / values.len() as f64,
        min: f64::from(values[0]),
        p05: percentile(values, 0.05),
        median: percentile(values, 0.5),
        p95: percentile(values, 0.95),
        max: f64::from(*values.last().expect("non-empty")),
    })
}

/// One histogram bin, `[lo, hi)` (the last bin includes `hi`).
#[derive(Clone, Debug, PartialEq)]
pub struct Bin {
    pub lo: f64,
    pub hi: f64,
    pub count: usize,
}

/// A histogram of sorted values over `[min, max]`.
pub fn histogram(sorted: &[f32], bins: usize) -> Vec<Bin> {
    let (Some(first), Some(last)) = (sorted.first(), sorted.last()) else {
        return Vec::new();
    };
    if bins == 0 {
        return Vec::new();
    }
    let lo = f64::from(*first);
    let mut hi = f64::from(*last);
    if hi - lo <= f64::EPSILON * lo.abs().max(1.0) {
        hi = lo + lo.abs().max(1e-6) * 0.01;
    }
    let width = (hi - lo) / bins as f64;
    let mut counts = vec![0usize; bins];
    for value in sorted {
        let index = ((f64::from(*value) - lo) / width).floor() as isize;
        counts[index.clamp(0, bins as isize - 1) as usize] += 1;
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, count)| Bin {
            lo: lo + width * i as f64,
            hi: lo + width * (i + 1) as f64,
            count,
        })
        .collect()
}

/// Decimals needed to tell values `step` apart (2–6).
pub fn decimals_for(step: f64) -> usize {
    if !(step.is_finite() && step > 0.0) {
        return 4;
    }
    ((-step.log10()).ceil() as i64 + 1).clamp(2, 6) as usize
}

/// Formats an error value with `decimals` decimals.
pub fn format_error(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}")
}

#[derive(PartialEq)]
struct Ranked<T> {
    score: f32,
    item: T,
}

impl<T: PartialEq> Eq for Ranked<T> {}

impl<T: PartialEq> PartialOrd for Ranked<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: PartialEq> Ord for Ranked<T> {
    // Reversed: the heap keeps the smallest of the current top scores on top.
    fn cmp(&self, other: &Self) -> Ordering {
        other.score.total_cmp(&self.score)
    }
}

/// The `k` highest-scoring items, highest first, in O(n log k).
pub fn top_k<T: PartialEq>(items: impl IntoIterator<Item = (f32, T)>, k: usize) -> Vec<(f32, T)> {
    if k == 0 {
        return Vec::new();
    }
    let mut heap: BinaryHeap<Ranked<T>> = BinaryHeap::with_capacity(k + 1);
    for (score, item) in items {
        if !score.is_finite() {
            continue;
        }
        if heap.len() < k {
            heap.push(Ranked { score, item });
        } else if heap.peek().is_some_and(|smallest| score > smallest.score) {
            heap.pop();
            heap.push(Ranked { score, item });
        }
    }
    let mut ranked: Vec<(f32, T)> = heap.into_iter().map(|r| (r.score, r.item)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    ranked
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_interpolate() {
        let sorted = [1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&sorted, 0.5), 2.5);
        assert!((percentile(&sorted, 0.95) - 3.85).abs() < 1e-9);
        assert_eq!(percentile(&sorted, 0.0), 1.0);
        assert_eq!(percentile(&sorted, 1.0), 4.0);
        assert_eq!(percentile(&[], 0.5), 0.0);
    }

    #[test]
    fn summaries_ignore_non_finite_values() {
        let mut values = vec![0.4, 0.1, f32::NAN, 0.3, 0.2, 0.9];
        let summary = summarize(&mut values).unwrap();
        assert_eq!(summary.count, 5);
        assert!((summary.mean - 0.38).abs() < 1e-6);
        assert!((summary.median - 0.3).abs() < 1e-6);
        assert!((summary.p95 - 0.8).abs() < 1e-6);
        assert!(summarize(&mut vec![]).is_none());
    }

    #[test]
    fn histograms_cover_the_range() {
        let values = [0.0051f32, 0.0052, 0.0055, 0.0060, 0.0090];
        let bins = histogram(&values, 4);
        assert_eq!(bins.len(), 4);
        assert_eq!(bins.iter().map(|b| b.count).sum::<usize>(), 5);
        assert_eq!(bins[3].count, 1);
        // Labels need enough decimals: typical errors are ~0.005.
        let decimals = decimals_for(bins[0].hi - bins[0].lo);
        assert!(decimals >= 4, "{decimals}");
        assert_ne!(
            format_error(bins[0].lo, decimals),
            format_error(bins[1].lo, decimals)
        );
        // Constant values still produce a bin.
        assert_eq!(histogram(&[0.5, 0.5], 3)[0].count, 2);
    }

    #[test]
    fn top_k_keeps_the_largest() {
        let items = [
            (0.1, 'a'),
            (0.9, 'b'),
            (0.5, 'c'),
            (f32::NAN, 'd'),
            (0.7, 'e'),
        ];
        let top = top_k(items, 2);
        assert_eq!(top, vec![(0.9, 'b'), (0.7, 'e')]);
        assert!(top_k(items, 0).is_empty());
    }
}
