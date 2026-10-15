//! Deterministic candidate orders for per-slide tile limits.
//!
//! The generator is SplitMix64, implemented here (rather than taken from a
//! crate) so that a given seed selects the same patches in every eov
//! version.

/// SplitMix64 pseudo-random generator.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform integer in `0..bound` (Lemire's multiply-shift, unbiased
    /// enough for sampling).
    pub fn below(&mut self, bound: u64) -> u64 {
        ((self.next_u64() as u128 * bound as u128) >> 64) as u64
    }
}

/// A per-slide seed derived from the run seed and the slide key, so that
/// adding or removing other slides does not change a slide's sample.
pub fn slide_seed(seed: u64, slide_key: &str) -> u64 {
    // FNV-1a of the key, mixed with the seed.
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in slide_key.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    SplitMix64::new(seed ^ hash).next_u64()
}

/// Shuffles `items` in place (Fisher–Yates).
pub fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for i in (1..items.len()).rev() {
        let j = rng.below(i as u64 + 1) as usize;
        items.swap(i, j);
    }
}

/// Reorders `items` so that every prefix is spread evenly over the
/// original order (bit-reversal permutation): taking the first k items
/// samples the slide uniformly without randomness.
pub fn spread_order<T: Copy>(items: &[T]) -> Vec<T> {
    let n = items.len();
    if n <= 2 {
        return items.to_vec();
    }
    let bits = usize::BITS - (n - 1).leading_zeros();
    let mut out = Vec::with_capacity(n);
    for i in 0..(1usize << bits) {
        let j = i.reverse_bits() >> (usize::BITS - bits);
        if j < n {
            out.push(items[j]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_reference_values() {
        // Reference sequence for seed 0 (Vigna's splitmix64.c).
        let mut rng = SplitMix64::new(0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn shuffles_are_deterministic_permutations() {
        let mut a: Vec<u32> = (0..1000).collect();
        let mut b = a.clone();
        shuffle(&mut a, 42);
        shuffle(&mut b, 42);
        assert_eq!(a, b);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
        let mut c: Vec<u32> = (0..1000).collect();
        shuffle(&mut c, 43);
        assert_ne!(a, c);
        assert_ne!(slide_seed(1, "a"), slide_seed(1, "b"));
    }

    #[test]
    fn spread_order_is_a_permutation_with_even_prefixes() {
        let items: Vec<u32> = (0..10).collect();
        let order = spread_order(&items);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, items);
        assert_eq!(&order[..4], &[0, 8, 4, 2]);
    }
}
