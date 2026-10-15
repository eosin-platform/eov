//! Small, allocation-free linear algebra for the stain model.
//!
//! Stain matrices have at most three columns (one optical-density vector per
//! stain), so everything here works on fixed-size arrays. Inversions and
//! eigen-decompositions are computed in `f64` and rounded to `f32` only at
//! the end, which keeps the derived per-pixel matrices accurate to well below
//! one 8-bit quantisation step.

pub(crate) type Vec3 = [f32; 3];

#[inline]
pub(crate) fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub(crate) fn norm(a: Vec3) -> f32 {
    dot(a, a).sqrt()
}

/// Scales `a` to unit length, or returns `None` for a (near-)zero vector.
pub(crate) fn normalize(a: Vec3) -> Option<Vec3> {
    let n = norm(a);
    if !n.is_finite() || n < 1e-6 {
        return None;
    }
    Some([a[0] / n, a[1] / n, a[2] / n])
}

/// Angle between two vectors in degrees (0 for parallel vectors).
pub(crate) fn angle_deg(a: Vec3, b: Vec3) -> f32 {
    let na = norm(a) as f64;
    let nb = norm(b) as f64;
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    let cos = (dot(a, b) as f64 / (na * nb)).clamp(-1.0, 1.0);
    cos.acos().to_degrees() as f32
}

/// Smallest pairwise angle between the given vectors, in degrees.
/// Returns 90° for fewer than two vectors.
pub(crate) fn min_pairwise_angle_deg(vectors: &[Vec3]) -> f32 {
    let mut best = 90.0f32;
    for i in 0..vectors.len() {
        for j in (i + 1)..vectors.len() {
            best = best.min(angle_deg(vectors[i], vectors[j]));
        }
    }
    best
}

fn gram(cols: &[Vec3]) -> [[f64; 3]; 3] {
    let mut g = [[0.0f64; 3]; 3];
    for i in 0..cols.len() {
        for j in 0..cols.len() {
            g[i][j] = (0..3).map(|k| cols[i][k] as f64 * cols[j][k] as f64).sum();
        }
    }
    g
}

/// Inverse of the leading `n`×`n` block of `m` (n = 1..=3) via the adjugate.
fn invert_small(m: &[[f64; 3]; 3], n: usize) -> Option<[[f64; 3]; 3]> {
    let mut inv = [[0.0f64; 3]; 3];
    match n {
        1 => {
            if m[0][0].abs() < 1e-12 {
                return None;
            }
            inv[0][0] = 1.0 / m[0][0];
        }
        2 => {
            let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
            if det.abs() < 1e-12 {
                return None;
            }
            inv[0][0] = m[1][1] / det;
            inv[0][1] = -m[0][1] / det;
            inv[1][0] = -m[1][0] / det;
            inv[1][1] = m[0][0] / det;
        }
        3 => {
            let c00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
            let c01 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
            let c02 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
            let det = m[0][0] * c00 + m[0][1] * c01 + m[0][2] * c02;
            if det.abs() < 1e-12 {
                return None;
            }
            inv[0][0] = c00 / det;
            inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
            inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
            inv[1][0] = c01 / det;
            inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
            inv[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
            inv[2][0] = c02 / det;
            inv[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
            inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
        }
        _ => return None,
    }
    Some(inv)
}

/// Moore–Penrose pseudo-inverse of the 3×n matrix whose columns are `cols`
/// (n = 1..=3): `(SᵀS)⁻¹ Sᵀ`.
///
/// Returns `n` meaningful rows (rows `n..3` are zero) such that
/// `c = P · od` is the least-squares concentration vector. For `n = 3` this is
/// the ordinary inverse. Returns `None` when the columns are linearly
/// dependent.
pub(crate) fn pseudo_inverse(cols: &[Vec3]) -> Option<[[f32; 3]; 3]> {
    let n = cols.len();
    if n == 0 || n > 3 {
        return None;
    }
    let g = gram(cols);
    let g_inv = invert_small(&g, n)?;
    let mut out = [[0.0f32; 3]; 3];
    for (i, row) in out.iter_mut().enumerate().take(n) {
        for (k, value) in row.iter_mut().enumerate() {
            *value = (0..n).map(|j| g_inv[i][j] * cols[j][k] as f64).sum::<f64>() as f32;
        }
    }
    Some(out)
}

/// Recovers the 3×n stain matrix (as columns) from its pseudo-inverse rows:
/// `S = Pᵀ (P Pᵀ)⁻¹`. Used to read legacy parameter blocks that only carry
/// the inverse.
pub(crate) fn matrix_from_pseudo_inverse(rows: &[Vec3]) -> Option<Vec<Vec3>> {
    let n = rows.len();
    if n == 0 || n > 3 {
        return None;
    }
    let g = gram(rows); // P Pᵀ
    let g_inv = invert_small(&g, n)?;
    let mut cols = vec![[0.0f32; 3]; n];
    for (j, col) in cols.iter_mut().enumerate() {
        for (k, value) in col.iter_mut().enumerate() {
            *value = (0..n).map(|i| rows[i][k] as f64 * g_inv[i][j]).sum::<f64>() as f32;
        }
    }
    Some(cols)
}

/// Eigen-decomposition of a symmetric 3×3 matrix by cyclic Jacobi rotations.
///
/// Returns the eigenvalues in descending order and the matching unit
/// eigenvectors (`vectors[i]` belongs to `values[i]`). Deterministic and
/// accurate to machine precision for the well-scaled covariance matrices used
/// here.
pub(crate) fn symmetric_eigen3(m: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..64 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        let scale = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if off <= 1e-30 * scale.max(1e-300) {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // A' = Jᵀ A J
            for row in a.iter_mut() {
                let akp = row[p];
                let akq = row[q];
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            let (rp, rq) = (a[p], a[q]);
            a[p] = std::array::from_fn(|k| c * rp[k] - s * rq[k]);
            a[q] = std::array::from_fn(|k| s * rp[k] + c * rq[k]);
            for row in v.iter_mut() {
                let vkp = row[p];
                let vkq = row[q];
                row[p] = c * vkp - s * vkq;
                row[q] = s * vkp + c * vkq;
            }
        }
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[j][j].total_cmp(&a[i][i]));
    let values = [
        a[order[0]][order[0]],
        a[order[1]][order[1]],
        a[order[2]][order[2]],
    ];
    let mut vectors = [[0.0f64; 3]; 3];
    for (dst, &src) in vectors.iter_mut().zip(order.iter()) {
        *dst = [v[0][src], v[1][src], v[2][src]];
    }
    (values, vectors)
}

/// 2-norm condition number of the 3×n matrix with the given columns.
///
/// For two unit vectors separated by an angle θ this equals `cot(θ/2)`.
/// Returns `f32::INFINITY` for rank-deficient input.
pub(crate) fn condition_number(cols: &[Vec3]) -> f32 {
    let n = cols.len();
    if n == 0 {
        return f32::INFINITY;
    }
    if n == 1 {
        return if norm(cols[0]) > 1e-9 {
            1.0
        } else {
            f32::INFINITY
        };
    }
    let g = gram(cols);
    let (max, min) = if n == 2 {
        let tr = 0.5 * (g[0][0] + g[1][1]);
        let diff = 0.5 * (g[0][0] - g[1][1]);
        let r = (diff * diff + g[0][1] * g[0][1]).sqrt();
        (tr + r, tr - r)
    } else {
        let (values, _) = symmetric_eigen3(g);
        (values[0], values[2])
    };
    if min <= 1e-15 * max.max(1e-300) {
        return f32::INFINITY;
    }
    (max / min).sqrt() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn pseudo_inverse_recovers_concentrations() {
        let h = normalize([0.65, 0.70, 0.29]).unwrap();
        let e = normalize([0.07, 0.99, 0.11]).unwrap();
        let p = pseudo_inverse(&[h, e]).unwrap();
        let od = [
            1.3 * h[0] + 0.4 * e[0],
            1.3 * h[1] + 0.4 * e[1],
            1.3 * h[2] + 0.4 * e[2],
        ];
        assert!(approx(dot(p[0], od), 1.3, 1e-5));
        assert!(approx(dot(p[1], od), 0.4, 1e-5));
        assert_eq!(p[2], [0.0; 3]);
    }

    #[test]
    fn three_column_pseudo_inverse_is_inverse() {
        let cols = [
            normalize([0.65, 0.70, 0.29]).unwrap(),
            normalize([0.07, 0.99, 0.11]).unwrap(),
            normalize([0.27, 0.57, 0.78]).unwrap(),
        ];
        let p = pseudo_inverse(&cols).unwrap();
        for (i, row) in p.iter().enumerate() {
            for (j, col) in cols.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(approx(dot(*row, *col), expected, 1e-5), "P·S[{i}][{j}]");
            }
        }
    }

    #[test]
    fn matrix_round_trips_through_pseudo_inverse() {
        let cols = vec![
            normalize([0.5626, 0.7201, 0.4062]).unwrap(),
            normalize([0.2159, 0.8012, 0.5581]).unwrap(),
        ];
        let p = pseudo_inverse(&cols).unwrap();
        let back = matrix_from_pseudo_inverse(&[p[0], p[1]]).unwrap();
        for (a, b) in cols.iter().zip(&back) {
            for k in 0..3 {
                assert!(approx(a[k], b[k], 1e-5), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn collinear_columns_have_no_inverse() {
        let a = [0.6, 0.7, 0.3];
        assert!(pseudo_inverse(&[a, a]).is_none());
        assert!(condition_number(&[a, a]).is_infinite());
    }

    #[test]
    fn condition_number_matches_angle_formula() {
        let theta = 30.0f32.to_radians();
        let a = [1.0, 0.0, 0.0];
        let b = [theta.cos(), theta.sin(), 0.0];
        let expected = 1.0 / (theta / 2.0).tan();
        assert!(approx(condition_number(&[a, b]), expected, 1e-3));
        assert!(approx(min_pairwise_angle_deg(&[a, b]), 30.0, 1e-3));
    }

    #[test]
    fn jacobi_eigen_decomposition_is_correct() {
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]];
        let (values, vectors) = symmetric_eigen3(m);
        assert!(values[0] >= values[1] && values[1] >= values[2]);
        for (value, vector) in values.iter().zip(vectors.iter()) {
            for row in 0..3 {
                let mv: f64 = (0..3).map(|k| m[row][k] * vector[k]).sum();
                assert!((mv - value * vector[row]).abs() < 1e-9);
            }
        }
    }
}
