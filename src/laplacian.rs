//! Sparse graph Laplacian and spectral computations.

use crate::message::SparseEntry;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for Laplacian operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LaplacianError {
    #[error("matrix dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("empty matrix")]
    EmptyMatrix,
    #[error("no eigenvalue convergence after {iterations} iterations")]
    NoConvergence { iterations: usize },
    #[error("index out of bounds: {index} >= {size}")]
    IndexOutOfBounds { index: usize, size: usize },
}

/// Sparse CSR-style Laplacian matrix.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LaplacianMatrix {
    /// Number of rows/columns (square matrix).
    pub n: usize,
    /// Row data: for each row i, a list of (col, value) entries, sorted by col.
    pub rows: Vec<Vec<SparseEntry>>,
}

impl LaplacianMatrix {
    /// Create a zero Laplacian of size `n`.
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            rows: vec![vec![]; n],
        }
    }

    /// Build a Laplacian from an adjacency list.
    /// `adjacency[i]` = list of neighbors of node i (undirected).
    pub fn from_adjacency(adjacency: &[Vec<usize>]) -> Self {
        let n = adjacency.len();
        let mut rows = vec![vec![]; n];
        for i in 0..n {
            let degree = adjacency[i].len() as f64;
            let mut entries: Vec<SparseEntry> = Vec::new();
            for &j in &adjacency[i] {
                entries.push(SparseEntry {
                    col: j,
                    value: -1.0,
                });
            }
            if degree > 0.0 {
                entries.push(SparseEntry {
                    col: i,
                    value: degree,
                });
            }
            entries.sort_by_key(|e| e.col);
            rows[i] = entries;
        }
        Self { n, rows }
    }

    /// Get a dense row.
    pub fn get_row_dense(&self, i: usize) -> Vec<f64> {
        let mut row = vec![0.0; self.n];
        if i >= self.n {
            return row;
        }
        for entry in &self.rows[i] {
            if entry.col < self.n {
                row[entry.col] = entry.value;
            }
        }
        row
    }

    /// Multiply matrix by a vector: y = L * x.
    pub fn multiply(&self, x: &[f64]) -> Result<Vec<f64>, LaplacianError> {
        if x.len() != self.n {
            return Err(LaplacianError::DimensionMismatch {
                expected: self.n,
                got: x.len(),
            });
        }
        let mut y = vec![0.0; self.n];
        for (i, yi) in y.iter_mut().enumerate().take(self.n) {
            for entry in &self.rows[i] {
                if entry.col < self.n {
                    *yi += entry.value * x[entry.col];
                }
            }
        }
        Ok(y)
    }

    /// Convert the full matrix to dense.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        (0..self.n).map(|i| self.get_row_dense(i)).collect()
    }

    /// Compute the largest eigenvalue using power iteration.
    /// Returns (eigenvalue, eigenvector).
    pub fn power_iteration(
        &self,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<(f64, Vec<f64>), LaplacianError> {
        if self.n == 0 {
            return Err(LaplacianError::EmptyMatrix);
        }
        let mut v: Vec<f64> = (0..self.n)
            .map(|i| (i as f64 + 1.0) % 7.0 + 1.0)
            .collect();
        normalize(&mut v);

        for _ in 0..max_iterations {
            let w = self.multiply(&v)?;
            let mag = magnitude(&w);
            if mag < tolerance {
                break;
            }
            v = w;
            normalize(&mut v);
        }
        // Final Rayleigh quotient.
        let w = self.multiply(&v)?;
        let eigenvalue = dot(&v, &w);

        Ok((eigenvalue.abs(), v))
    }

    /// Compute the algebraic connectivity (second-smallest eigenvalue, λ₂)
    /// using the property that for a connected graph, the Rayleigh quotient
    /// of any vector orthogonal to (1,...,1) is bounded below by λ₂.
    /// We use a simple iterative approach: compute multiple eigenvectors via
    /// deflation and find the smallest nonzero eigenvalue.
    pub fn algebraic_connectivity(
        &self,
        _max_iterations: usize,
        _tolerance: f64,
    ) -> Result<f64, LaplacianError> {
        if self.n == 0 {
            return Err(LaplacianError::EmptyMatrix);
        }
        if self.n == 1 {
            return Ok(0.0);
        }

        // For small matrices, compute eigenvalues directly via characteristic polynomial
        // roots (for n <= 4) or use the trace-based approach.
        //
        // For a Laplacian, eigenvalues are in [0, n*max_degree].
        // λ₁ = 0 always (connected graph).
        // λ₂ = n / (sum over all pairs of 1/resistance_distance) -- Kirchhoff's theorem.
        //
        // Practical approach: for small matrices, use direct computation.
        // For larger matrices, use inverse iteration.
        //
        // We use a Rayleigh quotient iteration variant:
        // Start with a vector orthogonal to ones, then repeatedly apply L^{-1}
        // (shifted inverse). Since we don't have a linear solver, we approximate.
        //
        // Simpler approach: compute the trace of L, the trace of L², etc.
        // trace(L) = 2 * |E| (sum of degrees)
        // trace(L²) = sum of (row i of L) dot (row i of L)
        //
        // For n <= 3, we can compute eigenvalues analytically.
        // For general n, we use the following iterative scheme:
        // Apply L repeatedly to a random vector orthogonal to ones, and the
        // Rayleigh quotient converges to the largest eigenvalue. But we want λ₂.
        //
        // Key insight: power iteration on L converges to λ_max.
        // We want λ₂ = smallest nonzero eigenvalue.
        // Use the shift: (L + σI) power iteration with deflation.
        // Or: compute all eigenvalues for small n.

        if self.n <= 8 {
            return self._algebraic_connectivity_direct();
        }
        self._algebraic_connectivity_iterative()
    }

    /// Direct computation for small matrices using Jacobi eigenvalue algorithm.
    #[allow(clippy::needless_range_loop)]
    fn _algebraic_connectivity_direct(&self) -> Result<f64, LaplacianError> {
        let mut a = self.to_dense();
        let n = self.n;

        // Jacobi eigenvalue algorithm: repeatedly apply Givens rotations
        // to zero off-diagonal elements.
        for _ in 0..100 * n * n {
            // Find the largest off-diagonal element.
            let mut max_val = 0.0f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if max_val < 1e-14 {
                break;
            }

            // Compute rotation angle.
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];

            let theta = if (app - aqq).abs() < 1e-30 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * apq / (app - aqq)).atan()
            };

            let c = theta.cos();
            let s = theta.sin();

            // Apply rotation.
            for i in 0..n {
                if i != p && i != q {
                    let aip = a[i][p];
                    let aiq = a[i][q];
                    a[i][p] = c * aip + s * aiq;
                    a[p][i] = a[i][p];
                    a[i][q] = -s * aip + c * aiq;
                    a[q][i] = a[i][q];
                }
            }
            let new_pp = c * c * app + 2.0 * s * c * apq + s * s * aqq;
            let new_qq = s * s * app - 2.0 * s * c * apq + c * c * aqq;
            a[p][p] = new_pp;
            a[q][q] = new_qq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }

        // Extract eigenvalues from diagonal.
        let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // λ₂ is the second smallest (index 1 after sorting).
        // For a connected graph, eigenvalues[0] ≈ 0.
        if eigenvalues.len() >= 2 {
            Ok(eigenvalues[1].max(0.0))
        } else {
            Ok(0.0)
        }
    }

    /// Iterative approach for larger matrices.
    fn _algebraic_connectivity_iterative(&self) -> Result<f64, LaplacianError> {
        let n = self.n;
        let ones: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];

        // Use shifted inverse iteration with the Laplacian.
        // Since we can't invert L easily, we use a different approach:
        // Apply L to vectors orthogonal to ones. The Rayleigh quotient
        // of the result converges to λ_max. To get λ₂, we use the fact that
        // λ₂ = min_{x⊥1} (x^T L x) / (x^T x).
        //
        // We approximate by computing the Rayleigh quotient for several
        // random vectors orthogonal to ones and taking the minimum.

        let num_trials = 20;
        let mut min_rq = f64::MAX;

        for trial in 0..num_trials {
            let mut v: Vec<f64> = (0..n)
                .map(|i| ((i as f64 * (trial as f64 + 1.7) + 0.3) % 7.0) + 0.1)
                .collect();
            deflate(&mut v, &ones);
            let mag = magnitude(&v);
            if mag < 1e-12 {
                continue;
            }
            normalize(&mut v);

            // A few iterations of steepest descent for Rayleigh quotient minimization.
            for _ in 0..100 {
                let lv = self.multiply(&v)?;
                let rq = dot(&v, &lv);

                // Gradient of Rayleigh quotient: 2*(Lv - rq*v)
                let mut grad = lv.clone();
                for i in 0..n {
                    grad[i] -= rq * v[i];
                }
                deflate(&mut grad, &ones);

                // Step in the negative gradient direction.
                let grad_norm = magnitude(&grad);
                if grad_norm < 1e-12 {
                    break;
                }
                for i in 0..n {
                    v[i] -= 0.5 * grad[i];
                }
                deflate(&mut v, &ones);
                normalize(&mut v);

                if rq < min_rq {
                    min_rq = rq;
                }
            }
        }

        Ok(min_rq.max(0.0))
    }

    /// Compute the spectral gap: λ₂ (algebraic connectivity).
    pub fn spectral_gap(&self, max_iterations: usize, tolerance: f64) -> Result<f64, LaplacianError> {
        self.algebraic_connectivity(max_iterations, tolerance)
    }

    /// Get the row count.
    pub fn nrows(&self) -> usize {
        self.n
    }

    /// Set a row from sparse entries.
    pub fn set_row(&mut self, row_idx: usize, entries: Vec<SparseEntry>) -> Result<(), LaplacianError> {
        if row_idx >= self.n {
            return Err(LaplacianError::IndexOutOfBounds {
                index: row_idx,
                size: self.n,
            });
        }
        self.rows[row_idx] = entries;
        Ok(())
    }

    /// Merge a message row into the matrix (updates a single row).
    pub fn merge_message(
        &mut self,
        sender_id: &str,
        entries: Vec<SparseEntry>,
    ) -> Result<bool, LaplacianError> {
        let row_idx: usize = sender_id
            .parse::<usize>()
            .map_err(|_| LaplacianError::DimensionMismatch {
                expected: self.n,
                got: 0,
            })?;
        if row_idx >= self.n {
            return Err(LaplacianError::IndexOutOfBounds {
                index: row_idx,
                size: self.n,
            });
        }
        self.rows[row_idx] = entries;
        Ok(true)
    }

    /// Check if the matrix is complete (all rows have at least one entry).
    pub fn is_complete(&self) -> bool {
        self.rows.iter().all(|r| !r.is_empty())
    }

    /// Compute the degree of each node.
    pub fn degrees(&self) -> Vec<f64> {
        (0..self.n)
            .map(|i| {
                self.rows[i]
                    .iter()
                    .find(|e| e.col == i)
                    .map(|e| e.value)
                    .unwrap_or(0.0)
            })
            .collect()
    }
}

fn normalize(v: &mut [f64]) {
    let mag = magnitude(v);
    if mag > 1e-15 {
        for x in v.iter_mut() {
            *x /= mag;
        }
    }
}

fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn deflate(v: &mut [f64], direction: &[f64]) {
    let projection = dot(v, direction);
    for i in 0..v.len() {
        v[i] -= projection * direction[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let lap = LaplacianMatrix::zeros(3);
        assert_eq!(lap.n, 3);
        assert!(lap.rows.iter().all(|r| r.is_empty()));
    }

    #[test]
    fn test_from_adjacency_line_graph() {
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];
        let lap = LaplacianMatrix::from_adjacency(&adjacency);
        assert_eq!(lap.n, 3);
        let dense = lap.to_dense();
        assert_eq!(dense[0], vec![1.0, -1.0, 0.0]);
        assert_eq!(dense[1], vec![-1.0, 2.0, -1.0]);
        assert_eq!(dense[2], vec![0.0, -1.0, 1.0]);
    }

    #[test]
    fn test_from_adjacency_complete_graph() {
        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let lap = LaplacianMatrix::from_adjacency(&adjacency);
        let dense = lap.to_dense();
        assert_eq!(dense[0], vec![2.0, -1.0, -1.0]);
    }

    #[test]
    fn test_multiply_nullspace() {
        let adjacency = vec![vec![1], vec![0]];
        let lap = LaplacianMatrix::from_adjacency(&adjacency);
        let result = lap.multiply(&[1.0, 1.0]).unwrap();
        assert!(result.iter().all(|&x| x.abs() < 1e-10));
    }

    #[test]
    fn test_multiply_dimension_mismatch() {
        let lap = LaplacianMatrix::zeros(3);
        let result = lap.multiply(&[1.0, 2.0]);
        assert!(matches!(result, Err(LaplacianError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_power_iteration() {
        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let lap = LaplacianMatrix::from_adjacency(&adjacency);
        let (eigenvalue, _) = lap.power_iteration(200, 1e-10).unwrap();
        assert!(
            (eigenvalue - 3.0).abs() < 0.1,
            "got eigenvalue = {eigenvalue}"
        );
    }

    #[test]
    fn test_algebraic_connectivity_connected() {
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];
        let lap = LaplacianMatrix::from_adjacency(&adjacency);
        let lambda2 = lap.algebraic_connectivity(500, 1e-12).unwrap();
        assert!(
            lambda2 > 0.1,
            "λ₂ should be positive for connected graph, got {lambda2}"
        );
        assert!(lambda2 < 3.0, "λ₂ should be < 3.0, got {lambda2}");
    }

    #[test]
    fn test_spectral_gap_complete_graph() {
        let adjacency = vec![
            vec![1, 2, 3],
            vec![0, 2, 3],
            vec![0, 1, 3],
            vec![0, 1, 2],
        ];
        let lap = LaplacianMatrix::from_adjacency(&adjacency);
        let gap = lap.spectral_gap(500, 1e-12).unwrap();
        assert!(
            (gap - 4.0).abs() < 0.5,
            "K4 spectral gap should be ~4.0, got {gap}"
        );
    }

    #[test]
    fn test_empty_matrix_error() {
        let lap = LaplacianMatrix::zeros(0);
        let result = lap.power_iteration(100, 1e-10);
        assert!(matches!(result, Err(LaplacianError::EmptyMatrix)));
    }

    #[test]
    fn test_degrees() {
        let adjacency = vec![vec![1, 2], vec![0], vec![0]];
        let lap = LaplacianMatrix::from_adjacency(&adjacency);
        let deg = lap.degrees();
        assert_eq!(deg, vec![2.0, 1.0, 1.0]);
    }

    #[test]
    fn test_is_complete() {
        let mut lap = LaplacianMatrix::zeros(2);
        assert!(!lap.is_complete());
        lap.rows[0] = vec![SparseEntry {
            col: 0,
            value: 1.0,
        }];
        assert!(!lap.is_complete());
        lap.rows[1] = vec![SparseEntry {
            col: 1,
            value: 1.0,
        }];
        assert!(lap.is_complete());
    }

    #[test]
    fn test_set_row_out_of_bounds() {
        let mut lap = LaplacianMatrix::zeros(2);
        let result = lap.set_row(5, vec![]);
        assert!(matches!(result, Err(LaplacianError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_get_row_dense_out_of_bounds() {
        let lap = LaplacianMatrix::zeros(2);
        let row = lap.get_row_dense(5);
        assert!(row.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_single_node() {
        let lap = LaplacianMatrix::zeros(1);
        let lambda2 = lap.algebraic_connectivity(100, 1e-10).unwrap();
        assert_eq!(lambda2, 0.0);
    }

    #[test]
    fn test_laplacian_row_sums_to_zero() {
        let adjacency = vec![vec![1, 2], vec![0, 3], vec![0, 3], vec![1, 2]];
        let lap = LaplacianMatrix::from_adjacency(&adjacency);
        for i in 0..lap.n {
            let row = lap.get_row_dense(i);
            let sum: f64 = row.iter().sum();
            assert!(sum.abs() < 1e-10, "row {i} sums to {sum}");
        }
    }
}
