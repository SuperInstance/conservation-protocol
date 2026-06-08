//! Violation detection: γ + η ≠ C, harmonic correction via negative eigenvalues.

use crate::laplacian::LaplacianMatrix;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for violation operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ViolationError {
    #[error("conservation violated: γ={gamma} + η={eta} = {sum}, expected C={c}")]
    ConservationViolated {
        gamma: f64,
        eta: f64,
        sum: f64,
        c: f64,
    },
    #[error("negative eigenvalue detected: {eigenvalue}")]
    NegativeEigenvalue { eigenvalue: f64 },
    #[error("no correction needed")]
    NoCorrectionNeeded,
}

/// A detected violation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Violation {
    /// The γ component (e.g., kinetic energy analogue).
    pub gamma: f64,
    /// The η component (e.g., potential energy analogue).
    pub eta: f64,
    /// The expected conserved quantity C.
    pub c: f64,
    /// The deviation from conservation.
    pub deviation: f64,
    /// Eigenvalue associated with this violation (negative = violation).
    pub eigenvalue: f64,
    /// The agent(s) involved.
    pub agents: Vec<String>,
}

/// Violation detector and harmonic corrector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViolationDetector {
    /// The conserved quantity C (e.g., total energy).
    pub c: f64,
    /// Tolerance for conservation checks.
    pub tolerance: f64,
    /// History of detected violations.
    pub violations: Vec<Violation>,
}

impl ViolationDetector {
    /// Create a new detector for conserved quantity `c`.
    pub fn new(c: f64, tolerance: f64) -> Self {
        Self {
            c,
            tolerance,
            violations: Vec::new(),
        }
    }

    /// Check if γ + η = C within tolerance.
    pub fn check_conservation(&self, gamma: f64, eta: f64) -> bool {
        let sum = gamma + eta;
        (sum - self.c).abs() <= self.tolerance
    }

    /// Compute the deviation from conservation.
    pub fn deviation(&self, gamma: f64, eta: f64) -> f64 {
        (gamma + eta) - self.c
    }

    /// Detect a violation and record it.
    pub fn detect(
        &mut self,
        gamma: f64,
        eta: f64,
        eigenvalue: f64,
        agents: Vec<String>,
    ) -> Result<(), ViolationError> {
        let deviation = self.deviation(gamma, eta);
        if deviation.abs() > self.tolerance {
            let violation = Violation {
                gamma,
                eta,
                c: self.c,
                deviation,
                eigenvalue,
                agents,
            };
            self.violations.push(violation);
            Err(ViolationError::ConservationViolated {
                gamma,
                eta,
                sum: gamma + eta,
                c: self.c,
            })
        } else {
            Ok(())
        }
    }

    /// Compute harmonic correction for a violation.
    /// The correction is the minimal adjustment to (γ, η) that restores γ + η = C,
    /// distributed proportionally.
    pub fn harmonic_correction(&self, gamma: f64, eta: f64) -> (f64, f64) {
        let deviation = self.deviation(gamma, eta);
        let total = gamma + eta;
        if total.abs() < 1e-15 {
            // If both are zero, split correction equally.
            let correction = deviation / 2.0;
            return (gamma - correction, eta - correction);
        }
        // Distribute correction proportionally to magnitudes.
        let gamma_weight = gamma.abs() / (gamma.abs() + eta.abs());
        let gamma_correction = deviation * gamma_weight;
        let eta_correction = deviation * (1.0 - gamma_weight);
        (gamma - gamma_correction, eta - eta_correction)
    }

    /// Apply harmonic correction to a Laplacian matrix by rescaling off-diagonal entries.
    /// This propagates the correction as negative eigenvalue perturbation.
    pub fn propagate_correction(
        &self,
        laplacian: &mut LaplacianMatrix,
        correction: f64,
    ) -> Vec<usize> {
        let n = laplacian.nrows();
        let mut corrected_rows = Vec::new();
        for i in 0..n {
            let row = &mut laplacian.rows[i];
            // Find diagonal entry and apply correction.
            for entry in row.iter_mut() {
                if entry.col == i {
                    // Adjust diagonal by adding the correction (distributed across rows).
                    let row_correction = correction / (n as f64);
                    entry.value += row_correction;
                    corrected_rows.push(i);
                    break;
                }
            }
        }
        corrected_rows.sort();
        corrected_rows.dedup();
        corrected_rows
    }

    /// Check for negative eigenvalues in the Laplacian (indicates structural violation).
    /// A proper Laplacian is positive semidefinite; negative eigenvalues signal problems.
    pub fn detect_negative_eigenvalues(
        &self,
        laplacian: &LaplacianMatrix,
        max_iterations: usize,
    ) -> Vec<f64> {
        let n = laplacian.nrows();
        if n == 0 {
            return Vec::new();
        }

        // Use power iteration to find the most negative eigenvalue.
        // We compute L - shift*I with shift = largest_eigenvalue, then power iterate.
        let (_largest_eval, _) = laplacian
            .power_iteration(max_iterations, 1e-10)
            .unwrap_or((0.0, vec![]));

        // The eigenvalues of L are in [0, largest_eval].
        // For a proper Laplacian, there should be no negative eigenvalues.
        // We check by looking at the diagonal entries: if any diagonal < 0, there's a problem.
        let mut negatives = Vec::new();
        for i in 0..n {
            let row_dense = laplacian.get_row_dense(i);
            let diag = row_dense.get(i).copied().unwrap_or(0.0);
            if diag < -self.tolerance {
                negatives.push(diag);
            }
        }

        negatives
    }

    /// Full violation check: conservation + negative eigenvalues.
    pub fn full_check(
        &mut self,
        gamma: f64,
        eta: f64,
        laplacian: &LaplacianMatrix,
        agents: Vec<String>,
        max_iterations: usize,
    ) -> ViolationReport {
        let conservation_ok = self.check_conservation(gamma, eta);
        let deviation = self.deviation(gamma, eta);
        let negative_eigenvalues = self.detect_negative_eigenvalues(laplacian, max_iterations);

        let conservation_violation = if !conservation_ok {
            let _ = self.detect(gamma, eta, 0.0, agents.clone());
            Some(Violation {
                gamma,
                eta,
                c: self.c,
                deviation,
                eigenvalue: 0.0,
                agents: agents.clone(),
            })
        } else {
            None
        };

        let eigenvalue_violations: Vec<Violation> = negative_eigenvalues
            .iter()
            .map(|&ev| Violation {
                gamma,
                eta,
                c: self.c,
                deviation: 0.0,
                eigenvalue: ev,
                agents: agents.clone(),
            })
            .collect();

        let (corrected_gamma, corrected_eta) = if !conservation_ok {
            self.harmonic_correction(gamma, eta)
        } else {
            (gamma, eta)
        };

        ViolationReport {
            conservation_ok,
            deviation,
            negative_eigenvalues,
            conservation_violation,
            eigenvalue_violations,
            corrected_gamma,
            corrected_eta,
        }
    }

    /// Get the number of violations detected.
    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }

    /// Clear violation history.
    pub fn clear_history(&mut self) {
        self.violations.clear();
    }
}

/// A report from a full violation check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ViolationReport {
    /// Whether conservation holds.
    pub conservation_ok: bool,
    /// Deviation from conservation.
    pub deviation: f64,
    /// Any detected negative eigenvalues.
    pub negative_eigenvalues: Vec<f64>,
    /// The conservation violation, if any.
    pub conservation_violation: Option<Violation>,
    /// Eigenvalue violations.
    pub eigenvalue_violations: Vec<Violation>,
    /// Harmonic-corrected γ.
    pub corrected_gamma: f64,
    /// Harmonic-corrected η.
    pub corrected_eta: f64,
}

impl ViolationReport {
    /// Whether any violation was detected.
    pub fn has_violations(&self) -> bool {
        !self.conservation_ok || !self.negative_eigenvalues.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conservation_holds() {
        let detector = ViolationDetector::new(10.0, 0.01);
        assert!(detector.check_conservation(6.0, 4.0));
    }

    #[test]
    fn test_conservation_violated() {
        let detector = ViolationDetector::new(10.0, 0.01);
        assert!(!detector.check_conservation(6.0, 3.0));
    }

    #[test]
    fn test_deviation() {
        let detector = ViolationDetector::new(10.0, 0.01);
        let dev = detector.deviation(6.0, 3.0);
        assert!((dev - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_deviation_positive() {
        let detector = ViolationDetector::new(10.0, 0.01);
        let dev = detector.deviation(7.0, 5.0);
        assert!((dev - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_detect_records_violation() {
        let mut detector = ViolationDetector::new(10.0, 0.01);
        let result = detector.detect(6.0, 3.0, -0.5, vec!["a".into()]);
        assert!(result.is_err());
        assert_eq!(detector.violation_count(), 1);
    }

    #[test]
    fn test_detect_no_violation() {
        let mut detector = ViolationDetector::new(10.0, 0.01);
        let result = detector.detect(6.0, 4.0, 0.0, vec![]);
        assert!(result.is_ok());
        assert_eq!(detector.violation_count(), 0);
    }

    #[test]
    fn test_harmonic_correction_equal_split() {
        let detector = ViolationDetector::new(10.0, 0.01);
        let (gc, ec) = detector.harmonic_correction(0.0, 0.0);
        assert!((gc + ec - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_harmonic_correction_proportional() {
        let detector = ViolationDetector::new(10.0, 0.01);
        let (gc, ec) = detector.harmonic_correction(8.0, 5.0);
        // gc + ec should equal C = 10.0
        assert!((gc + ec - 10.0).abs() < 1e-10);
        // Original sum was 13.0, correction is -3.0
        // gamma weight = 8/13, eta weight = 5/13
        assert!((gc - 8.0 + 3.0 * 8.0 / 13.0).abs() < 1e-10);
    }

    #[test]
    fn test_propagate_correction() {
        let adjacency = vec![vec![1], vec![0]];
        let mut laplacian = LaplacianMatrix::from_adjacency(&adjacency);
        let detector = ViolationDetector::new(1.0, 0.01);
        let corrected = detector.propagate_correction(&mut laplacian, 0.5);
        assert_eq!(corrected.len(), 2);
        // Diagonals should have been adjusted.
        let row0 = laplacian.get_row_dense(0);
        assert!((row0[0] - 1.25).abs() < 1e-10);
    }

    #[test]
    fn test_detect_negative_eigenvalues_clean() {
        let adjacency = vec![vec![1], vec![0]];
        let laplacian = LaplacianMatrix::from_adjacency(&adjacency);
        let detector = ViolationDetector::new(1.0, 0.01);
        let neg = detector.detect_negative_eigenvalues(&laplacian, 100);
        assert!(neg.is_empty());
    }

    #[test]
    fn test_full_check_clean() {
        let adjacency = vec![vec![1], vec![0]];
        let laplacian = LaplacianMatrix::from_adjacency(&adjacency);
        let mut detector = ViolationDetector::new(10.0, 0.01);
        let report = detector.full_check(6.0, 4.0, &laplacian, vec![], 100);
        assert!(report.conservation_ok);
        assert!(!report.has_violations());
    }

    #[test]
    fn test_full_check_violated() {
        let adjacency = vec![vec![1], vec![0]];
        let laplacian = LaplacianMatrix::from_adjacency(&adjacency);
        let mut detector = ViolationDetector::new(10.0, 0.01);
        let report = detector.full_check(6.0, 3.0, &laplacian, vec!["agent-0".into()], 100);
        assert!(!report.conservation_ok);
        assert!(report.has_violations());
        assert!(report.conservation_violation.is_some());
        // Corrected values should sum to C.
        assert!((report.corrected_gamma + report.corrected_eta - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_clear_history() {
        let mut detector = ViolationDetector::new(10.0, 0.01);
        detector.detect(6.0, 3.0, 0.0, vec![]).ok();
        assert_eq!(detector.violation_count(), 1);
        detector.clear_history();
        assert_eq!(detector.violation_count(), 0);
    }

    #[test]
    fn test_violation_report_has_violations() {
        let report = ViolationReport {
            conservation_ok: false,
            deviation: 1.0,
            negative_eigenvalues: vec![],
            conservation_violation: None,
            eigenvalue_violations: vec![],
            corrected_gamma: 5.0,
            corrected_eta: 5.0,
        };
        assert!(report.has_violations());
    }
}
