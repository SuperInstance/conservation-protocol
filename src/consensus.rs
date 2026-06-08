//! Consensus tracking via spectral gap.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Consensus states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusState {
    /// Not enough data yet; waiting for gossip to fill the Laplacian.
    Awaiting,
    /// Gossip is progressing; spectral gap is increasing.
    Converging,
    /// Spectral gap exceeds the threshold — consensus reached.
    Reached,
    /// Consensus lost (spectral gap dropped below threshold after being reached).
    Lost,
}

/// Errors for consensus operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ConsensusError {
    #[error("consensus not reached yet")]
    NotReached,
    #[error("consensus lost")]
    ConsensusLost,
}

/// Tracks convergence via the spectral gap (algebraic connectivity λ₂).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConsensusTracker {
    /// The spectral gap threshold for consensus.
    threshold: f64,
    /// Current spectral gap value.
    current_gap: f64,
    /// Current consensus state.
    state: ConsensusState,
    /// History of spectral gap values.
    history: Vec<f64>,
    /// Number of rounds since state last changed.
    rounds_in_state: usize,
}

impl ConsensusTracker {
    /// Create a new tracker with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            current_gap: 0.0,
            state: ConsensusState::Awaiting,
            history: Vec::new(),
            rounds_in_state: 0,
        }
    }

    /// Check convergence with a new spectral gap reading.
    /// Returns `true` if consensus is reached.
    pub fn check_convergence(&mut self, spectral_gap: f64) -> bool {
        self.history.push(spectral_gap);
        self.current_gap = spectral_gap;
        self.rounds_in_state += 1;

        let new_state = if spectral_gap >= self.threshold {
            ConsensusState::Reached
        } else if spectral_gap > 0.0 {
            if self.state == ConsensusState::Reached {
                ConsensusState::Lost
            } else {
                ConsensusState::Converging
            }
        } else {
            if self.state == ConsensusState::Reached {
                ConsensusState::Lost
            } else {
                ConsensusState::Awaiting
            }
        };

        if new_state != self.state {
            self.state = new_state;
            self.rounds_in_state = 0;
        }

        self.state == ConsensusState::Reached
    }

    /// Get the current spectral gap.
    pub fn current_spectral_gap(&self) -> f64 {
        self.current_gap
    }

    /// Get the threshold.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Get the current consensus state.
    pub fn state(&self) -> &ConsensusState {
        &self.state
    }

    /// Get the history of spectral gap values.
    pub fn history(&self) -> &[f64] {
        &self.history
    }

    /// Number of rounds in current state.
    pub fn rounds_in_state(&self) -> usize {
        self.rounds_in_state
    }

    /// Check if the gap is monotonically increasing over last `n` rounds.
    pub fn is_monotonically_increasing(&self, n: usize) -> bool {
        if self.history.len() < n + 1 {
            return false;
        }
        let tail = &self.history[self.history.len() - n - 1..];
        tail.windows(2).all(|w| w[1] >= w[0])
    }

    /// Get the rate of change of the spectral gap over last `n` rounds.
    pub fn convergence_rate(&self, n: usize) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }
        let tail_len = n.min(self.history.len());
        let tail = &self.history[self.history.len() - tail_len..];
        if tail.len() < 2 {
            return 0.0;
        }
        let first = tail[0];
        let last = tail[tail.len() - 1];
        (last - first) / (tail_len as f64 - 1.0)
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.current_gap = 0.0;
        self.state = ConsensusState::Awaiting;
        self.history.clear();
        self.rounds_in_state = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker() {
        let tracker = ConsensusTracker::new(1.0);
        assert_eq!(tracker.threshold(), 1.0);
        assert_eq!(*tracker.state(), ConsensusState::Awaiting);
        assert_eq!(tracker.current_spectral_gap(), 0.0);
    }

    #[test]
    fn test_awaiting_to_converging() {
        let mut tracker = ConsensusTracker::new(1.0);
        let reached = tracker.check_convergence(0.5);
        assert!(!reached);
        assert_eq!(*tracker.state(), ConsensusState::Converging);
    }

    #[test]
    fn test_converging_to_reached() {
        let mut tracker = ConsensusTracker::new(1.0);
        tracker.check_convergence(0.5);
        let reached = tracker.check_convergence(1.2);
        assert!(reached);
        assert_eq!(*tracker.state(), ConsensusState::Reached);
    }

    #[test]
    fn test_reached_to_lost() {
        let mut tracker = ConsensusTracker::new(1.0);
        tracker.check_convergence(0.5);
        tracker.check_convergence(1.2);
        assert_eq!(*tracker.state(), ConsensusState::Reached);
        let reached = tracker.check_convergence(0.8);
        assert!(!reached);
        assert_eq!(*tracker.state(), ConsensusState::Lost);
    }

    #[test]
    fn test_history() {
        let mut tracker = ConsensusTracker::new(1.0);
        tracker.check_convergence(0.1);
        tracker.check_convergence(0.3);
        tracker.check_convergence(0.7);
        assert_eq!(tracker.history(), &[0.1, 0.3, 0.7]);
    }

    #[test]
    fn test_monotonically_increasing() {
        let mut tracker = ConsensusTracker::new(10.0);
        for v in [0.1, 0.3, 0.5, 0.8, 1.0] {
            tracker.check_convergence(v);
        }
        assert!(tracker.is_monotonically_increasing(3));
    }

    #[test]
    fn test_not_monotonically_increasing() {
        let mut tracker = ConsensusTracker::new(10.0);
        for v in [0.1, 0.5, 0.3, 0.8] {
            tracker.check_convergence(v);
        }
        assert!(!tracker.is_monotonically_increasing(3));
    }

    #[test]
    fn test_convergence_rate() {
        let mut tracker = ConsensusTracker::new(10.0);
        tracker.check_convergence(0.0);
        tracker.check_convergence(1.0);
        let rate = tracker.convergence_rate(2);
        assert!((rate - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut tracker = ConsensusTracker::new(1.0);
        tracker.check_convergence(5.0);
        tracker.reset();
        assert_eq!(*tracker.state(), ConsensusState::Awaiting);
        assert_eq!(tracker.current_spectral_gap(), 0.0);
        assert!(tracker.history().is_empty());
    }

    #[test]
    fn test_rounds_in_state() {
        let mut tracker = ConsensusTracker::new(1.0);
        tracker.check_convergence(0.3);
        // State changed from Awaiting to Converging, so rounds resets to 0.
        assert_eq!(tracker.rounds_in_state(), 0);
        tracker.check_convergence(0.5);
        // Still Converging, incremented.
        assert_eq!(tracker.rounds_in_state(), 1);
        // State change to Reached resets counter to 0.
        tracker.check_convergence(1.5);
        assert_eq!(tracker.rounds_in_state(), 0);
        // Still in Reached state.
        tracker.check_convergence(1.2);
        assert_eq!(tracker.rounds_in_state(), 1);
    }

    #[test]
    fn test_zero_gap_stays_awaiting() {
        let mut tracker = ConsensusTracker::new(1.0);
        tracker.check_convergence(0.0);
        assert_eq!(*tracker.state(), ConsensusState::Awaiting);
    }
}
