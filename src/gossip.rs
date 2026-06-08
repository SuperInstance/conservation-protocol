//! Gossip protocol for broadcasting Laplacian rows.

use crate::consensus::ConsensusTracker;
use crate::laplacian::LaplacianMatrix;
use crate::message::ConservationMessage;
use thiserror::Error;

/// Errors for gossip operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum GossipError {
    #[error("agent {agent} not found in network")]
    AgentNotFound { agent: String },
    #[error("network size mismatch")]
    NetworkSizeMismatch,
    #[error("laplacian error: {0}")]
    Laplacian(#[from] crate::laplacian::LaplacianError),
}

/// A gossip round result.
#[derive(Debug, Clone, PartialEq)]
pub struct GossipRoundResult {
    /// Number of rows updated in this round.
    pub rows_updated: usize,
    /// Whether the network has converged.
    pub converged: bool,
    /// Current spectral gap.
    pub spectral_gap: f64,
}

/// The gossip protocol engine.
#[derive(Debug, Clone)]
pub struct GossipProtocol {
    /// Agent id for this node.
    agent_id: String,
    /// Agent index in the network.
    agent_index: usize,
    /// Network size.
    n: usize,
    /// Local view of the global Laplacian.
    laplacian: LaplacianMatrix,
    /// Adjacency list (who we talk to).
    adjacency: Vec<Vec<usize>>,
    /// Consensus tracker.
    consensus: ConsensusTracker,
    /// Outgoing message queue.
    outgoing: Vec<ConservationMessage>,
    /// Monotonic clock for timestamps.
    clock: u64,
}

impl GossipProtocol {
    /// Create a new gossip protocol node.
    pub fn new(
        agent_id: impl Into<String>,
        agent_index: usize,
        n: usize,
        adjacency: Vec<Vec<usize>>,
        spectral_threshold: f64,
    ) -> Self {
        let agent_id = agent_id.into();
        // Build our own row immediately.
        let mut laplacian = LaplacianMatrix::zeros(n);
        if agent_index < n {
            let degree = adjacency.get(agent_index).map(|a| a.len()).unwrap_or(0) as f64;
            let mut entries = Vec::new();
            if let Some(neighbors) = adjacency.get(agent_index) {
                for &j in neighbors {
                    entries.push(crate::message::SparseEntry { col: j, value: -1.0 });
                }
            }
            if degree > 0.0 {
                entries.push(crate::message::SparseEntry { col: agent_index, value: degree });
            }
            entries.sort_by_key(|e| e.col);
            laplacian.rows[agent_index] = entries;
        }

        Self {
            agent_id,
            agent_index,
            n,
            laplacian,
            adjacency,
            consensus: ConsensusTracker::new(spectral_threshold),
            outgoing: Vec::new(),
            clock: 1,
        }
    }

    /// Get the agent id.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Get the network size.
    pub fn network_size(&self) -> usize {
        self.n
    }

    /// Get a reference to the local Laplacian.
    pub fn laplacian(&self) -> &LaplacianMatrix {
        &self.laplacian
    }

    /// Get a mutable reference to the local Laplacian.
    pub fn laplacian_mut(&mut self) -> &mut LaplacianMatrix {
        &mut self.laplacian
    }

    /// Broadcast our row to all neighbors.
    pub fn broadcast(&mut self) -> Vec<ConservationMessage> {
        self.clock += 1;
        let row = self.laplacian.rows.get(self.agent_index).cloned().unwrap_or_default();
        let msg = ConservationMessage::new(
            self.agent_index.to_string(),
            row,
            self.clock,
        );
        let messages = vec![msg.clone(); self.adjacency.get(self.agent_index).map(|a| a.len()).unwrap_or(0)];
        self.outgoing.push(msg);
        // Return one message per neighbor (they're all the same).
        messages
    }

    /// Receive and merge a message from a neighbor.
    pub fn receive(&mut self, msg: &ConservationMessage) -> Result<bool, GossipError> {
        if !msg.verify_signature() {
            return Ok(false);
        }
        let sender = msg.sender_id.clone();
        let entries = msg.laplacian_row.clone();
        self.laplacian.merge_message(&sender, entries)?;
        Ok(true)
    }

    /// Run a complete gossip round: broadcast to all neighbors, receive from all.
    /// This simulates a synchronous round.
    pub fn gossip_round<F>(&mut self, receive_fn: F) -> Result<GossipRoundResult, GossipError>
    where
        F: Fn(usize) -> Option<ConservationMessage>,
    {
        let mut updated = 0;

        // Collect messages first to avoid borrow conflict.
        let neighbor_indices: Vec<usize> = self
            .adjacency
            .get(self.agent_index)
            .cloned()
            .unwrap_or_default();
        let messages: Vec<ConservationMessage> = neighbor_indices
            .iter()
            .filter_map(|&idx| receive_fn(idx))
            .collect();

        for msg in messages {
            if self.receive(&msg)? {
                updated += 1;
            }
        }

        // Compute spectral gap.
        let spectral_gap = self.laplacian.spectral_gap(200, 1e-10).unwrap_or(0.0);
        let converged = self.consensus.check_convergence(spectral_gap);

        // Broadcast our updated row.
        self.broadcast();

        Ok(GossipRoundResult {
            rows_updated: updated,
            converged,
            spectral_gap,
        })
    }

    /// Check if the local Laplacian view is complete (all rows filled).
    pub fn is_view_complete(&self) -> bool {
        self.laplacian.is_complete()
    }

    /// Get the current consensus state.
    pub fn consensus_state(&self) -> &crate::consensus::ConsensusState {
        self.consensus.state()
    }

    /// Get the current spectral gap.
    pub fn current_spectral_gap(&self) -> f64 {
        self.consensus.current_spectral_gap()
    }

    /// Run gossip until convergence or max rounds.
    pub fn run_until_convergence<F>(
        &mut self,
        max_rounds: usize,
        receive_fn: F,
    ) -> Result<GossipRoundResult, GossipError>
    where
        F: Fn(usize, usize) -> Option<ConservationMessage>,
    {
        for round in 0..max_rounds {
            let result = self.gossip_round(|neighbor_idx| receive_fn(round, neighbor_idx))?;
            if result.converged {
                return Ok(result);
            }
        }
        // Return final state even if not converged.
        let spectral_gap = self.laplacian.spectral_gap(200, 1e-10).unwrap_or(0.0);
        Ok(GossipRoundResult {
            rows_updated: 0,
            converged: false,
            spectral_gap,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::SparseEntry;

    #[test]
    fn test_new_protocol() {
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];
        let proto = GossipProtocol::new("agent-0", 0, 3, adjacency, 0.5);
        assert_eq!(proto.agent_id(), "agent-0");
        assert_eq!(proto.network_size(), 3);
    }

    #[test]
    fn test_broadcast() {
        let adjacency = vec![vec![1], vec![0]];
        let mut proto = GossipProtocol::new("agent-0", 0, 2, adjacency, 0.5);
        let msgs = proto.broadcast();
        assert_eq!(msgs.len(), 1); // one neighbor
        assert_eq!(msgs[0].sender_id, "0");
    }

    #[test]
    fn test_receive_valid_message() {
        let adjacency = vec![vec![1], vec![0]];
        let mut proto = GossipProtocol::new("agent-0", 0, 2, adjacency, 0.5);

        let msg = ConservationMessage::new(
            "1",
            vec![
                SparseEntry { col: 0, value: -1.0 },
                SparseEntry { col: 1, value: 1.0 },
            ],
            1,
        );
        let result = proto.receive(&msg).unwrap();
        assert!(result);
    }

    #[test]
    fn test_receive_tampered_message() {
        let adjacency = vec![vec![1], vec![0]];
        let mut proto = GossipProtocol::new("agent-0", 0, 2, adjacency, 0.5);

        let mut msg = ConservationMessage::new("1", vec![], 1);
        msg.signature = "tampered".into();
        let result = proto.receive(&msg).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_is_view_complete() {
        let adjacency = vec![vec![1], vec![0]];
        let mut proto = GossipProtocol::new("agent-0", 0, 2, adjacency, 0.5);
        // Our own row is set, but the other agent's row is not.
        assert!(!proto.is_view_complete());

        // Receive the other agent's row.
        let msg = ConservationMessage::new(
            "1",
            vec![
                SparseEntry { col: 0, value: -1.0 },
                SparseEntry { col: 1, value: 1.0 },
            ],
            1,
        );
        proto.receive(&msg).unwrap();
        assert!(proto.is_view_complete());
    }

    #[test]
    fn test_convergence_detection() {
        // Complete graph converges fast.
        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let mut proto = GossipProtocol::new("agent-0", 0, 3, adjacency, 0.1);

        // Fill all rows to simulate a complete view.
        let msg1 = ConservationMessage::new(
            "1",
            vec![
                SparseEntry { col: 0, value: -1.0 },
                SparseEntry { col: 1, value: 2.0 },
                SparseEntry { col: 2, value: -1.0 },
            ],
            1,
        );
        let msg2 = ConservationMessage::new(
            "2",
            vec![
                SparseEntry { col: 0, value: -1.0 },
                SparseEntry { col: 1, value: -1.0 },
                SparseEntry { col: 2, value: 2.0 },
            ],
            1,
        );
        proto.receive(&msg1).unwrap();
        proto.receive(&msg2).unwrap();

        // Compute spectral gap directly from the laplacian.
        let gap = proto.laplacian().spectral_gap(200, 1e-10).unwrap();
        // For K3, λ₂ = 3.0, which should exceed threshold 0.1.
        assert!(gap > 0.1, "spectral gap should be positive for complete graph, got {gap}");
    }

    #[test]
    fn test_consensus_state_initially_awaiting() {
        let adjacency = vec![vec![1], vec![0]];
        let proto = GossipProtocol::new("agent-0", 0, 2, adjacency, 0.5);
        assert_eq!(*proto.consensus_state(), crate::consensus::ConsensusState::Awaiting);
    }
}
