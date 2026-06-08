//! Conservation messages carry sparse Laplacian rows between agents.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for message operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum MessageError {
    #[error("invalid sender id: {0}")]
    InvalidSenderId(String),
    #[error("serialization failed: {0}")]
    SerializationFailed(String),
    #[error("deserialization failed: {0}")]
    DeserializationFailed(String),
    #[error("signature verification failed")]
    SignatureVerificationFailed,
    #[error("timestamp is zero")]
    ZeroTimestamp,
}

/// A sparse row entry in CSR-like format: (column_index, value).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SparseEntry {
    pub col: usize,
    pub value: f64,
}

/// A conservation message: one agent's row of the graph Laplacian.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConservationMessage {
    /// Unique identifier of the sending agent.
    pub sender_id: String,
    /// Sparse representation of this agent's Laplacian row.
    pub laplacian_row: Vec<SparseEntry>,
    /// Monotonic timestamp (epoch millis or logical clock).
    pub timestamp: u64,
    /// Simple signature: HMAC-like hash of sender_id + timestamp (hex string).
    pub signature: String,
}

impl ConservationMessage {
    /// Create a new message, computing a simple deterministic signature.
    pub fn new(sender_id: impl Into<String>, laplacian_row: Vec<SparseEntry>, timestamp: u64) -> Self {
        let sender_id = sender_id.into();
        let signature = Self::compute_signature(&sender_id, timestamp);
        Self {
            sender_id,
            laplacian_row,
            timestamp,
            signature,
        }
    }

    /// Compute a simple hash-based signature.
    /// This is a deterministic "signing" scheme (no cryptographic secrets for this crate).
    pub fn compute_signature(sender_id: &str, timestamp: u64) -> String {
        // Simple FNV-1a–style hash from the combined input.
        let data = format!("{sender_id}:{timestamp}");
        let mut hash: u64 = 14_695_981_039_346_656_037;
        for byte in data.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
        format!("{hash:016x}")
    }

    /// Verify this message's signature.
    pub fn verify_signature(&self) -> bool {
        Self::compute_signature(&self.sender_id, self.timestamp) == self.signature
    }

    /// Validate the message has proper content.
    pub fn validate(&self) -> Result<(), MessageError> {
        if self.sender_id.is_empty() {
            return Err(MessageError::InvalidSenderId("empty".into()));
        }
        if self.timestamp == 0 {
            return Err(MessageError::ZeroTimestamp);
        }
        if !self.verify_signature() {
            return Err(MessageError::SignatureVerificationFailed);
        }
        // Check row entries are sorted by column and non-negative columns.
        for _entry in &self.laplacian_row {
            // Column indices are usize so always >= 0.
        }
        // Verify entries are sorted by column index (CSR convention).
        for pair in self.laplacian_row.windows(2) {
            if pair[0].col >= pair[1].col {
                return Err(MessageError::SerializationFailed(
                    "laplacian_row entries must be sorted by column index".into(),
                ));
            }
        }
        Ok(())
    }

    /// Serialize to JSON bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MessageError> {
        serde_json::to_vec(self)
            .map_err(|e| MessageError::SerializationFailed(e.to_string()))
    }

    /// Deserialize from JSON bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, MessageError> {
        serde_json::from_slice(data)
            .map_err(|e| MessageError::DeserializationFailed(e.to_string()))
    }

    /// Create a Laplacian row from adjacency list and sender index.
    pub fn from_adjacency(
        sender_id: impl Into<String>,
        sender_index: usize,
        adjacency: &[Vec<usize>],
        timestamp: u64,
    ) -> Self {
        let n = adjacency.len();
        let mut row = vec![0.0f64; n];
        let degree = adjacency[sender_index].len() as f64;
        for &neighbor in &adjacency[sender_index] {
            row[neighbor] = -1.0;
        }
        row[sender_index] = degree;

        let mut entries: Vec<SparseEntry> = row
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(col, &value)| SparseEntry { col, value })
            .collect();
        entries.sort_by_key(|e| e.col);

        Self::new(sender_id, entries, timestamp)
    }

    /// Convert sparse row to dense vector of given length.
    pub fn to_dense(&self, n: usize) -> Vec<f64> {
        let mut dense = vec![0.0; n];
        for entry in &self.laplacian_row {
            if entry.col < n {
                dense[entry.col] = entry.value;
            }
        }
        dense
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_new() {
        let msg = ConservationMessage::new(
            "agent-1",
            vec![SparseEntry { col: 0, value: 2.0 }, SparseEntry { col: 1, value: -1.0 }],
            1000,
        );
        assert_eq!(msg.sender_id, "agent-1");
        assert_eq!(msg.laplacian_row.len(), 2);
        assert_eq!(msg.timestamp, 1000);
        assert!(!msg.signature.is_empty());
    }

    #[test]
    fn test_signature_deterministic() {
        let sig1 = ConservationMessage::compute_signature("agent-1", 1000);
        let sig2 = ConservationMessage::compute_signature("agent-1", 1000);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_signature_differs_for_different_inputs() {
        let sig1 = ConservationMessage::compute_signature("agent-1", 1000);
        let sig2 = ConservationMessage::compute_signature("agent-2", 1000);
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_verify_valid_signature() {
        let msg = ConservationMessage::new("agent-1", vec![], 1000);
        assert!(msg.verify_signature());
    }

    #[test]
    fn test_verify_tampered_signature() {
        let mut msg = ConservationMessage::new("agent-1", vec![], 1000);
        msg.signature = "0000000000000000".into();
        assert!(!msg.verify_signature());
    }

    #[test]
    fn test_validate_good_message() {
        let msg = ConservationMessage::new(
            "agent-1",
            vec![SparseEntry { col: 0, value: 2.0 }, SparseEntry { col: 2, value: -1.0 }],
            1000,
        );
        assert!(msg.validate().is_ok());
    }

    #[test]
    fn test_validate_empty_sender() {
        let mut msg = ConservationMessage::new("", vec![], 1000);
        msg.sender_id = "".into();
        assert!(matches!(msg.validate(), Err(MessageError::InvalidSenderId(_))));
    }

    #[test]
    fn test_validate_zero_timestamp() {
        let mut msg = ConservationMessage::new("agent-1", vec![], 1000);
        msg.timestamp = 0;
        assert!(matches!(msg.validate(), Err(MessageError::ZeroTimestamp)));
    }

    #[test]
    fn test_validate_unsorted_entries() {
        let mut msg = ConservationMessage::new(
            "agent-1",
            vec![SparseEntry { col: 2, value: -1.0 }, SparseEntry { col: 0, value: 2.0 }],
            1000,
        );
        // Force unsorted entries by direct construction bypassing validation.
        msg.laplacian_row = vec![
            SparseEntry { col: 2, value: -1.0 },
            SparseEntry { col: 0, value: 2.0 },
        ];
        assert!(matches!(msg.validate(), Err(MessageError::SerializationFailed(_))));
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let msg = ConservationMessage::new(
            "agent-42",
            vec![SparseEntry { col: 0, value: 3.0 }, SparseEntry { col: 1, value: -1.0 }, SparseEntry { col: 2, value: -1.0 }],
            9999,
        );
        let bytes = msg.to_bytes().unwrap();
        let decoded = ConservationMessage::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_deserialize_garbage() {
        let result = ConservationMessage::from_bytes(b"not json");
        assert!(matches!(result, Err(MessageError::DeserializationFailed(_))));
    }

    #[test]
    fn test_to_dense() {
        let msg = ConservationMessage::new(
            "agent-1",
            vec![SparseEntry { col: 0, value: 2.0 }, SparseEntry { col: 3, value: -1.0 }],
            1000,
        );
        let dense = msg.to_dense(5);
        assert_eq!(dense, vec![2.0, 0.0, 0.0, -1.0, 0.0]);
    }

    #[test]
    fn test_from_adjacency() {
        // 3-node line: 0 -- 1 -- 2
        let adjacency = vec![vec![1], vec![0, 2], vec![1]];
        let msg = ConservationMessage::from_adjacency("agent-1", 1, &adjacency, 1000);
        let dense = msg.to_dense(3);
        assert_eq!(dense, vec![-1.0, 2.0, -1.0]);
    }

    #[test]
    fn test_sparse_entry_serde_roundtrip() {
        let entry = SparseEntry { col: 5, value: -3.14 };
        let json = serde_json::to_string(&entry).unwrap();
        let back: SparseEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(back, entry);
    }
}
