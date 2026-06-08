//! # Conservation Protocol
//!
//! Agent-to-agent communication where the Laplacian matrix of the network IS the message.
//!
//! Instead of sending JSON payloads, agents broadcast rows of the graph Laplacian.
//! Eigenvalues encode global state; eigenvectors encode individual roles.
//! Consensus is reached when the spectral gap (λ₂) exceeds a threshold.

pub mod consensus;
pub mod gossip;
pub mod laplacian;
pub mod message;
pub mod violation;

pub use consensus::ConsensusTracker;
pub use gossip::GossipProtocol;
pub use laplacian::LaplacianMatrix;
pub use message::ConservationMessage;
pub use violation::ViolationDetector;
