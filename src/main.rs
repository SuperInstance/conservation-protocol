//! Conservation Protocol: the Laplacian of the network IS the message.
//!
//! Demonstrates building a graph Laplacian, computing spectral properties,
//! running gossip rounds, and checking conservation laws.

use conservation_protocol::{
    LaplacianMatrix, GossipProtocol, ViolationDetector,
};

fn main() {
    println!("conservation-protocol: Laplacian gossip agent messaging\n");

    // Build a Laplacian from a simple triangle graph
    let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    let lap = LaplacianMatrix::from_adjacency(&adjacency);
    println!("3-node triangle: {} rows", lap.nrows());

    // Compute algebraic connectivity
    let lambda2 = lap.algebraic_connectivity(500, 1e-12).unwrap();
    println!("Algebraic connectivity (λ₂): {:.4}", lambda2);

    // Gossip protocol
    let mut proto = GossipProtocol::new("agent-0", 0, 3, adjacency, 0.5);
    let msgs = proto.broadcast();
    println!("Broadcast {} messages", msgs.len());

    // Conservation check
    let detector = ViolationDetector::new(10.0, 0.01);
    let ok = detector.check_conservation(6.0, 4.0);
    println!("Conservation (6+4=10): {}", if ok { "OK" } else { "VIOLATED" });
}
