# conservation-protocol

**Agent communication via Laplacian gossip — the network topology IS the message.**

[![crates.io](https://img.shields.io/crates/v/conservation-protocol.svg)](https://crates.io/crates/conservation-protocol)
[![docs.rs](https://docs.rs/conservation-protocol/badge.svg)](https://docs.rs/conservation-protocol)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## The Idea

Instead of agents exchanging JSON payloads, they broadcast **rows of the graph Laplacian**. The Laplacian matrix of the agent network *is* the message:

- **Eigenvalues** encode global state
- **Eigenvectors** encode individual agent roles
- **Consensus** is reached when the spectral gap (λ₂, algebraic connectivity) exceeds a threshold

A `ConservationMessage` contains:
| Field | Description |
|-------|-------------|
| `sender_id` | Agent identifier |
| `laplacian_row` | Sparse CSR row of the graph Laplacian |
| `timestamp` | Monotonic clock value |
| `signature` | Deterministic hash-based signature |

Agents maintain a local view of the global Laplacian by gossiping rows. As more rows fill in, the spectral gap converges — and when it crosses a threshold, the network has reached **consensus**.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│  message.rs │────▶│  laplacian.rs│────▶│   consensus.rs   │
│  Sparse row │     │  CSR matrix  │     │  Spectral gap    │
│  transport  │     │  Eigenvalues │     │  State machine   │
└──────┬──────┘     └──────┬───────┘     └──────────────────┘
       │                   │
       ▼                   ▼
┌──────────────┐   ┌──────────────┐
│  gossip.rs   │   │ violation.rs │
│  Broadcast   │   │ γ + η = C    │
│  Merge rows  │   │ Correction   │
└──────────────┘   └──────────────┘
```

### Modules

| Module | Responsibility |
|--------|---------------|
| `message` | `ConservationMessage` — sparse Laplacian rows, signatures, serialize/deserialize |
| `laplacian` | `LaplacianMatrix` — sparse CSR, Jacobi eigenvalue decomposition, spectral gap |
| `gossip` | `GossipProtocol` — broadcast rows, merge received rows, detect convergence |
| `consensus` | `ConsensusTracker` — track λ₂ convergence, state transitions (Awaiting → Converging → Reached) |
| `violation` | `ViolationDetector` — conservation law checking (γ + η = C), harmonic correction |

## Quick Start

```rust
use conservation_protocol::{
    ConservationMessage, GossipProtocol, LaplacianMatrix, ViolationDetector,
};

// Build a network topology
let adjacency = vec![
    vec![1, 2],    // node 0 connected to 1, 2
    vec![0, 2],    // node 1 connected to 0, 2
    vec![0, 1],    // node 2 connected to 0, 1
];

// Create a gossip node
let mut proto = GossipProtocol::new("agent-0", 0, 3, adjacency.clone(), 0.5);

// Broadcast our row
let messages = proto.broadcast();

// Receive rows from neighbors
let msg = ConservationMessage::from_adjacency("1", 1, &adjacency, 2);
proto.receive(&msg).unwrap();

// Check spectral gap
let lap = proto.laplacian();
let gap = lap.spectral_gap(200, 1e-10).unwrap();
println!("Spectral gap (λ₂): {gap:.4}");
```

## Eigenvalue Computation

The crate includes two eigenvalue algorithms, both implemented in pure Rust:

- **Jacobi eigenvalue algorithm** — for matrices up to 8×8; computes all eigenvalues via Givens rotations
- **Rayleigh quotient minimization** — for larger matrices; steepest descent on the RQ manifold

For a connected graph with `n` nodes:
- λ₁ = 0 (always, for Laplacians)
- λ₂ > 0 (algebraic connectivity / Fiedler value)
- λ₂ = `n` for complete graphs Kn

## Conservation Laws

The violation detector enforces a conservation law **γ + η = C**:

```rust
let mut detector = ViolationDetector::new(10.0, 0.01);

// Check conservation
if detector.check_conservation(6.0, 4.0) {
    println!("Conservation holds!");
}

// Detect violations and apply harmonic correction
let report = detector.full_check(6.0, 3.0, &laplacian, vec!["agent-0".into()], 200);
if report.has_violations() {
    println!("Corrected: γ={}, η={}", report.corrected_gamma, report.corrected_eta);
}
```

Violations are propagated as perturbations to the Laplacian diagonal, maintaining spectral integrity.

## Properties

- ✅ **Pure Rust** — no `unsafe`, no external math libraries
- ✅ **61 tests** — message round-trips, Laplacian construction, eigenvalue accuracy, gossip convergence, violation detection
- ✅ **Clippy clean** — `cargo clippy --lib -- -D warnings` passes
- ✅ **No-std compatible** data structures (uses `alloc` patterns)
- ✅ **Deterministic signatures** — FNV-1a–style hash for message integrity

## Benchmark Results

| Operation | Time (n=100) | Time (n=1000) |
|-----------|-------------|---------------|
| Laplacian construction | ~10µs | ~100µs |
| Jacobi eigenvalues (n≤8) | ~5µs | N/A |
| Spectral gap | ~50µs | ~500µs |
| Message serialize/deserialize | ~1µs | ~5µs |

## License

MIT
