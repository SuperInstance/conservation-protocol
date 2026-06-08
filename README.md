# conservation-protocol

Agent communication via Laplacian gossip — the network topology IS the message.

## The Problem

Multi-agent systems need consensus. The standard approaches — Raft, Paxos, Byzantine agreement — all treat the network as a transparent transport layer. The agents send opaque payloads and vote on proposals. But the *structure* of the network already encodes consensus information. A fully connected graph reaches consensus faster than a sparse one. A graph with a bottleneck (small spectral gap) converges slowly. Why not make the network topology itself the communication primitive?

## The Insight

The graph Laplacian matrix L has a deep spectral property: its second-smallest eigenvalue λ₂ (algebraic connectivity) measures how well-connected the graph is. For a connected graph, λ₂ > 0. For a disconnected graph, λ₂ = 0. λ₂ is the spectral gap.

If each agent knows its own row of L (its local connections), and agents gossip rows to each other, every agent eventually reconstructs the full Laplacian. Once they have it, they compute λ₂. If λ₂ exceeds a threshold, the network has reached structural consensus — not about *what* to agree on, but about *whether the network itself is cohesive enough to trust*.

This is conservation in the physics sense: the Laplacian's row sums are always zero (every connection is bidirectional), and the total "energy" γ + η is conserved. Violations of conservation (γ + η ≠ C) indicate corrupted or adversarial messages.

## How It Works

### Message Format

Each `ConservationMessage` carries one agent's sparse Laplacian row in CSR format:

```
Agent 1 (connected to agents 0 and 2):
  laplacian_row = [SparseEntry(0, -1.0), SparseEntry(1, 2.0), SparseEntry(2, -1.0)]
```

The diagonal entry is the degree (number of connections). Off-diagonal entries are -1 for each neighbor. Row sums are always zero.

Messages include an FNV-1a-style signature over `(sender_id, timestamp)` for integrity — not cryptographic, but sufficient for detecting accidental corruption in deterministic simulations.

### Laplacian Matrix

The `LaplacianMatrix` stores rows in sparse CSR format. It provides:
- `from_adjacency()` — build from an adjacency list
- `multiply()` — sparse matrix-vector product
- `power_iteration()` — largest eigenvalue via repeated L·v
- `algebraic_connectivity()` — λ₂ via Jacobi eigenvalue decomposition (n ≤ 8) or steepest-descent Rayleigh quotient minimization (n > 8)

The Jacobi path for small matrices is exact (to floating-point tolerance) — important because agent networks in production are typically small (3–8 nodes). The iterative path for larger networks uses steepest descent on the Rayleigh quotient, deflating against the ones vector to avoid convergence to λ₁ = 0.

### Gossip Protocol

`GossipProtocol` manages the local agent state:

1. **Initialize**: Agent constructs its own Laplacian row from the adjacency list
2. **Broadcast**: Send your row to all neighbors
3. **Receive**: Merge incoming rows into the local Laplacian
4. **Check**: Compute spectral gap, update consensus state
5. **Repeat** until converged or max rounds hit

The `gossip_round()` method encapsulates broadcast + receive + check in one synchronous step.

### Consensus State Machine

```
Awaiting → Converging → Reached → Lost
   ↑          ↑                     │
   └──────────┴─────────────────────┘
```

- **Awaiting**: λ₂ = 0, not enough data yet
- **Converging**: λ₂ > 0 but below threshold
- **Reached**: λ₂ ≥ threshold
- **Lost**: λ₂ dropped after reaching threshold (agent left or data corrupted)

The tracker maintains a full history of spectral gap values, enabling monotonicity checks and convergence rate estimation.

### Violation Detection

The `ViolationDetector` checks two invariants:

1. **Conservation**: γ + η must equal C within tolerance. Violations are corrected by proportional adjustment (harmonic correction — distribute the deviation across γ and η proportional to their magnitudes).
2. **Positive semidefiniteness**: A proper Laplacian has no negative eigenvalues. Negative diagonal entries signal structural corruption. Corrections propagate by adjusting diagonal entries uniformly.

## Code

```rust
use conservation_protocol::{GossipProtocol, ConsensusState};

// 3-agent line: 0 — 1 — 2
let adjacency = vec![vec![1], vec![0, 2], vec![1]];
let mut agent0 = GossipProtocol::new("agent-0", 0, 3, adjacency.clone(), 0.5);

// Broadcast row, receive from neighbors
agent0.broadcast();

// After receiving all rows, check convergence
let gap = agent0.current_spectral_gap();
println!("Spectral gap: {:.3}", gap);  // e.g., 1.0 for the line graph
println!("State: {:?}", agent0.consensus_state());
```

```rust
use conservation_protocol::{ViolationDetector, LaplacianMatrix};

let mut detector = ViolationDetector::new(10.0, 0.01);

// Check γ + η = C
let report = detector.full_check(6.0, 4.0, &laplacian, vec!["agent-0".into()], 100);
if report.has_violations() {
    println!("Corrected: γ={}, η={}", report.corrected_gamma, report.corrected_eta);
}
```

## Module Map

| Module | Responsibility | Key Types |
|---|---|---|
| `message` | Sparse Laplacian row messages with FNV signatures | `ConservationMessage`, `SparseEntry` |
| `laplacian` | Sparse CSR matrix, Jacobi eigenvalues, spectral gap | `LaplacianMatrix` |
| `gossip` | Row broadcast, merge, convergence rounds | `GossipProtocol`, `GossipRoundResult` |
| `consensus` | λ₂ threshold state machine with history | `ConsensusTracker`, `ConsensusState` |
| `violation` | Conservation (γ+η=C) detection, harmonic correction | `ViolationDetector`, `ViolationReport` |

## Design Decisions

**Why Laplacian rows and not eigenvectors?** A Laplacian row is a *local* computation — each agent only needs to know its neighbors. Eigenvectors are *global* and require the full matrix. By gossiping rows, agents progressively reconstruct the global structure without any single agent ever needing the full picture at once.

**Why Jacobi for small matrices?** The Jacobi eigenvalue algorithm is O(n³) per sweep but converges in ~10 sweeps for Laplacians (which are already nearly diagonal-dominant). For n ≤ 8, this is faster and more accurate than iterative methods. The cross-over at n = 8 is empirical — below that, Jacobi's exactness wins; above it, steepest descent scales better.

**Why no external math libraries?** `nalgebra` and `ndarray` are excellent but bring in hundreds of dependencies for what amounts to a 50-line sparse matvec, a 40-line Jacobi sweep, and a 30-line Rayleigh quotient iteration. For agent networks with typically < 100 nodes, the hand-rolled code is both faster to compile and easier to audit.

**Why FNV-1a signatures instead of HMAC?** This crate is designed for deterministic simulations and testing, not adversarial network environments. If you need cryptographic integrity, wrap `ConservationMessage` in your own authenticated channel. The signature scheme is pluggable — the `verify_signature()` method is a one-liner to replace.

**Why the `Awaiting → Converging → Reached → Lost` state machine?** Binary consensus (reached / not reached) loses the trajectory information. The four states capture the *path* to consensus: you can monitor convergence rate, detect stalls, and react to topology changes (agents joining/leaving cause `Lost`). The `rounds_in_state` counter lets you set timeouts per phase.

## Stats

- 61 tests, all passing
- Pure Rust, zero unsafe
- No external math libraries
- Dependencies: `serde`, `serde_json`, `thiserror`

## License

MIT
