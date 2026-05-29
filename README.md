# conservation-protocol

**Agent-to-agent communication where the Laplacian IS the message.**

Pure Rust, zero dependencies. Eigenvalue spectra as identity, cosine similarity as alignment, conservation ratio as confidence.

## The problem

Agents need to decide: *should I trust this other agent? Should I compose with it?*

Today, agents negotiate through text — JSON payloads, capability documents, trust scores expressed in natural language. All of it is gameable. An agent can claim whatever capabilities it wants. A reputation can be farmed. A text-based trust score can be spoofed.

Spectral fingerprints can't be faked. You either have the graph structure or you don't. The eigenvalues of your Laplacian are a mathematical consequence of what you *are*, not what you *say*.

## The core idea

Every agent has an internal structure — a graph of capabilities, states, or relationships. Compute the Laplacian of that graph. The eigenvalue spectrum is the agent's **spectral fingerprint**. When two agents meet, they exchange fingerprints. The cosine similarity of their eigenvalue spectra (`α`, alignment) tells them how structurally similar they are. The conservation ratio (`CR = λ₂ / λₙ`) tells them how internally coherent each one is. Together, these numbers gate trust and composition — no words required.

```
SpectralFingerprint → alignment(α) → trust → composition gate
```

## How it works

### 1. Build a fingerprint from a graph

```rust
use conservation_protocol::*;

// Your agent's internal structure as an adjacency matrix
let graph = vec![
    vec![0.0, 1.0, 1.0, 0.0],
    vec![1.0, 0.0, 1.0, 1.0],
    vec![1.0, 1.0, 0.0, 1.0],
    vec![0.0, 1.0, 1.0, 0.0],
];

let fp = SpectralFingerprint::from_graph(&graph);
println!("eigenvalues: {:?}", fp.eigenvalues);
println!("CR = {:.4}", fp.conservation);  // λ₂ / λₙ
```

The fingerprint contains:
- **eigenvalues** — sorted spectrum of the graph Laplacian
- **conservation** — the CR, ratio of Fiedler value to max eigenvalue
- **fiedler** — the Fiedler vector (eigenvector of λ₂), useful for partitioning
- **capacity** — graph size

### 2. Two agents handshake

```rust
let agent_a = AgentIdentity::new(&graph_a, "agent-alpha");
let agent_b = AgentIdentity::new(&graph_b, "agent-beta");

let mut proto = ConservationProtocol::new(agent_a.clone());

// A sends a Hello to B with its fingerprint
let greeting = proto.greet(&agent_b);
println!("alignment α = {:.4}", greeting.alignment);

// A receives B's response
let response = proto.receive(greeting).unwrap();
match &response.message {
    SpectralMessage::Confirm(cr) => println!("Trusted! CR = {:.4}", cr),
    SpectralMessage::Reject(reason) => println!("Distrusted: {}", reason),
    _ => {}
}
```

### 3. Alignment and trust

- **alignment(α)** — cosine similarity of eigenvalue spectra. 1.0 = identical structure, 0.0 = orthogonal.
- **trust** — if α ≥ 0.3 (configurable), the agents trust each other.
- **misaligned_fraction** — `1 - α`. This is the agent's *individuality*: how much of its structure is unique.

```rust
let alpha = fp_a.alignment(&fp_b);
let individuality = fp_a.misaligned_fraction(&fp_b);
println!("We share {:.1}% structure, differ by {:.1}%",
         alpha * 100.0, individuality * 100.0);
```

### 4. The CompositionGate

When agents consider merging, the gate decides:

| Result | CR | Meaning |
|--------|----|---------|
| **Approved** | ≥ 0.67 | The composed system is coherent enough to proceed |
| **Deferred** | 0.3 – 0.67 | Possible with work — the gate returns specific edge suggestions to improve connectivity |
| **Rejected** | < 0.3 | The composition would be incoherent. Don't do it. |

```rust
let gate = CompositionGate::default();
let composed = proto.compose(&agent_b);

match gate.evaluate(&composed) {
    GateResult::Approved(cr) => {
        println!("✓ Approved (CR = {:.4})", cr);
        // Proceed with composition
    }
    GateResult::Deferred(suggestions) => {
        println!("⏳ Deferred. Add these edges to improve CR:");
        for (a, b) in suggestions {
            println!("  connect node {} ↔ {}", a, b);
        }
    }
    GateResult::Rejected(reason) => {
        println!("✗ Rejected: {}", reason);
    }
}
```

The golden ratio target: `1/φ ≈ 0.618`. The default gate uses 0.67 as the approval threshold (slightly above golden ratio) to ensure composed systems are well-connected but not overly constrained.

### 5. Wire format

Fingerprints serialize to a compact binary format for network transmission:

```rust
// Encode
let bytes: Vec<u8> = fp.encode();

// Decode
let recovered = SpectralFingerprint::decode(&bytes)?;

// Use in any transport — HTTP, WebSocket, whatever
send_over_network(&bytes);
```

Format: `[capacity: u64][eigenvalue_count: u64][eigenvalues: f64...][conservation: f64][fiedler_count: u64][fiedler: f64...]` — all little-endian.

### 6. Routing

An agent that knows multiple other agents can route messages by structural similarity:

```rust
let prefs = proto.routing_preference();
// Returns Vec<(agent_id, alignment)> sorted by alignment descending
// Route to the most structurally similar agent first
```

## The math

- **Graph Laplacian**: `L = D - A` where D is the degree matrix and A is the adjacency matrix.
- **Eigenvalue spectrum**: the sorted eigenvalues of L. This is an isomorphism invariant — same graph, same spectrum (regardless of node labeling).
- **Conservation Ratio (CR)**: `λ₂ / λₙ`. The Fiedler value over the max eigenvalue. Measures how well-connected the graph is relative to its size. High CR = tight, coherent structure. Low CR = loose, fragmented.
- **Alignment (α)**: cosine similarity of two sorted eigenvalue vectors. Measures structural similarity between two agents.
- **Misaligned fraction**: `1 - α`. This is *individuality* — the part of your structure that doesn't overlap with another agent.

## Honest limitations

This is not a silver bullet. Know what you're getting:

1. **Cospectral graphs fool it.** Non-isomorphic graphs can have identical eigenvalue spectra. This is rare for random graphs but possible for carefully constructed ones. If an adversary knows your spectrum and engineers a matching one, alignment alone won't catch it.

2. **Requires pre-shared graph structure.** Both agents need to agree on what the adjacency matrix represents. The protocol doesn't solve the semantic alignment problem — it solves the structural one.

3. **Doesn't capture semantic meaning.** Two agents with similar eigenvalue spectra may have completely different capabilities. Alignment measures structural similarity, not functional compatibility.

4. **QR iteration is O(n³).** This implementation uses QR iteration for eigenvalue computation. For very large graphs, you'd want Lanczos or ARPACK. The code is written for clarity, not speed.

5. **Composition is heuristic.** The composed graph is built by block-diagonal concatenation with coupling edges weighted by alignment. This is one reasonable construction, not the only one.

## Architecture

```
SpectralFingerprint   — eigenvalue spectrum, CR, Fiedler vector, encode/decode
AgentIdentity         — fingerprint + id + capabilities + confidence
ConservationProtocol  — greet, receive, trust, compose, routing
CompositionGate       — approve / defer / reject with suggestions
Envelope/SpectralMessage — wire protocol for agent communication
```

## Running

```bash
cargo run
```

This runs the demo: two small graphs, their fingerprints, a handshake, and a composition gate evaluation.

## Testing

```bash
cargo test
```

Tests cover encode/decode roundtrips, alignment of identical graphs (= 1.0), alignment of different graphs (< 1.0), gate approved/rejected/deferred cases, trust thresholds, and routing preference ordering.

## License

MIT

## Ecosystem Integration

- Defines the wire protocol and service interfaces for conservation-law enforcement
- Consumed by `a2a-constraint-protocol` for agent-to-agent constraint communication
- Integrates with `conservation-regime` for regime-aware protocol handling
- Feeds `emergent-coupling` for detecting emergent conservation structures
- Central to the fleet coordination and multi-agent architecture

