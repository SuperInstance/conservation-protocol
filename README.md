# conservation-protocol

Laplacian messaging protocol — encode, transmit, and verify graph spectral properties across distributed systems using conservation ratios.

## What This Gives You

- **Spectral fingerprints** — encode graph Laplacian eigenvalues as portable fingerprints
- **Conservation ratio messaging** — transmit CR = λ₂/λₙ across the network
- **Compatibility checking** — determine if two systems are spectrally aligned
- **Anomaly propagation** — broadcast anomaly detection across nodes
- **14 tests** — verified protocol encoding/decoding

## Quick Start

```rust
use conservation_protocol::{SpectralFingerprint, ConsonanceMessage};

// Build spectral fingerprint from graph adjacency
let adj = vec![
    vec![0.0, 1.0, 1.0],
    vec![1.0, 0.0, 1.0],
    vec![1.0, 1.0, 0.0],
];
let fp = SpectralFingerprint::from_graph(&adj);

println!("Conservation ratio: {:.4}", fp.conservation);
println!("Eigenvalues: {:?}", fp.eigenvalues);
println!("Fiedler vector: {:?}", fp.fiedler);

// Encode for transmission
let encoded = fp.encode();

// Check compatibility between two systems
let fp2 = SpectralFingerprint::from_graph(&adj2);
let compatible = fp.is_compatible(&fp2, threshold=0.95);
```

## API Reference

| Type | Description |
|---|---|
| `SpectralFingerprint` | Eigenvalues, CR, Fiedler vector, encode/decode |
| `.alignment(other)` | Cosine similarity of eigenvalue spectra |
| `.is_compatible(other, threshold)` | Check spectral alignment |
| `.misaligned_fraction(other)` | 1 − alignment |
| `.encode()` / `SpectralFingerprint::decode()` | Binary serialization |

## How It Fits

The **messaging protocol** of the conservation spectral ecosystem:

- [conservation-spectral-python](https://github.com/SuperInstance/conservation-spectral-python) — Python SDK
- [conservation-spectral-js](https://github.com/SuperInstance/conservation-spectral-js) — TypeScript SDK
- [conservation-spectral-ada](https://github.com/SuperInstance/conservation-spectral-ada) — Ada port (DO-178C)
- [conservation-conformance](https://github.com/SuperInstance/conservation-conformance) — cross-language conformance tests
- [constraint-mux](https://github.com/SuperInstance/constraint-mux) — serial multiplexer using this protocol

## Testing

```bash
cargo test  # 14 tests
```

## Installation

```bash
cargo add conservation-protocol
```

## License

MIT
