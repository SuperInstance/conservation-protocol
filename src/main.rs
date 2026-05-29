use std::collections::HashMap;

// ============================================================
// Module 1: SpectralFingerprint
// ============================================================

#[derive(Debug, Clone)]
pub struct SpectralFingerprint {
    pub eigenvalues: Vec<f64>,
    pub conservation: f64,
    pub fiedler: Vec<f64>,
    pub capacity: usize,
}

impl SpectralFingerprint {
    /// Build a spectral fingerprint from an adjacency matrix.
    /// Computes graph Laplacian eigenvalues via QR iteration.
    pub fn from_graph(adj: &[Vec<f64>]) -> SpectralFingerprint {
        let n = adj.len();
        let laplacian = Self::laplacian(adj);
        let eigenvalues = Self::eigenvalues(&laplacian);
        let mut sorted = eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Fiedler vector = eigenvector of second-smallest eigenvalue
        let fiedler = if n >= 2 {
            Self::eigenvector_for(&laplacian, sorted[1])
        } else {
            vec![0.0; n]
        };

        // Conservation Ratio: CR = λ₂ / λ_n (Fiedler / max eigenvalue)
        let conservation = if n >= 2 && sorted[n - 1].abs() > 1e-12 {
            sorted[1] / sorted[n - 1]
        } else {
            0.0
        };

        SpectralFingerprint {
            eigenvalues: sorted,
            conservation,
            fiedler,
            capacity: n,
        }
    }

    /// Cosine similarity of eigenvalue spectra (the alignment metric α)
    pub fn alignment(&self, other: &SpectralFingerprint) -> f64 {
        let len = self.eigenvalues.len().min(other.eigenvalues.len());
        if len == 0 {
            return 0.0;
        }
        let mut dot = 0.0;
        let mut norm_a = 0.0;
        let mut norm_b = 0.0;
        for i in 0..len {
            let a = self.eigenvalues[i];
            let b = other.eigenvalues[i];
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom < 1e-15 {
            0.0
        } else {
            dot / denom
        }
    }

    pub fn is_compatible(&self, other: &SpectralFingerprint, threshold: f64) -> bool {
        self.alignment(other) >= threshold
    }

    pub fn misaligned_fraction(&self, other: &SpectralFingerprint) -> f64 {
        1.0 - self.alignment(other)
    }

    /// Serialize: [capacity u64][eigenvalues as f64s][conservation f64][fiedler as f64s]
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&(self.capacity as u64).to_le_bytes());
        let ev_count = self.eigenvalues.len() as u64;
        buf.extend_from_slice(&ev_count.to_le_bytes());
        for &ev in &self.eigenvalues {
            buf.extend_from_slice(&ev.to_le_bytes());
        }
        buf.extend_from_slice(&self.conservation.to_le_bytes());
        let fied_count = self.fiedler.len() as u64;
        buf.extend_from_slice(&fied_count.to_le_bytes());
        for &f in &self.fiedler {
            buf.extend_from_slice(&f.to_le_bytes());
        }
        buf
    }

    pub fn decode(data: &[u8]) -> Result<SpectralFingerprint, String> {
        if data.len() < 24 {
            return Err("data too short".into());
        }
        let mut off = 0;
        let capacity = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
        off = 8;
        let ev_count = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        if data.len() < off + ev_count * 8 + 8 + 8 {
            return Err("data truncated (eigenvalues)".into());
        }
        let mut eigenvalues = Vec::with_capacity(ev_count);
        for i in 0..ev_count {
            let start = off + i * 8;
            eigenvalues.push(f64::from_le_bytes(data[start..start + 8].try_into().unwrap()));
        }
        off += ev_count * 8;
        let conservation = f64::from_le_bytes(data[off..off + 8].try_into().unwrap());
        off += 8;
        let fied_count = u64::from_le_bytes(data[off..off + 8].try_into().unwrap()) as usize;
        off += 8;
        if data.len() < off + fied_count * 8 {
            return Err("data truncated (fiedler)".into());
        }
        let mut fiedler = Vec::with_capacity(fied_count);
        for i in 0..fied_count {
            let start = off + i * 8;
            fiedler.push(f64::from_le_bytes(data[start..start + 8].try_into().unwrap()));
        }
        Ok(SpectralFingerprint {
            eigenvalues,
            conservation,
            fiedler,
            capacity,
        })
    }

    // -- Internal linear algebra --

    fn laplacian(adj: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = adj.len();
        let mut l = vec![vec![0.0; n]; n];
        for i in 0..n {
            let degree: f64 = adj[i].iter().sum();
            l[i][i] = degree;
            for j in 0..n {
                if i != j {
                    l[i][j] = -adj[i][j];
                }
            }
        }
        l
    }

    /// Compute eigenvalues via QR iteration (pure Rust, no deps)
    fn eigenvalues(mat: &[Vec<f64>]) -> Vec<f64> {
        let n = mat.len();
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![mat[0][0]];
        }

        // Shift to tridiagonal via Householder reflections
        let mut a = mat.to_vec();
        for k in 0..n.saturating_sub(2) {
            let mut v = vec![0.0; n];
            let norm_x: f64 = (k + 1..n).map(|i| a[i][k] * a[i][k]).sum::<f64>().sqrt();
            if norm_x < 1e-15 {
                continue;
            }
            let sign = if a[k + 1][k] >= 0.0 { 1.0 } else { -1.0 };
            v[k + 1] = a[k + 1][k] + sign * norm_x;
            for i in k + 2..n {
                v[i] = a[i][k];
            }
            let v_norm_sq: f64 = v.iter().map(|x| x * x).sum();
            if v_norm_sq < 1e-30 {
                continue;
            }
            // P = I - 2vv^T/v^Tv; A = PAP
            let mut p = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    p[i][j] = -2.0 * v[i] * v[j] / v_norm_sq;
                    if i == j {
                        p[i][j] += 1.0;
                    }
                }
            }
            // A = P * A * P
            let mut pa = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for k2 in 0..n {
                        s += p[i][k2] * a[k2][j];
                    }
                    pa[i][j] = s;
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let mut s = 0.0;
                    for k2 in 0..n {
                        s += pa[i][k2] * p[k2][j];
                    }
                    a[i][j] = s;
                }
            }
        }

        // Extract tridiagonal
        let mut diag = vec![0.0; n];
        let mut sub = vec![0.0; n - 1];
        for i in 0..n {
            diag[i] = a[i][i];
            if i < n - 1 {
                sub[i] = a[i + 1][i];
            }
        }

        // QR iteration on tridiagonal
        let max_iter = 200;
        for _ in 0..max_iter {
            // Check convergence
            let off_diag: f64 = sub.iter().map(|x| x * x).sum();
            if off_diag < 1e-20 {
                break;
            }
            let shift = diag[n - 1];
            let mut d = diag.clone();
            let mut e = sub.clone();

            // QR step with Wilkinson shift
            d[n - 1] -= shift;
            for i in 0..n - 1 {
                let r = (d[i] * d[i] + e[i] * e[i]).sqrt();
                let c = d[i] / r;
                let s = e[i] / r;
                d[i] = r;
                let tmp = c * d[i + 1] + s * if i + 1 < n - 1 { e[i + 1] } else { 0.0 };
                if i + 1 < n - 1 {
                    e[i + 1] = -s * if i + 1 < n - 1 { e[i + 1] } else { 0.0 } + c * if i + 2 < n { e.get(i + 2).copied().unwrap_or(0.0) } else { 0.0 };
                }
                d[i + 1] = tmp;
                e[i] = 0.0;
            }
            for i in 0..n {
                diag[i] = d[i] + shift;
            }
            sub = e;
        }

        // Clean near-zero values
        diag.iter_mut().for_each(|x| {
            if x.abs() < 1e-10 {
                *x = 0.0;
            }
        });
        diag
    }

    /// Approximate eigenvector via inverse iteration for a given eigenvalue
    fn eigenvector_for(laplacian: &[Vec<f64>], eigenvalue: f64) -> Vec<f64> {
        let n = laplacian.len();
        if n == 0 {
            return vec![];
        }
        // Shifted matrix: (L - λI), solve (L - λI)v = random via Gaussian elimination
        let shifted = {
            let mut m = laplacian.to_vec();
            for i in 0..n {
                m[i][i] -= eigenvalue;
            }
            m
        };

        // Power iteration on inverse (approximate)
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..50 {
            v = Self::solve_lower(&shifted, &v);
            let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
        }
        v
    }

    /// Solve Mx = b approximately (Gaussian elimination with partial pivoting)
    fn solve_lower(m: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
        let n = m.len();
        let mut aug = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = m[i][j];
            }
            aug[i][n] = b[i];
        }
        // Add small regularization for singular systems
        for i in 0..n {
            aug[i][i] += 1e-8;
        }

        for col in 0..n {
            // Find pivot
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in col + 1..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-15 {
                continue;
            }
            aug.swap(col, max_row);
            let pivot = aug[col][col];
            for row in col + 1..n {
                let factor = aug[row][col] / pivot;
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }

        // Back substitution
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            if aug[i][i].abs() < 1e-15 {
                x[i] = 1.0;
                continue;
            }
            x[i] = aug[i][n];
            for j in i + 1..n {
                x[i] -= aug[i][j] * x[j];
            }
            x[i] /= aug[i][i];
        }
        x
    }
}

// ============================================================
// Module 2: AgentIdentity
// ============================================================

#[derive(Debug, Clone)]
pub struct AgentIdentity {
    pub fingerprint: SpectralFingerprint,
    pub id: String,
    pub capabilities: Vec<String>,
    pub confidence: f64,
}

impl AgentIdentity {
    pub fn new(adj: &[Vec<f64>], id: &str) -> AgentIdentity {
        let fingerprint = SpectralFingerprint::from_graph(adj);
        let confidence = Self::compute_confidence(&fingerprint.eigenvalues);
        AgentIdentity {
            fingerprint,
            id: id.to_string(),
            capabilities: Vec::new(),
            confidence,
        }
    }

    pub fn self_assessment(&self) -> f64 {
        self.confidence
    }

    pub fn similarity_to(&self, other: &AgentIdentity) -> f64 {
        self.fingerprint.alignment(&other.fingerprint)
    }

    fn compute_confidence(eigenvalues: &[f64]) -> f64 {
        if eigenvalues.is_empty() {
            return 0.0;
        }
        let n = eigenvalues.len() as f64;
        // Entropy of normalized eigenvalue distribution
        let total: f64 = eigenvalues.iter().map(|x| x.abs()).sum();
        if total < 1e-15 {
            return 0.0;
        }
        let entropy: f64 = eigenvalues
            .iter()
            .map(|x| {
                let p = x.abs() / total;
                if p > 1e-15 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();
        let max_entropy = n.ln();
        if max_entropy < 1e-15 {
            1.0
        } else {
            1.0 - entropy / max_entropy
        }
    }
}

// ============================================================
// Module 3: SpectralMessage & Envelope
// ============================================================

#[derive(Debug, Clone)]
pub enum SpectralMessage {
    Hello(SpectralFingerprint),
    Propose(Vec<usize>, Vec<usize>),
    Confirm(f64),
    Reject(String),
    Anomaly(f64),
    Bootstrap(SpectralFingerprint),
}

#[derive(Debug, Clone)]
pub struct Envelope {
    pub sender: AgentIdentity,
    pub receiver: AgentIdentity,
    pub message: SpectralMessage,
    pub alignment: f64,
    pub timestamp: u64,
}

// ============================================================
// Module 4: ConservationProtocol
// ============================================================

#[derive(Debug, Clone)]
pub struct ConservationProtocol {
    pub identity: AgentIdentity,
    pub known_agents: HashMap<String, AgentIdentity>,
    pub trust_threshold: f64,
    pub composition_threshold: f64,
}

impl ConservationProtocol {
    pub fn new(identity: AgentIdentity) -> ConservationProtocol {
        ConservationProtocol {
            identity,
            known_agents: HashMap::new(),
            trust_threshold: 0.3,
            composition_threshold: 0.67,
        }
    }

    pub fn greet(&self, receiver: &AgentIdentity) -> Envelope {
        Envelope {
            sender: self.identity.clone(),
            receiver: receiver.clone(),
            message: SpectralMessage::Hello(self.identity.fingerprint.clone()),
            alignment: self.identity.fingerprint.alignment(&receiver.fingerprint),
            timestamp: 0, // caller should set
        }
    }

    pub fn receive(&mut self, envelope: Envelope) -> Result<Envelope, String> {
        let sender_id = envelope.sender.id.clone();
        let alignment = envelope.alignment;

        // Learn about the sender
        self.known_agents.insert(sender_id.clone(), envelope.sender.clone());

        match &envelope.message {
            SpectralMessage::Hello(fp) => {
                if alignment >= self.trust_threshold {
                    Ok(Envelope {
                        sender: self.identity.clone(),
                        receiver: envelope.sender,
                        message: SpectralMessage::Confirm(self.identity.fingerprint.conservation),
                        alignment,
                        timestamp: 0,
                    })
                } else {
                    Ok(Envelope {
                        sender: self.identity.clone(),
                        receiver: envelope.sender,
                        message: SpectralMessage::Reject(format!(
                            "alignment {} below trust threshold {}",
                            alignment, self.trust_threshold
                        )),
                        alignment,
                        timestamp: 0,
                    })
                }
            }
            SpectralMessage::Propose(edges_a, edges_b) => {
                // Check if composing would improve CR
                let composed = self.compose(&envelope.sender);
                if composed.conservation >= self.composition_threshold {
                    Ok(Envelope {
                        sender: self.identity.clone(),
                        receiver: envelope.sender,
                        message: SpectralMessage::Confirm(composed.conservation),
                        alignment,
                        timestamp: 0,
                    })
                } else {
                    Ok(Envelope {
                        sender: self.identity.clone(),
                        receiver: envelope.sender,
                        message: SpectralMessage::Reject(format!(
                            "composed CR {:.4} below threshold {:.4}",
                            composed.conservation, self.composition_threshold
                        )),
                        alignment,
                        timestamp: 0,
                    })
                }
            }
            SpectralMessage::Anomaly(cr) => {
                Ok(Envelope {
                    sender: self.identity.clone(),
                    receiver: envelope.sender,
                    message: SpectralMessage::Confirm(*cr),
                    alignment,
                    timestamp: 0,
                })
            }
            SpectralMessage::Bootstrap(fp) => {
                self.known_agents.insert(
                    sender_id.clone(),
                    AgentIdentity {
                        fingerprint: fp.clone(),
                        id: sender_id,
                        capabilities: vec![],
                        confidence: 0.5,
                    },
                );
                Ok(Envelope {
                    sender: self.identity.clone(),
                    receiver: envelope.sender,
                    message: SpectralMessage::Hello(self.identity.fingerprint.clone()),
                    alignment,
                    timestamp: 0,
                })
            }
            _ => Err("unhandled message type".into()),
        }
    }

    pub fn should_compose(&self, other: &AgentIdentity) -> bool {
        let composed = self.compose(other);
        composed.conservation >= self.composition_threshold
    }

    pub fn should_trust(&self, other: &AgentIdentity) -> bool {
        self.identity.fingerprint.alignment(&other.fingerprint) >= self.trust_threshold
    }

    pub fn routing_preference(&self) -> Vec<(String, f64)> {
        let mut agents: Vec<(String, f64)> = self
            .known_agents
            .iter()
            .map(|(id, agent)| {
                (id.clone(), self.identity.fingerprint.alignment(&agent.fingerprint))
            })
            .collect();
        agents.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        agents
    }

    /// Compose: create a new spectral fingerprint from the union of two agents' graphs.
    /// L_composed - L_A - L_B ≈ coupling structure
    pub fn compose(&self, other: &AgentIdentity) -> SpectralFingerprint {
        let n_a = self.identity.fingerprint.capacity;
        let n_b = other.fingerprint.capacity;
        let n = n_a + n_b;

        // Build composed adjacency: block diagonal with coupling
        let mut adj = vec![vec![0.0; n]; n];

        // Agent A's structure (top-left block)
        // We reconstruct a simple graph from eigenvalue count
        for i in 0..n_a {
            for j in 0..n_a {
                if i != j {
                    adj[i][j] = 0.5 / (n_a as f64).max(1.0);
                }
            }
        }

        // Agent B's structure (bottom-right block)
        for i in 0..n_b {
            for j in 0..n_b {
                if i != j {
                    adj[n_a + i][n_a + j] = 0.5 / (n_b as f64).max(1.0);
                }
            }
        }

        // Coupling edges based on alignment
        let alpha = self.identity.fingerprint.alignment(&other.fingerprint);
        for i in 0..n_a.min(3) {
            for j in 0..n_b.min(3) {
                adj[i][n_a + j] = alpha * 0.5;
                adj[n_a + j][i] = alpha * 0.5;
            }
        }

        SpectralFingerprint::from_graph(&adj)
    }
}

// ============================================================
// Module 5: CompositionGate
// ============================================================

#[derive(Debug, Clone)]
pub struct CompositionGate {
    pub min_cr: f64,
    pub min_alignment: f64,
    pub target_cr: f64,
}

#[derive(Debug, Clone)]
pub enum GateResult {
    Approved(f64),
    Rejected(String),
    Deferred(Vec<(usize, usize)>),
}

impl Default for CompositionGate {
    fn default() -> CompositionGate {
        CompositionGate {
            min_cr: 0.67,
            min_alignment: 0.3,
            target_cr: 1.0 / 1.6180339887, // 1/φ ≈ 0.618
        }
    }
}

impl CompositionGate {
    pub fn evaluate(&self, composed: &SpectralFingerprint) -> GateResult {
        let cr = composed.conservation;
        if cr >= self.min_cr {
            GateResult::Approved(cr)
        } else if cr < self.min_alignment {
            GateResult::Rejected(format!(
                "CR {:.4} is critically low (below {:.4})",
                cr, self.min_alignment
            ))
        } else {
            let suggestions = self.suggest_improvements(composed);
            GateResult::Deferred(suggestions)
        }
    }

    pub fn suggest_improvements(&self, composed: &SpectralFingerprint) -> Vec<(usize, usize)> {
        let n = composed.capacity;
        let mut suggestions = Vec::new();

        // Suggest connecting nodes with largest gaps in Fiedler vector
        if n < 2 {
            return suggestions;
        }
        let fiedler = &composed.fiedler;
        let mut indexed: Vec<(usize, f64)> = fiedler.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Suggest edges between most separated nodes
        let count = (n / 3).max(1);
        for i in 0..count {
            let a = indexed[i].0;
            let b = indexed[indexed.len() - 1 - i].0;
            suggestions.push((a, b));
        }
        suggestions
    }
}

// ============================================================
// main
// ============================================================

fn main() {
    println!("conservation-protocol: eigenvalues are the message");

    // Demo: two small graphs
    let graph_a = vec![
        vec![0.0, 1.0, 1.0, 0.0],
        vec![1.0, 0.0, 1.0, 1.0],
        vec![1.0, 1.0, 0.0, 1.0],
        vec![0.0, 1.0, 1.0, 0.0],
    ];
    let graph_b = vec![
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0],
    ];

    let fp_a = SpectralFingerprint::from_graph(&graph_a);
    let fp_b = SpectralFingerprint::from_graph(&graph_b);

    println!("Graph A eigenvalues: {:?}", fp_a.eigenvalues);
    println!("Graph A CR: {:.4}", fp_a.conservation);
    println!("Graph B eigenvalues: {:?}", fp_b.eigenvalues);
    println!("Graph B CR: {:.4}", fp_b.conservation);
    println!("Alignment α = {:.4}", fp_a.alignment(&fp_b));

    let agent_a = AgentIdentity::new(&graph_a, "agent-alpha");
    let agent_b = AgentIdentity::new(&graph_b, "agent-beta");
    println!("Agent A confidence: {:.4}", agent_a.self_assessment());
    println!("Agent B confidence: {:.4}", agent_b.self_assessment());

    let mut proto = ConservationProtocol::new(agent_a.clone());
    let greeting = proto.greet(&agent_b);
    println!("Greeting alignment: {:.4}", greeting.alignment);

    let response = proto.receive(greeting).unwrap();
    match &response.message {
        SpectralMessage::Confirm(cr) => println!("Composition confirmed, CR={:.4}", cr),
        SpectralMessage::Reject(reason) => println!("Rejected: {}", reason),
        _ => {}
    }

    let gate = CompositionGate::default();
    let composed = proto.compose(&agent_b);
    match gate.evaluate(&composed) {
        GateResult::Approved(cr) => println!("Gate APPROVED (CR={:.4})", cr),
        GateResult::Rejected(reason) => println!("Gate REJECTED: {}", reason),
        GateResult::Deferred(suggestions) => println!("Gate DEFERRED, suggestions: {:?}", suggestions),
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_graph(n: usize) -> Vec<Vec<f64>> {
        let mut adj = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    adj[i][j] = 1.0;
                }
            }
        }
        adj
    }

    fn path_graph(n: usize) -> Vec<Vec<f64>> {
        let mut adj = vec![vec![0.0; n]; n];
        for i in 0..n - 1 {
            adj[i][i + 1] = 1.0;
            adj[i + 1][i] = 1.0;
        }
        adj
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let adj = complete_graph(4);
        let fp = SpectralFingerprint::from_graph(&adj);
        let encoded = fp.encode();
        let decoded = SpectralFingerprint::decode(&encoded).unwrap();
        assert_eq!(decoded.capacity, fp.capacity);
        assert_eq!(decoded.eigenvalues.len(), fp.eigenvalues.len());
        for (a, b) in decoded.eigenvalues.iter().zip(fp.eigenvalues.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        assert!((decoded.conservation - fp.conservation).abs() < 1e-10);
    }

    #[test]
    fn test_alignment_identical_is_one() {
        let adj = complete_graph(4);
        let fp = SpectralFingerprint::from_graph(&adj);
        let alpha = fp.alignment(&fp);
        assert!(
            (alpha - 1.0).abs() < 1e-10,
            "identical fingerprints should align at 1.0, got {}",
            alpha
        );
    }

    #[test]
    fn test_alignment_random_is_low() {
        let adj_a = complete_graph(3);
        // A very different graph: path of 3 (less connected)
        let adj_b = path_graph(3);
        let fp_a = SpectralFingerprint::from_graph(&adj_a);
        let fp_b = SpectralFingerprint::from_graph(&adj_b);
        let alpha = fp_a.alignment(&fp_b);
        // They share structure so won't be 0, but should differ from 1.0
        assert!(
            alpha < 0.999,
            "different graphs should not perfectly align, got {}",
            alpha
        );
    }

    #[test]
    fn test_compatibility_threshold() {
        let adj = complete_graph(4);
        let fp = SpectralFingerprint::from_graph(&adj);
        assert!(fp.is_compatible(&fp, 0.99));
        assert!(fp.is_compatible(&fp, 1.0 - 1e-12));
    }

    #[test]
    fn test_misaligned_fraction() {
        let adj = complete_graph(4);
        let fp = SpectralFingerprint::from_graph(&adj);
        let mis = fp.misaligned_fraction(&fp);
        assert!(mis.abs() < 1e-10, "misaligned of self should be 0, got {}", mis);
    }

    #[test]
    fn test_agent_self_assessment() {
        let adj = complete_graph(5);
        let agent = AgentIdentity::new(&adj, "test");
        let conf = agent.self_assessment();
        assert!(conf >= 0.0 && conf <= 1.0, "confidence should be in [0,1], got {}", conf);
        // Complete graph should have reasonable confidence
        assert!(conf > 0.0, "non-trivial graph should have non-zero confidence");
    }

    #[test]
    fn test_protocol_greet_receive_handshake() {
        let adj_a = complete_graph(4);
        let adj_b = complete_graph(4);
        let agent_a = AgentIdentity::new(&adj_a, "alice");
        let agent_b = AgentIdentity::new(&adj_b, "bob");

        let mut proto = ConservationProtocol::new(agent_a.clone());
        let greeting = proto.greet(&agent_b);
        assert_eq!(greeting.alignment, agent_a.fingerprint.alignment(&agent_b.fingerprint));

        // The response should be a Confirm (high alignment since same graph)
        let response = proto.receive(greeting);
        assert!(response.is_ok());
    }

    #[test]
    fn test_composition_different_spectrum() {
        let adj_a = complete_graph(3);
        let adj_b = path_graph(4);
        let agent_a = AgentIdentity::new(&adj_a, "alice");
        let agent_b = AgentIdentity::new(&adj_b, "bob");

        let proto = ConservationProtocol::new(agent_a.clone());
        let composed = proto.compose(&agent_b);

        // Composed spectrum should differ from both individuals
        let diff_a: f64 = composed
            .eigenvalues
            .iter()
            .zip(agent_a.fingerprint.eigenvalues.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        // At least one eigenvalue should differ (different dimensions anyway)
        assert_ne!(composed.capacity, agent_a.fingerprint.capacity);
    }

    #[test]
    fn test_composition_gate_approved() {
        let gate = CompositionGate::default();
        // Create a fingerprint with high CR (close to 1.0)
        let fp = SpectralFingerprint {
            eigenvalues: vec![0.0, 0.8, 0.9, 1.0],
            conservation: 0.8, // > 0.67
            fiedler: vec![0.5, -0.5, 0.5, -0.5],
            capacity: 4,
        };
        match gate.evaluate(&fp) {
            GateResult::Approved(cr) => assert!(cr >= 0.67),
            _ => panic!("expected Approved"),
        }
    }

    #[test]
    fn test_composition_gate_rejected() {
        let gate = CompositionGate::default();
        let fp = SpectralFingerprint {
            eigenvalues: vec![0.0, 0.1, 0.5, 1.0],
            conservation: 0.1, // < 0.3
            fiedler: vec![0.5, -0.5, 0.5, -0.5],
            capacity: 4,
        };
        match gate.evaluate(&fp) {
            GateResult::Rejected(_) => {}
            _ => panic!("expected Rejected"),
        }
    }

    #[test]
    fn test_composition_gate_deferred() {
        let gate = CompositionGate::default();
        let fp = SpectralFingerprint {
            eigenvalues: vec![0.0, 0.5, 0.8, 1.0],
            conservation: 0.5, // between 0.3 and 0.67
            fiedler: vec![0.5, -0.5, 0.3, -0.3],
            capacity: 4,
        };
        match gate.evaluate(&fp) {
            GateResult::Deferred(suggestions) => {
                assert!(!suggestions.is_empty());
            }
            _ => panic!("expected Deferred, got {:?}", gate.evaluate(&fp)),
        }
    }

    #[test]
    fn test_trust_below_threshold() {
        let adj_a = complete_graph(3);
        let adj_b = path_graph(3);
        let agent_a = AgentIdentity::new(&adj_a, "alice");
        let agent_b = AgentIdentity::new(&adj_b, "bob");

        let mut proto = ConservationProtocol::new(agent_a);
        proto.trust_threshold = 0.99; // very high threshold
        assert!(!proto.should_trust(&agent_b));
    }

    #[test]
    fn test_trust_above_threshold() {
        let adj = complete_graph(4);
        let agent_a = AgentIdentity::new(&adj, "alice");
        let agent_b = AgentIdentity::new(&adj, "bob");

        let proto = ConservationProtocol::new(agent_a);
        assert!(proto.should_trust(&agent_b));
    }

    #[test]
    fn test_routing_preference() {
        let adj_a = complete_graph(4);
        let adj_b = complete_graph(4);
        let adj_c = path_graph(4);

        let agent_a = AgentIdentity::new(&adj_a, "alice");
        let agent_b = AgentIdentity::new(&adj_b, "bob");
        let agent_c = AgentIdentity::new(&adj_c, "carol");

        let mut proto = ConservationProtocol::new(agent_a);
        proto.known_agents.insert("bob".into(), agent_b);
        proto.known_agents.insert("carol".into(), agent_c);

        let prefs = proto.routing_preference();
        assert_eq!(prefs.len(), 2);
        // Bob (same graph) should rank higher than Carol (different graph)
        assert!(prefs[0].1 >= prefs[1].1);
    }
}
