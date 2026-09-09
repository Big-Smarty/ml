//! Working baseline reinterprets the six stored values as runtime rows. Build a real layout adapter and coordinate-wise parity check.
include!("common/ch55.rs");
fn map_weights(weights: &[[f64; 3]; 2]) -> [[f64; 2]; 3] {
    // Baseline: reshape the bytes. A port must instead preserve (output,input) coordinates.
    [
        [weights[0][0], weights[0][1]],
        [weights[0][2], weights[1][0]],
        [weights[1][1], weights[1][2]],
    ]
}
// Baseline compares only the summed outputs; opposite coordinate errors can cancel.
fn verify_parity(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    crate::ensure(actual.len() == expected.len(), "parity lengths differ")?;
    crate::ensure(
        crate::close(actual.iter().sum(), expected.iter().sum()),
        "aggregate outputs differ; implement per-coordinate diagnostics",
    )
}
impl Runtime {
    // Working inference path accumulates by input row. Keep this independent of Dense::forward.
    fn forward(&self, x: [f64; 3]) -> [f64; 2] {
        let mut out = self.bias;
        for (row, input) in self.weights.iter().zip(x) {
            for (o, w) in out.iter_mut().zip(row) {
                *o += input * w;
            }
        }
        out
    }
    // An inference-only baseline does not train its parameters yet.
    fn gradient(&self, _x: [f64; 3], _target: [f64; 2]) -> Self {
        Self {
            weights: [[0.0; 2]; 3],
            bias: [0.0; 2],
        }
    }
    fn update(&mut self, _gradient: &Self, _rate: f64) {}
}
fn compare(model: &Dense) -> Result<(), String> {
    let runtime = Runtime {
        weights: map_weights(&model.weights),
        bias: model.bias,
    };
    // Baseline asks only whether one input's aggregate output agrees.
    verify_parity(&runtime.forward(INPUT), &model.forward(INPUT))
}
