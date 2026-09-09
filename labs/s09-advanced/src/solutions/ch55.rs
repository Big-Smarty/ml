//! Worked solution: Map output/input coordinates instead of reshaping storage. Keep an independent input-major runtime forward and gradient implementation, compare every coordinate, and update both weight and bias blocks before comparing new predictions. Actual artifact bytes have a fixed validated format, separate from the optional Burn reference.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Map coordinates, not just bytes. Input-major runtime rows contain one weight for each output.
include!("../common/ch55.rs");
fn map_weights(weights: &[[f64; 3]; 2]) -> [[f64; 2]; 3] {
    std::array::from_fn(|input| std::array::from_fn(|output| weights[output][input]))
}
fn verify_parity(actual: &[f64], expected: &[f64]) -> Result<(), String> {
    crate::ensure(actual.len() == expected.len(), "parity lengths differ")?;
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        crate::ensure(crate::close(a,b),&format!("coordinate {index}: runtime {a}, scratch {b}; inspect layout, bias and reduction before changing tolerance"))?;
    }
    Ok(())
}
#[test]
fn full_port_and_artifact_check() {
    check().expect("non-square port parity");
}

impl Runtime {
    // Independent runtime loop: input-major accumulation, not the scratch forward function.
    fn forward(&self, x: [f64; 3]) -> [f64; 2] {
        let mut out = self.bias;
        for (row, input) in self.weights.iter().zip(x) {
            for (o, w) in out.iter_mut().zip(row) {
                *o += input * w;
            }
        }
        out
    }
    fn gradient(&self, x: [f64; 3], target: [f64; 2]) -> Self {
        let mut error = self.forward(x);
        for (e, t) in error.iter_mut().zip(target) {
            *e -= t;
        }
        Self {
            weights: std::array::from_fn(|i| error.map(|e| e * x[i])),
            bias: error,
        }
    }
    fn update(&mut self, gradient: &Self, rate: f64) {
        for (p, g) in self
            .weights
            .iter_mut()
            .flatten()
            .chain(&mut self.bias)
            .zip(gradient.weights.iter().flatten().chain(&gradient.bias))
        {
            *p -= rate * g;
        }
    }
}

fn compare(model: &Dense) -> Result<(), String> {
    let runtime = Runtime {
        weights: map_weights(&model.weights),
        bias: model.bias,
    };
    for x in [INPUT, [-1.0, 3.0, 0.25], [0.0; 3]] {
        verify_parity(&runtime.forward(x), &model.forward(x))?;
    }
    let expected = model.gradient(INPUT, TARGET);
    let actual = runtime.gradient(INPUT, TARGET);
    verify_parity(
        &actual.weights.into_iter().flatten().collect::<Vec<_>>(),
        &map_weights(&expected.weights)
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
    )?;
    verify_parity(&actual.bias, &expected.bias)?;
    let mut updated = runtime.clone();
    updated.update(&actual, 0.1);
    let mut scratch = model.clone();
    for (p, g) in scratch
        .weights
        .iter_mut()
        .flatten()
        .chain(&mut scratch.bias)
        .zip(expected.weights.iter().flatten().chain(&expected.bias))
    {
        *p -= 0.1 * g;
    }
    verify_parity(&updated.forward(INPUT), &scratch.forward(INPUT))
}
