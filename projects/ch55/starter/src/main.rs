fn forward(weight: [[f32; 2]; 2], bias: [f32; 2], input: [f32; 2]) -> [f32; 2] {
    std::array::from_fn(|o| weight[o][0] * input[0] + weight[o][1] * input[1] + bias[o])
}

fn mean_squared_loss(output: [f32; 2], target: [f32; 2]) -> f32 {
    output
        .iter()
        .zip(target)
        .map(|(prediction, target)| (prediction - target).powi(2))
        .sum::<f32>()
        / 2.0
}

fn main() {
    let output = forward([[0.2, -0.4], [0.7, 0.1]], [0.05, -0.2], [1.5, -2.0]);
    println!(
        "output {output:?}; MSE {:.4}",
        mean_squared_loss(output, [1.0, -0.5])
    );
}

#[cfg(test)]
fn weight_gradient(error: f32, input: f32) -> f32 {
    let _ = (error, input);
    // TODO: multiply the output derivative by the input, for mean over two outputs.
    todo!("multiply the output derivative by input[0]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn port_the_gradient_rule() {
        let output = forward([[0.2, -0.4], [0.7, 0.1]], [0.05, -0.2], [1.5, -2.0]);
        let error = output[0] - 1.0;
        // TODO: return dloss/dweight[0][0] for mean squared error over two outputs.
        let gradient = weight_gradient(error, 1.5);
        assert!((gradient - 0.225).abs() < 1e-6, "error was {error}");
    }
}
