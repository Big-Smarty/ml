type Vec2 = [f64; 2];
type Mat2 = [[f64; 2]; 2];

#[cfg(test)]
// One normalized loop body from the reference's repeated power_iteration_from.
fn power_step(a: Mat2, v: Vec2) -> Vec2 {
    let _ = (a, v);
    // TODO: multiply a by v, then divide the result by its Euclidean norm.
    todo!("implement one normalized power-iteration step")
}

fn main() {
    let matrix: Mat2 = [[2.0, 1.0], [1.0, 2.0]];
    let vector: Vec2 = [1.0, 0.0];
    let first_product: Vec2 = matrix.map(|row| row[0] * vector[0] + row[1] * vector[1]);
    println!("known matrix-vector product: {first_product:?}; now normalize it in power_step");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn first_step_matches_hand_calculation() {
        let v = power_step([[2.0, 1.0], [1.0, 2.0]], [1.0, 0.0]);
        assert!((v[0] - 2.0 / 5.0_f64.sqrt()).abs() < 1e-12);
        assert!((v[1] - 1.0 / 5.0_f64.sqrt()).abs() < 1e-12);
    }
}
