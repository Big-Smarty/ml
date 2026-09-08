fn dot(weights: &[f64], features: &[f64]) -> Result<f64, &'static str> {
    let _ = (weights, features);
    todo!("reject unequal lengths, then multiply matching entries and sum")
}

fn main() -> Result<(), &'static str> {
    let _guided_todo: fn(&[f64], &[f64]) -> Result<f64, &'static str> = dot;
    println!(
        "prior checkpoint: one-input prediction is {}",
        2.0 * 3.0 + 0.5
    );
    println!("Run cargo test to implement the new dot-product TODO.");
    Ok(())
}

#[test]
fn dot_uses_every_feature_once() -> Result<(), &'static str> {
    assert_eq!(dot(&[2.0, -1.0], &[3.0, 4.0])?, 2.0);
    assert!(dot(&[1.0], &[1.0, 2.0]).is_err());
    Ok(())
}
