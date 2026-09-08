fn accumulate(existing: &mut f64, contribution: f64) {
    let _ = (existing, contribution);
    todo!("shared graph paths must add their gradient contributions")
}

fn main() {
    let _guided: fn(&mut f64, f64) = accumulate;
    let x = 3.0_f64;
    println!("prior checkpoint: x*x+x = {}", x * x + x);
}

#[test]
fn shared_paths_add() {
    let mut g = 0.0;
    accumulate(&mut g, 2.0);
    assert_eq!(g, 2.0);
    accumulate(&mut g, 3.0);
    assert_eq!(g, 5.0);
}
