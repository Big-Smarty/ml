fn add_gradient(total: &mut f64, contribution: f64) {
    let _ = (total, contribution);
    todo!("shared graph paths must add their gradient contributions")
}

fn main() {
    let _guided: fn(&mut f64, f64) = add_gradient;
    let x = 3.0_f64;
    println!("prior checkpoint: x*x+x = {}", x * x + x);
}

#[test]
fn shared_paths_add() {
    let mut g = 0.0;
    add_gradient(&mut g, 2.0);
    assert_eq!(g, 2.0);
    add_gradient(&mut g, 3.0);
    assert_eq!(g, 5.0);
}
