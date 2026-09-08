// TODO: add the new path's contribution to the existing gradient.
fn add_gradient(total: &mut f64, contribution: f64) {
    let _ = (total, contribution);
    todo!("add the gradient contribution")
}
fn main() {
    let mut g = 0.0;
    add_gradient(&mut g, 2.0);
    println!("{g}");
}
#[test]
fn shared_use() {
    let mut g = 0.0;
    add_gradient(&mut g, 2.0);
    assert_eq!(g, 2.0);
    add_gradient(&mut g, 3.0);
    assert_eq!(g, 5.0);
}
