//! Compute all three gradients from the SAME old vectors before applying any updates.
//! Binary nDCG discounts actual rank and divides by the best possible ordering at k.
use crate::recommendation::{self, sigmoid, Core, MatrixFactorization, FACTORS};
pub(crate) const CORE: Core = Core {
    update,
    query_metrics,
};
pub(crate) fn update(
    model: &mut MatrixFactorization,
    user: usize,
    positive: usize,
    negative: usize,
    learning_rate: f64,
    regularization: f64,
) {
    let old_user = model.users[user];
    let old_positive = model.items[positive];
    let old_negative = model.items[negative];
    let ordering_signal = sigmoid(-(model.score(user, positive) - model.score(user, negative)));
    for factor in 0..FACTORS {
        let user_gradient = -ordering_signal * (old_positive[factor] - old_negative[factor])
            + 2.0 * regularization * old_user[factor];
        let positive_gradient =
            -ordering_signal * old_user[factor] + 2.0 * regularization * old_positive[factor];
        let negative_gradient =
            ordering_signal * old_user[factor] + 2.0 * regularization * old_negative[factor];
        model.users[user][factor] -= learning_rate * user_gradient;
        model.items[positive][factor] -= learning_rate * positive_gradient;
        model.items[negative][factor] -= learning_rate * negative_gradient;
    }
}
pub(crate) fn query_metrics(order: &[usize], relevant: &[usize], k: usize) -> (f64, f64, f64) {
    let k = k.min(order.len());
    if k == 0 || relevant.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut hits = 0;
    let mut dcg = 0.0;
    for (rank, item) in order.iter().take(k).enumerate() {
        if relevant.contains(item) {
            hits += 1;
            dcg += 1.0 / (rank as f64 + 2.0).log2();
        }
    }
    let ideal: f64 = (0..k.min(relevant.len()))
        .map(|rank| 1.0 / (rank as f64 + 2.0).log2())
        .sum();
    (
        hits as f64 / relevant.len() as f64,
        hits as f64 / k as f64,
        dcg / ideal,
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    recommendation::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    recommendation::check(CORE)
}
