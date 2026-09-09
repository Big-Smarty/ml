//! Working item-only factor learner; user features remain fixed.
//! Train both tables with a simultaneous pairwise update, and replace nCG with nDCG.
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
        let positive_gradient =
            -ordering_signal * old_user[factor] + 2.0 * regularization * old_positive[factor];
        let negative_gradient =
            ordering_signal * old_user[factor] + 2.0 * regularization * old_negative[factor];
        // Fixed user features: the baseline trains item vectors only.
        model.items[positive][factor] -= learning_rate * positive_gradient;
        model.items[negative][factor] -= learning_rate * negative_gradient;
    }
}
pub(crate) fn query_metrics(order: &[usize], relevant: &[usize], k: usize) -> (f64, f64, f64) {
    let k = k.min(order.len());
    if k == 0 || relevant.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let hits = order
        .iter()
        .take(k)
        .filter(|item| relevant.contains(item))
        .count() as f64;
    // Un-discounted normalized cumulative gain: does not distinguish first from third.
    (
        hits / relevant.len() as f64,
        hits / k as f64,
        hits / k.min(relevant.len()) as f64,
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    recommendation::run(CORE, args)
}
pub fn check() -> Result<(), String> {
    recommendation::check(CORE)
}
