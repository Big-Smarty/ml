//! Pairwise matrix factorization and top-k ranking metrics for implicit feedback.
const USERS: usize = 4;
const ITEMS: usize = 6;
const FACTORS: usize = 2;

const TRAIN_POS: [[usize; 2]; USERS] = [[0, 1], [0, 2], [3, 4], [3, 5]];
const HELD_OUT: [usize; USERS] = [2, 1, 5, 4];
const EXPLICIT_NEG: [[usize; 3]; USERS] = [[3, 4, 5], [3, 4, 5], [0, 1, 2], [0, 1, 2]];

fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[derive(Clone)]
struct MatrixFactorization {
    users: [[f64; FACTORS]; USERS],
    items: [[f64; FACTORS]; ITEMS],
}

impl MatrixFactorization {
    fn new() -> Self {
        let mut users = [[0.0; FACTORS]; USERS];
        let mut items = [[0.0; FACTORS]; ITEMS];
        for (u, user) in users.iter_mut().enumerate() {
            for (f, value) in user.iter_mut().enumerate() {
                *value = ((u * 7 + f * 3 + 1) as f64).sin() * 0.15;
            }
        }
        for (i, item) in items.iter_mut().enumerate() {
            for (f, value) in item.iter_mut().enumerate() {
                *value = ((i * 11 + f * 5 + 2) as f64).cos() * 0.15;
            }
        }
        Self { users, items }
    }
    fn score(&self, user: usize, item: usize) -> f64 {
        self.users[user]
            .iter()
            .zip(self.items[item])
            .map(|(a, b)| a * b)
            .sum()
    }
    fn triple_loss(
        &self,
        user: usize,
        positive: usize,
        negative: usize,
        regularization: f64,
    ) -> f64 {
        let margin = self.score(user, positive) - self.score(user, negative);
        let penalty = self.users[user]
            .iter()
            .chain(&self.items[positive])
            .chain(&self.items[negative])
            .map(|v| v * v)
            .sum::<f64>();
        softplus(-margin) + regularization * penalty
    }
    fn train(&mut self, epochs: usize, rate: f64, regularization: f64) {
        for _ in 0..epochs {
            for user in 0..USERS {
                for (p_at, &positive) in TRAIN_POS[user].iter().enumerate() {
                    let negative = EXPLICIT_NEG[user][p_at % EXPLICIT_NEG[user].len()];
                    let old_u = self.users[user];
                    let old_p = self.items[positive];
                    let old_n = self.items[negative];
                    let q = sigmoid(-(self.score(user, positive) - self.score(user, negative)));
                    for f in 0..FACTORS {
                        let gu = -q * (old_p[f] - old_n[f]) + 2.0 * regularization * old_u[f];
                        let gp = -q * old_u[f] + 2.0 * regularization * old_p[f];
                        let gn = q * old_u[f] + 2.0 * regularization * old_n[f];
                        self.users[user][f] -= rate * gu;
                        self.items[positive][f] -= rate * gp;
                        self.items[negative][f] -= rate * gn;
                    }
                }
            }
        }
    }
    fn mean_training_loss(&self, regularization: f64) -> f64 {
        let mut total = 0.0;
        let mut n = 0;
        for user in 0..USERS {
            for (p_at, &positive) in TRAIN_POS[user].iter().enumerate() {
                total += self.triple_loss(user, positive, EXPLICIT_NEG[user][p_at], regularization);
                n += 1;
            }
        }
        total / n as f64
    }
    fn ranked_candidates(&self, user: usize) -> Vec<usize> {
        let mut candidates = vec![HELD_OUT[user]];
        candidates.extend(EXPLICIT_NEG[user]);
        candidates.sort_by(|&a, &b| {
            self.score(user, b)
                .total_cmp(&self.score(user, a))
                .then_with(|| a.cmp(&b))
        });
        candidates
    }
}

fn ranking_metrics(model: &MatrixFactorization, k: usize) -> (f64, f64, f64) {
    let k = k.min(1 + EXPLICIT_NEG[0].len());
    let mut recall = 0.0;
    let mut precision = 0.0;
    let mut ndcg = 0.0;
    for (user, &held_out) in HELD_OUT.iter().enumerate() {
        let rank = model
            .ranked_candidates(user)
            .iter()
            .position(|&i| i == held_out)
            .unwrap();
        if rank < k {
            recall += 1.0;
            precision += 1.0 / k as f64;
            ndcg += 1.0 / (rank as f64 + 2.0).log2();
        }
    }
    (
        recall / USERS as f64,
        precision / USERS as f64,
        ndcg / USERS as f64,
    )
}

fn main() {
    let mut model = MatrixFactorization::new();
    let before_loss = model.mean_training_loss(0.002);
    let before = ranking_metrics(&model, 2);
    model.train(800, 0.04, 0.002);
    let after_loss = model.mean_training_loss(0.002);
    let after = ranking_metrics(&model, 2);
    println!("pairwise training loss {before_loss:.5} -> {after_loss:.5}");
    println!(
        "held-out Recall@2 {:.1}% -> {:.1}%, Precision@2 {:.1}% -> {:.1}%, nDCG@2 {:.1}% -> {:.1}%",
        before.0 * 100.0,
        after.0 * 100.0,
        before.1 * 100.0,
        after.1 * 100.0,
        before.2 * 100.0,
        after.2 * 100.0
    );
    for (user, &held_out) in HELD_OUT.iter().enumerate() {
        println!(
            "user {user}: held-out item {}, ranked candidates {:?}",
            held_out,
            model.ranked_candidates(user)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pairwise_gradient_matches_central_difference() {
        let mut model = MatrixFactorization::new();
        let (u, p, n, reg) = (0, 0, 3, 0.002);
        let q = sigmoid(-(model.score(u, p) - model.score(u, n)));
        let analytic = -q * (model.items[p][0] - model.items[n][0]) + 2.0 * reg * model.users[u][0];
        let h = 1e-5;
        model.users[u][0] += h;
        let plus = model.triple_loss(u, p, n, reg);
        model.users[u][0] -= 2.0 * h;
        let minus = model.triple_loss(u, p, n, reg);
        let numeric = (plus - minus) / (2.0 * h);
        assert!((analytic - numeric).abs() < 1e-6 + 1e-4 * numeric.abs());
    }
    #[test]
    fn training_improves_loss_and_held_out_ranking() {
        let mut model = MatrixFactorization::new();
        let before_loss = model.mean_training_loss(0.002);
        model.train(800, 0.04, 0.002);
        let metrics = ranking_metrics(&model, 2);
        assert!(model.mean_training_loss(0.002) < before_loss * 0.25);
        assert!(metrics.0 >= 0.75 && metrics.2 >= 0.75, "{metrics:?}");
    }
    #[test]
    fn stable_softplus_handles_extreme_values() {
        assert!(softplus(1000.0).is_finite());
        assert!(softplus(-1000.0).is_finite());
    }
}
