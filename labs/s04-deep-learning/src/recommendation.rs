//! Pairwise matrix factorization and top-k ranking metrics for implicit feedback.
const USERS: usize = 4;
const ITEMS: usize = 6;
pub(crate) const FACTORS: usize = 2;

type QueryMetrics = fn(&[usize], &[usize], usize) -> (f64, f64, f64);
#[derive(Clone, Copy)]
pub(crate) struct Core {
    pub update: fn(&mut MatrixFactorization, usize, usize, usize, f64, f64),
    pub query_metrics: QueryMetrics,
}
// Each triple is (user ID, preferred item ID, less-preferred item ID).
type TrainingTriple = (usize, usize, usize);
const TRAIN_DATA: [TrainingTriple; 8] = [
    (0, 0, 3),
    (0, 1, 4),
    (1, 0, 3),
    (1, 2, 4),
    (2, 3, 0),
    (2, 4, 1),
    (3, 3, 0),
    (3, 5, 1),
];
const HELD_OUT: [usize; USERS] = [2, 1, 5, 4];
const EXPLICIT_NEG: [[usize; 3]; USERS] = [[3, 4, 5], [3, 4, 5], [0, 1, 2], [0, 1, 2]];

fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

#[derive(Clone)]
pub(crate) struct MatrixFactorization {
    core: Core,
    pub(crate) users: [[f64; FACTORS]; USERS],
    pub(crate) items: [[f64; FACTORS]; ITEMS],
}

fn validate_data(data: &[TrainingTriple]) {
    assert!(
        !data.is_empty(),
        "pairwise data must contain at least one triple"
    );
    assert!(
        data.iter().all(|&(user, positive, negative)| user < USERS
            && positive < ITEMS
            && negative < ITEMS
            && positive != negative),
        "triples require valid user/item IDs and distinct positive/negative items"
    );
}

impl MatrixFactorization {
    fn new(core: Core) -> Self {
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
        Self { core, users, items }
    }
    pub(crate) fn score(&self, user: usize, item: usize) -> f64 {
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
    fn train(
        &mut self,
        data: &[TrainingTriple],
        epochs: usize,
        learning_rate: f64,
        regularization: f64,
    ) {
        validate_data(data);
        for _ in 0..epochs {
            for &(user, positive, negative) in data {
                (self.core.update)(
                    self,
                    user,
                    positive,
                    negative,
                    learning_rate,
                    regularization,
                );
            }
        }
    }
    fn loss(&self, data: &[TrainingTriple], regularization: f64) -> f64 {
        validate_data(data);
        data.iter()
            .map(|&(user, positive, negative)| {
                self.triple_loss(user, positive, negative, regularization)
            })
            .sum::<f64>()
            / data.len() as f64
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
    let mut result = (0.0, 0.0, 0.0);
    for (user, held_out) in HELD_OUT.iter().enumerate() {
        let metrics = (model.core.query_metrics)(&model.ranked_candidates(user), &[*held_out], k);
        result.0 += metrics.0;
        result.1 += metrics.1;
        result.2 += metrics.2;
    }
    (
        result.0 / USERS as f64,
        result.1 / USERS as f64,
        result.2 / USERS as f64,
    )
}

pub(crate) fn run(core: Core, args: &[String]) -> Result<(), String> {
    if let [flag, stage] = args {
        if flag == "--checkpoint" {
            return check_stage(core, stage);
        }
    }
    let mut triples = TRAIN_DATA;
    if args == ["--rotate-negatives"] {
        for (j, (user, _, negative)) in triples.iter_mut().enumerate() {
            *negative = EXPLICIT_NEG[*user][(j + 1) % 3];
        }
    } else if !args.is_empty() {
        return Err("usage: 24 [--rotate-negatives]".into());
    }
    println!("training triples={triples:?}; epochs=800, learning_rate=0.04, regularization=0.002; four explicit candidates per user");
    let mut model = MatrixFactorization::new(core);
    let before_loss = model.loss(&triples, 0.002);
    let before = ranking_metrics(&model, 2);
    model.train(&triples, 800, 0.04, 0.002);
    let after_loss = model.loss(&triples, 0.002);
    let after = ranking_metrics(&model, 2);
    println!("regularized pairwise training loss {before_loss:.5} -> {after_loss:.5}");
    println!(
        "held-out Recall@2 {:.1}% -> {:.1}%, Precision@2 {:.1}% -> {:.1}%, gain@2 (baseline nCG; goal nDCG) {:.1}% -> {:.1}%",
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
    Ok(())
}
pub(crate) fn check(core: Core) -> Result<(), String> {
    check_stage(core, "all")
}
fn check_stage(core: Core, stage: &str) -> Result<(), String> {
    if !["update", "metrics", "training", "all"].contains(&stage) {
        return Err(format!("unknown chapter24 checkpoint {stage}"));
    }

    let model = MatrixFactorization::new(core);
    let mut changed = model.clone();
    let (u, p, n, rate, reg) = (1, 2, 5, 0.07, 0.013);
    (core.update)(&mut changed, u, p, n, rate, reg);
    for group in 0..3 {
        for factor in 0..FACTORS {
            let mut plus = model.clone();
            let mut minus = model.clone();
            let (before, after) = match group {
                0 => {
                    plus.users[u][factor] += 1e-5;
                    minus.users[u][factor] -= 1e-5;
                    (model.users[u][factor], changed.users[u][factor])
                }
                1 => {
                    plus.items[p][factor] += 1e-5;
                    minus.items[p][factor] -= 1e-5;
                    (model.items[p][factor], changed.items[p][factor])
                }
                _ => {
                    plus.items[n][factor] += 1e-5;
                    minus.items[n][factor] -= 1e-5;
                    (model.items[n][factor], changed.items[n][factor])
                }
            };
            let numeric = (plus.triple_loss(u, p, n, reg) - minus.triple_loss(u, p, n, reg)) / 2e-5;
            crate::close(
                "simultaneous pairwise vector update",
                (before - after) / rate,
                numeric,
            )?;
        }
    }
    if stage == "update" {
        println!("PASS update checkpoint");
        return Ok(());
    }
    let metrics = (core.query_metrics)(&[4, 2, 1, 3, 0], &[1, 3], 3);
    crate::close("multi-relevant recall", metrics.0, 0.5)?;
    crate::close("multi-relevant precision", metrics.1, 1.0 / 3.0)?;
    crate::close(
        "rank-discounted and ideal-normalized DCG",
        metrics.2,
        0.5 / (1.0 + 1.0 / 3.0_f64.log2()),
    )?;
    if stage == "metrics" {
        println!("PASS metrics checkpoint");
        return Ok(());
    }
    for &(user, pos, neg) in &TRAIN_DATA {
        if pos == HELD_OUT[user] || neg == HELD_OUT[user] {
            return Err("GOAL_NOT_MET: held-out interaction leaked into optimization".into());
        }
    }
    let mut trained = MatrixFactorization::new(core);
    let before = trained.loss(&TRAIN_DATA, 0.002);
    trained.train(&TRAIN_DATA, 800, 0.04, 0.002);
    let after = ranking_metrics(&trained, 2);
    if trained.loss(&TRAIN_DATA, 0.002) >= before * 0.25 || after.0 < 0.75 || after.2 < 0.75 {
        return Err(format!(
            "GOAL_NOT_MET: training/ranking goal: loss={}, metrics={after:?}",
            trained.loss(&TRAIN_DATA, 0.002)
        ));
    }
    println!("PASS six factor derivatives, multi-relevant rank metrics, excluded holdouts; Recall@2 {:.3}, Precision@2 {:.3}, nDCG@2 {:.3}",after.0,after.1,after.2);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pairwise_gradient_matches_central_difference() {
        let mut model = MatrixFactorization::new(crate::solutions::ch24::CORE);
        let (u, p, n, reg) = (0, 0, 3, 0.002);
        let ordering_signal = sigmoid(-(model.score(u, p) - model.score(u, n)));
        let analytic = -ordering_signal * (model.items[p][0] - model.items[n][0])
            + 2.0 * reg * model.users[u][0];
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
        let mut model = MatrixFactorization::new(crate::solutions::ch24::CORE);
        let before_loss = model.loss(&TRAIN_DATA, 0.002);
        model.train(&TRAIN_DATA, 800, 0.04, 0.002);
        let metrics = ranking_metrics(&model, 2);
        assert!(model.loss(&TRAIN_DATA, 0.002) < before_loss * 0.25);
        assert!(metrics.0 >= 0.75 && metrics.2 >= 0.75, "{metrics:?}");
    }
    #[test]
    fn training_and_loss_use_the_supplied_triples() {
        let mut model = MatrixFactorization::new(crate::solutions::ch24::CORE);
        let data = [(1, 0, 3)];
        let untouched_user = model.users[0];
        let trained_user = model.users[1];
        assert_eq!(model.loss(&data, 0.002), model.triple_loss(1, 0, 3, 0.002));
        model.train(&data, 1, 0.04, 0.002);
        assert_eq!(model.users[0], untouched_user);
        assert_ne!(model.users[1], trained_user);
    }
    #[test]
    fn stable_softplus_handles_extreme_values() {
        assert!(softplus(1000.0).is_finite());
        assert!(softplus(-1000.0).is_finite());
    }
}
