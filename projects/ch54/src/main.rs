//! A quantitative audit across interpretation, robustness, fairness, privacy, security, and causality.

#[derive(Clone, Copy, Debug)]
struct Row {
    x: [f64; 2],
    group: char,
    label: bool,
    member: bool,
}

const ROWS: [Row; 12] = [
    Row {
        x: [1.0, 1.0],
        group: 'A',
        label: true,
        member: true,
    },
    Row {
        x: [0.8, 0.0],
        group: 'A',
        label: true,
        member: true,
    },
    Row {
        x: [0.4, 0.0],
        group: 'A',
        label: false,
        member: false,
    },
    Row {
        x: [-0.8, 0.0],
        group: 'A',
        label: false,
        member: true,
    },
    Row {
        x: [-1.0, 1.0],
        group: 'A',
        label: false,
        member: true,
    },
    Row {
        x: [0.2, -0.5],
        group: 'A',
        label: true,
        member: false,
    },
    Row {
        x: [1.0, -1.0],
        group: 'B',
        label: true,
        member: true,
    },
    Row {
        x: [0.5, -1.0],
        group: 'B',
        label: true,
        member: false,
    },
    Row {
        x: [0.3, -0.5],
        group: 'B',
        label: false,
        member: true,
    },
    Row {
        x: [-0.5, 1.0],
        group: 'B',
        label: false,
        member: true,
    },
    Row {
        x: [0.7, 0.5],
        group: 'B',
        label: false,
        member: false,
    },
    Row {
        x: [-1.0, -1.0],
        group: 'B',
        label: false,
        member: true,
    },
];

#[derive(Clone, Copy)]
struct Model {
    weights: [f64; 2],
    bias: f64,
    threshold: f64,
}

impl Model {
    fn logit(self, x: [f64; 2]) -> Result<f64, &'static str> {
        if self.weights.iter().any(|value| !value.is_finite())
            || !self.bias.is_finite()
            || !self.threshold.is_finite()
            || !(0.0..=1.0).contains(&self.threshold)
        {
            return Err("model parameters and threshold are invalid");
        }
        if x.iter()
            .any(|v| !v.is_finite() || !(-5.0..=5.0).contains(v))
        {
            return Err("features must be finite and within [-5,5]");
        }
        let output = self.weights[0] * x[0] + self.weights[1] * x[1] + self.bias;
        output
            .is_finite()
            .then_some(output)
            .ok_or("logit is nonfinite")
    }

    fn probability(self, x: [f64; 2]) -> Result<f64, &'static str> {
        Ok(1.0 / (1.0 + (-self.logit(x)?).exp()))
    }

    fn predicts_positive(self, x: [f64; 2]) -> Result<bool, &'static str> {
        Ok(self.probability(x)? >= self.threshold)
    }

    fn loss(self, row: Row) -> Result<f64, &'static str> {
        let logit = self.logit(row.x)?;
        Ok(if row.label {
            softplus(-logit)
        } else {
            softplus(logit)
        })
    }
}

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

fn train_on_members(rows: &[Row]) -> Result<Model, &'static str> {
    let training: Vec<_> = rows.iter().copied().filter(|row| row.member).collect();
    if training.is_empty() {
        return Err("membership demonstration needs training members");
    }
    let mut model = Model {
        weights: [0.0; 2],
        bias: 0.0,
        threshold: 0.5,
    };
    for _ in 0..400 {
        let mut dw = [0.0; 2];
        let mut db = 0.0;
        for row in &training {
            let error = model.probability(row.x)? - f64::from(row.label);
            dw[0] += error * row.x[0];
            dw[1] += error * row.x[1];
            db += error;
        }
        let scale = 0.05 / training.len() as f64;
        model.weights[0] -= scale * dw[0];
        model.weights[1] -= scale * dw[1];
        model.bias -= scale * db;
    }
    Ok(model)
}

#[derive(Debug)]
struct GroupMetrics {
    group: char,
    count: usize,
    accuracy: f64,
    positive_rate: f64,
    true_positive_rate: Option<f64>,
    false_positive_rate: Option<f64>,
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

fn group_metrics(model: Model, rows: &[Row], group: char) -> Result<GroupMetrics, &'static str> {
    let selected: Vec<_> = rows
        .iter()
        .copied()
        .filter(|row| row.group == group)
        .collect();
    if selected.is_empty() {
        return Err("group has no rows");
    }
    let mut correct = 0;
    let mut predicted_positive = 0;
    let mut positives = 0;
    let mut negatives = 0;
    let mut true_positives = 0;
    let mut false_positives = 0;
    for row in &selected {
        let prediction = model.predicts_positive(row.x)?;
        correct += usize::from(prediction == row.label);
        predicted_positive += usize::from(prediction);
        positives += usize::from(row.label);
        negatives += usize::from(!row.label);
        true_positives += usize::from(prediction && row.label);
        false_positives += usize::from(prediction && !row.label);
    }
    Ok(GroupMetrics {
        group,
        count: selected.len(),
        accuracy: correct as f64 / selected.len() as f64,
        positive_rate: predicted_positive as f64 / selected.len() as f64,
        true_positive_rate: rate(true_positives, positives),
        false_positive_rate: rate(false_positives, negatives),
    })
}

fn local_attribution(
    model: Model,
    x: [f64; 2],
    baseline: [f64; 2],
) -> Result<[f64; 2], &'static str> {
    let original = model.logit(x)?;
    let mut result = [0.0; 2];
    for feature in 0..2 {
        let mut ablated = x;
        ablated[feature] = baseline[feature];
        result[feature] = original - model.logit(ablated)?;
    }
    Ok(result)
}

fn robustness(model: Model, rows: &[Row], radius: f64) -> Result<(usize, f64), &'static str> {
    if !radius.is_finite() || radius < 0.0 {
        return Err("radius must be finite and nonnegative");
    }
    let mut flips = 0;
    let mut maximum_change = 0.0_f64;
    for row in rows {
        let original = model.probability(row.x)?;
        for delta in [-radius, radius] {
            let changed = model.probability([row.x[0] + delta, row.x[1]])?;
            flips += usize::from((original >= model.threshold) != (changed >= model.threshold));
            maximum_change = maximum_change.max((original - changed).abs());
        }
    }
    Ok((flips, maximum_change))
}

#[derive(Debug)]
struct MembershipAudit {
    true_positive_rate: f64,
    false_positive_rate: f64,
    advantage: f64,
}

fn membership_attack(
    model: Model,
    rows: &[Row],
    loss_threshold: f64,
) -> Result<MembershipAudit, &'static str> {
    if !loss_threshold.is_finite() || loss_threshold < 0.0 {
        return Err("loss threshold must be finite and nonnegative");
    }
    let mut member_count = 0;
    let mut nonmember_count = 0;
    let mut member_guesses = 0;
    let mut nonmember_guesses = 0;
    for &row in rows {
        let guess = model.loss(row)? < loss_threshold;
        if row.member {
            member_count += 1;
            member_guesses += usize::from(guess);
        } else {
            nonmember_count += 1;
            nonmember_guesses += usize::from(guess);
        }
    }
    let tpr = rate(member_guesses, member_count).ok_or("no members")?;
    let fpr = rate(nonmember_guesses, nonmember_count).ok_or("no nonmembers")?;
    Ok(MembershipAudit {
        true_positive_rate: tpr,
        false_positive_rate: fpr,
        advantage: tpr - fpr,
    })
}

fn parse_input(text: &str) -> Result<[f64; 2], &'static str> {
    let fields: Vec<_> = text.split(',').collect();
    if fields.len() != 3 || fields[0] != "schema=1" {
        return Err("expected schema=1,f0=<number>,f1=<number>");
    }
    let f0: f64 = fields[1]
        .strip_prefix("f0=")
        .ok_or("missing f0")?
        .parse()
        .map_err(|_| "invalid f0")?;
    let f1: f64 = fields[2]
        .strip_prefix("f1=")
        .ok_or("missing f1")?
        .parse()
        .map_err(|_| "invalid f1")?;
    let values = [f0, f1];
    if values
        .iter()
        .any(|v| !v.is_finite() || !(-5.0..=5.0).contains(v))
    {
        return Err("features must be finite and within [-5,5]");
    }
    Ok(values)
}

#[derive(Clone, Copy)]
struct Stratum {
    treated_total: usize,
    treated_outcomes: usize,
    control_total: usize,
    control_outcomes: usize,
}

fn association_contrasts(strata: &[Stratum]) -> Result<(f64, f64), &'static str> {
    if strata.is_empty()
        || strata.iter().any(|s| {
            s.treated_total == 0
                || s.control_total == 0
                || s.treated_outcomes > s.treated_total
                || s.control_outcomes > s.control_total
        })
    {
        return Err("every valid stratum needs treated and control observations");
    }
    let treated_total: usize = strata.iter().map(|s| s.treated_total).sum();
    let control_total: usize = strata.iter().map(|s| s.control_total).sum();
    let crude = strata.iter().map(|s| s.treated_outcomes).sum::<usize>() as f64
        / treated_total as f64
        - strata.iter().map(|s| s.control_outcomes).sum::<usize>() as f64 / control_total as f64;
    let standardized = strata
        .iter()
        .map(|s| {
            s.treated_outcomes as f64 / s.treated_total as f64
                - s.control_outcomes as f64 / s.control_total as f64
        })
        .sum::<f64>()
        / strata.len() as f64;
    Ok((crude, standardized))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model {
        weights: [0.8, 0.4],
        bias: -0.2,
        threshold: 0.5,
    };
    for group in ['A', 'B'] {
        let metrics = group_metrics(model, &ROWS, group)?;
        println!(
            "group {} n={} accuracy={:.3} positive_rate={:.3} TPR={:.3} FPR={:.3}",
            metrics.group,
            metrics.count,
            metrics.accuracy,
            metrics.positive_rate,
            metrics.true_positive_rate.ok_or("TPR undefined")?,
            metrics.false_positive_rate.ok_or("FPR undefined")?
        );
    }
    let attribution = local_attribution(model, ROWS[0].x, [0.0, 0.0])?;
    println!(
        "local logit ablation contributions: feature0={:.3}, feature1={:.3}",
        attribution[0], attribution[1]
    );
    let (flips, change) = robustness(model, &ROWS, 0.25)?;
    println!(
        "robustness at radius 0.25: {flips} boundary flips, max probability change {change:.3}"
    );
    let privacy = membership_attack(train_on_members(&ROWS)?, &ROWS, 0.45)?;
    println!(
        "loss-threshold membership attack: TPR={:.3} FPR={:.3} advantage={:.3}",
        privacy.true_positive_rate, privacy.false_positive_rate, privacy.advantage
    );
    println!(
        "security boundary rejects NaN: {}",
        parse_input("schema=1,f0=NaN,f1=0").is_err()
    );
    let strata = [
        Stratum {
            treated_total: 10,
            treated_outcomes: 1,
            control_total: 90,
            control_outcomes: 18,
        },
        Stratum {
            treated_total: 90,
            treated_outcomes: 54,
            control_total: 10,
            control_outcomes: 7,
        },
    ];
    let (crude, standardized) = association_contrasts(&strata)?;
    println!(
        "outcome-rate difference: crude={crude:+.3}, equal-stratum standardized={standardized:+.3}"
    );
    println!("These diagnostics expose evidence gaps; they do not certify fairness, privacy, robustness, or causality.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_metrics_use_their_own_denominators() -> Result<(), &'static str> {
        let model = Model {
            weights: [0.8, 0.4],
            bias: -0.2,
            threshold: 0.5,
        };
        let a = group_metrics(model, &ROWS, 'A')?;
        let b = group_metrics(model, &ROWS, 'B')?;
        assert_ne!(a.positive_rate, b.positive_rate);
        assert_ne!(a.true_positive_rate, b.true_positive_rate);
        let privacy = membership_attack(train_on_members(&ROWS)?, &ROWS, 0.45)?;
        assert!(privacy.advantage.is_finite());
        Ok(())
    }

    #[test]
    fn boundaries_and_causal_adjustment_change_the_conclusion() -> Result<(), &'static str> {
        assert_eq!(parse_input("schema=1,f0=1.0,f1=-2.0")?, [1.0, -2.0]);
        assert!(parse_input("schema=1,f0=inf,f1=0").is_err());
        assert!(parse_input("schema=2,f0=1,f1=0").is_err());
        let strata = [
            Stratum {
                treated_total: 10,
                treated_outcomes: 1,
                control_total: 90,
                control_outcomes: 18,
            },
            Stratum {
                treated_total: 90,
                treated_outcomes: 54,
                control_total: 10,
                control_outcomes: 7,
            },
        ];
        let (crude, standardized) = association_contrasts(&strata)?;
        assert!(crude > 0.0 && standardized < 0.0);
        Ok(())
    }
}
