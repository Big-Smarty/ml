// A quantitative audit across interpretation, robustness, fairness, privacy, security, and causality.

#[derive(Clone, Copy, Debug)]
struct Row {
    features: [f64; 2],
    group: char,
    label: bool,
    member: bool,
}

const ROWS: [Row; 12] = [
    Row {
        features: [1.0, 1.0],
        group: 'A',
        label: true,
        member: true,
    },
    Row {
        features: [0.8, 0.0],
        group: 'A',
        label: true,
        member: true,
    },
    Row {
        features: [0.4, 0.0],
        group: 'A',
        label: false,
        member: false,
    },
    Row {
        features: [-0.8, 0.0],
        group: 'A',
        label: false,
        member: true,
    },
    Row {
        features: [-1.0, 1.0],
        group: 'A',
        label: false,
        member: true,
    },
    Row {
        features: [0.2, -0.5],
        group: 'A',
        label: true,
        member: false,
    },
    Row {
        features: [1.0, -1.0],
        group: 'B',
        label: true,
        member: true,
    },
    Row {
        features: [0.5, -1.0],
        group: 'B',
        label: true,
        member: false,
    },
    Row {
        features: [0.3, -0.5],
        group: 'B',
        label: false,
        member: true,
    },
    Row {
        features: [-0.5, 1.0],
        group: 'B',
        label: false,
        member: true,
    },
    Row {
        features: [0.7, 0.5],
        group: 'B',
        label: false,
        member: false,
    },
    Row {
        features: [-1.0, -1.0],
        group: 'B',
        label: false,
        member: true,
    },
];

#[derive(Clone, Copy, Debug)]
struct Model {
    weights: [f64; 2],
    bias: f64,
    threshold: f64,
}

impl Model {
    fn logit(&self, features: [f64; 2]) -> Result<f64, &'static str> {
        if self.weights.iter().any(|value| !value.is_finite())
            || !self.bias.is_finite()
            || !self.threshold.is_finite()
            || !(0.0..=1.0).contains(&self.threshold)
        {
            return Err("model parameters and threshold are invalid");
        }
        if features
            .iter()
            .any(|v| !v.is_finite() || !(-5.0..=5.0).contains(v))
        {
            return Err("features must be finite and within [-5,5]");
        }
        let output = self.weights[0] * features[0] + self.weights[1] * features[1] + self.bias;
        output
            .is_finite()
            .then_some(output)
            .ok_or("logit is nonfinite")
    }

    fn probability(&self, features: [f64; 2]) -> Result<f64, &'static str> {
        Ok(sigmoid(self.logit(features)?))
    }

    fn predict(&self, features: [f64; 2]) -> Result<bool, &'static str> {
        Ok(self.probability(features)? >= self.threshold)
    }

    fn loss(&self, row: Row) -> Result<f64, &'static str> {
        binary_cross_entropy_from_logit(self.logit(row.features)?, f64::from(row.label))
    }
}

fn sigmoid(logit: f64) -> f64 {
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exp = logit.exp();
        exp / (1.0 + exp)
    }
}

fn binary_cross_entropy_from_logit(logit: f64, target: f64) -> Result<f64, &'static str> {
    if !logit.is_finite() || !target.is_finite() || !(0.0..=1.0).contains(&target) {
        return Err("logit must be finite and target must be in [0, 1]");
    }
    Ok(logit.max(0.0) - logit * target + (-logit.abs()).exp().ln_1p())
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
    let learning_rate = 0.05;
    for _ in 0..400 {
        let mut weight_gradient = [0.0; 2];
        let mut bias_gradient = 0.0;
        for row in &training {
            let error = model.probability(row.features)? - f64::from(row.label);
            weight_gradient[0] += error * row.features[0];
            weight_gradient[1] += error * row.features[1];
            bias_gradient += error;
        }
        let n = training.len() as f64;
        model.weights[0] -= learning_rate * weight_gradient[0] / n;
        model.weights[1] -= learning_rate * weight_gradient[1] / n;
        model.bias -= learning_rate * bias_gradient / n;
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

#[derive(Debug)]
struct MembershipAudit {
    true_positive_rate: f64,
    false_positive_rate: f64,
    advantage: f64,
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

fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let model = Model {
        weights: [0.8, 0.4],
        bias: -0.2,
        threshold: 0.5,
    };
    for group in ['A', 'B'] {
        let metrics = group_metrics(&model, &ROWS, group)?;
        println!(
            "group {} n={} accuracy={:.3} positive_rate={:.3} TPR={:?} FPR={:?}",
            metrics.group,
            metrics.count,
            metrics.accuracy,
            metrics.positive_rate,
            metrics.true_positive_rate,
            metrics.false_positive_rate
        );
    }
    let attribution = local_attribution(&model, ROWS[0].features, [0.0, 0.0])?;
    println!(
        "local logit ablation contributions: feature0={:.3}, feature1={:.3}",
        attribution[0], attribution[1]
    );
    let (flips, change) = robustness(&model, &ROWS, 0.25)?;
    println!(
        "robustness at radius 0.25: {flips} boundary flips, max probability change {change:.3}"
    );
    let member_model = train_on_members(&ROWS)?;
    let privacy = membership_attack(&member_model, &ROWS, 0.45)?;
    println!(
        "loss-threshold membership attack: TPR={:?} FPR={:?} advantage={:.3}",
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
