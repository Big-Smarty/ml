//! Worked solution: Filter the requested group before computing every confusion count and denominator. Undefined conditional rates remain None. The other audit routines retain their declared baseline, perturbation, attack populations, parsing grammar, and association target; none of these arithmetic checks certifies the broader property.
//! Compare intermediate quantities with the lesson before using the final goal check.
//! Worked solution for chapter 54. Read the comments and lesson explanations before comparing.
include!("../common/ch54.rs");
include!("../checks/ch54.rs");
fn group_metrics(model: &Model, rows: &[Row], group: char) -> Result<GroupMetrics, &'static str> {
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
        let prediction = model.predict(row.features)?;
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
    model: &Model,
    features: [f64; 2],
    baseline: [f64; 2],
) -> Result<[f64; 2], &'static str> {
    let original = model.logit(features)?;
    let mut result = [0.0; 2];
    for feature in 0..2 {
        let mut ablated = features;
        ablated[feature] = baseline[feature];
        result[feature] = original - model.logit(ablated)?;
    }
    Ok(result)
}

fn robustness(model: &Model, rows: &[Row], radius: f64) -> Result<(usize, f64), &'static str> {
    if !radius.is_finite() || radius < 0.0 {
        return Err("radius must be finite and nonnegative");
    }
    let mut flips = 0;
    let mut maximum_change = 0.0_f64;
    for row in rows {
        let original = model.probability(row.features)?;
        for delta in [-radius, radius] {
            let changed = model.probability([row.features[0] + delta, row.features[1]])?;
            flips += usize::from((original >= model.threshold) != (changed >= model.threshold));
            maximum_change = maximum_change.max((original - changed).abs());
        }
    }
    Ok((flips, maximum_change))
}

fn membership_attack(
    model: &Model,
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
        let a = group_metrics(&model, &ROWS, 'A')?;
        let b = group_metrics(&model, &ROWS, 'B')?;
        assert!((a.accuracy - 2.0 / 3.0).abs() < 1e-12);
        assert!((a.positive_rate - 0.5).abs() < 1e-12);
        assert_eq!(a.true_positive_rate, Some(2.0 / 3.0));
        assert_eq!(a.false_positive_rate, Some(1.0 / 3.0));
        assert!((b.accuracy - 2.0 / 3.0).abs() < 1e-12);
        assert!((b.positive_rate - 1.0 / 3.0).abs() < 1e-12);
        assert_eq!(b.true_positive_rate, Some(0.5));
        assert_eq!(b.false_positive_rate, Some(0.25));
        let attribution = local_attribution(&model, ROWS[0].features, [0.0; 2])?;
        assert!((attribution[0] - 0.8).abs() < 1e-12);
        assert!((attribution[1] - 0.4).abs() < 1e-12);
        let (flips, maximum_change) = robustness(&model, &ROWS, 0.25)?;
        assert_eq!(flips, 4);
        assert!((maximum_change - 0.049_953_391_920_153_47).abs() < 1e-12);
        let member_model = train_on_members(&ROWS)?;
        let privacy = membership_attack(&member_model, &ROWS, 0.45)?;
        assert_eq!(privacy.true_positive_rate, 0.875);
        assert_eq!(privacy.false_positive_rate, 0.0);
        assert_eq!(privacy.advantage, 0.875);
        Ok(())
    }

    #[test]
    fn boundaries_and_causal_adjustment_change_the_conclusion() -> Result<(), &'static str> {
        assert_eq!(parse_input("schema=1,f0=1.0,f1=-2.0")?, [1.0, -2.0]);
        assert!(parse_input("schema=1&x=1.0").is_err());
        assert!(parse_input("schema=1,f0=inf,f1=0").is_err());
        assert!(parse_input("schema=2,f0=1,f1=0").is_err());
        assert_eq!(sigmoid(-1_000.0), 0.0);
        assert_eq!(binary_cross_entropy_from_logit(1_000.0, 0.0)?, 1_000.0);
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
