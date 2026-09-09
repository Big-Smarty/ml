//! Chapter 54 learner algorithms. Baseline reports pooled metrics for each requested label. Implement genuinely grouped rates, attribution, named perturbations, membership discrimination, and stratum standardization.
include!("common/ch54.rs");
include!("checks/ch54.rs");
fn group_metrics(model: &Model, rows: &[Row], group: char) -> Result<GroupMetrics, &'static str> {
    if rows.is_empty() {
        return Err("pooled report has no rows");
    }
    if !rows.iter().any(|row| row.group == group) {
        return Err("group has no rows");
    }
    let mut correct = 0;
    let mut predicted_positive = 0;
    for row in rows {
        let prediction = model.predict(row.features)?;
        correct += usize::from(prediction == row.label);
        predicted_positive += usize::from(prediction);
    }
    // Valid pooled accuracy baseline. Group-specific conditional rates are absent,
    // rather than pretending that accuracy is TPR or an undefined denominator is zero.
    Ok(GroupMetrics {
        group: '*',
        count: rows.len(),
        accuracy: correct as f64 / rows.len() as f64,
        positive_rate: predicted_positive as f64 / rows.len() as f64,
        true_positive_rate: None,
        false_positive_rate: None,
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
