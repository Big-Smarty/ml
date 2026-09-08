//! Bootstrap uncertainty and probability calibration on course-authored predictions.
#[derive(Clone, Copy)]
struct Prediction {
    probability: f64,
    label: u8,
}

fn validate_predictions(data: &[Prediction]) -> Result<(), &'static str> {
    if data.is_empty() {
        return Err("evaluation needs at least one prediction");
    }
    if data
        .iter()
        .any(|p| !p.probability.is_finite() || !(0.0..=1.0).contains(&p.probability) || p.label > 1)
    {
        return Err("probabilities must be in [0,1] and labels must be 0 or 1");
    }
    Ok(())
}

fn accuracy(data: &[Prediction]) -> Result<f64, &'static str> {
    validate_predictions(data)?;
    Ok(data
        .iter()
        .filter(|p| (p.probability >= 0.5) == (p.label == 1))
        .count() as f64
        / data.len() as f64)
}

fn brier(data: &[Prediction]) -> Result<f64, &'static str> {
    validate_predictions(data)?;
    Ok(data
        .iter()
        .map(|p| (p.probability - p.label as f64).powi(2))
        .sum::<f64>()
        / data.len() as f64)
}

#[derive(Clone, Copy)]
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn index(&mut self, n: usize) -> usize {
        self.next() as usize % n
    }
}

fn bootstrap_accuracy(
    data: &[Prediction],
    repetitions: usize,
    seed: u64,
) -> Result<(f64, f64), &'static str> {
    accuracy(data)?;
    if repetitions < 40 {
        return Err("use at least 40 bootstrap repetitions");
    }
    let mut rng = Rng(if seed == 0 { 1 } else { seed });
    let mut estimates = Vec::with_capacity(repetitions);
    let mut sample = Vec::with_capacity(data.len());
    for _ in 0..repetitions {
        sample.clear();
        for _ in 0..data.len() {
            sample.push(data[rng.index(data.len())]);
        }
        estimates.push(accuracy(&sample)?);
    }
    estimates.sort_by(f64::total_cmp);
    let lower = estimates[((repetitions as f64 * 0.025).floor() as usize).min(repetitions - 1)];
    let upper = estimates[((repetitions as f64 * 0.975).floor() as usize).min(repetitions - 1)];
    Ok((lower, upper))
}

fn calibration_bins(
    data: &[Prediction],
    bins: usize,
) -> Result<Vec<(usize, f64, f64)>, &'static str> {
    validate_predictions(data)?;
    if bins == 0 {
        return Err("calibration needs at least one bin");
    }
    let mut totals = vec![(0usize, 0.0, 0.0); bins];
    for p in data {
        let i = ((p.probability * bins as f64) as usize).min(bins - 1);
        totals[i].0 += 1;
        totals[i].1 += p.probability;
        totals[i].2 += p.label as f64;
    }
    Ok(totals
        .into_iter()
        .filter(|x| x.0 > 0)
        .map(|(n, ps, ys)| (n, ps / n as f64, ys / n as f64))
        .collect())
}

fn main() -> Result<(), &'static str> {
    let data = [
        Prediction {
            probability: 0.05,
            label: 0,
        },
        Prediction {
            probability: 0.15,
            label: 0,
        },
        Prediction {
            probability: 0.30,
            label: 1,
        },
        Prediction {
            probability: 0.45,
            label: 0,
        },
        Prediction {
            probability: 0.55,
            label: 1,
        },
        Prediction {
            probability: 0.70,
            label: 0,
        },
        Prediction {
            probability: 0.85,
            label: 1,
        },
        Prediction {
            probability: 0.95,
            label: 1,
        },
    ];
    let (lo, hi) = bootstrap_accuracy(&data, 2_000, 7)?;
    println!(
        "accuracy {:.3}; 95% bootstrap percentile confidence interval [{lo:.3}, {hi:.3}]",
        accuracy(&data)?
    );
    println!("Brier score {:.4}", brier(&data)?);
    for (n, mean_probability, positive_frequency) in calibration_bins(&data, 4)? {
        println!(
            "bin n={n}: mean probability {mean_probability:.3}, positive frequency {positive_frequency:.3}"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn arithmetic_and_determinism() {
        let d = [
            Prediction {
                probability: 0.1,
                label: 0,
            },
            Prediction {
                probability: 0.8,
                label: 1,
            },
            Prediction {
                probability: 0.6,
                label: 0,
            },
            Prediction {
                probability: 0.9,
                label: 1,
            },
        ];
        assert_eq!(accuracy(&d).unwrap(), 0.75);
        assert!((brier(&d).unwrap() - 0.105).abs() < 1e-12);
        assert_eq!(
            bootstrap_accuracy(&d, 400, 3).unwrap(),
            bootstrap_accuracy(&d, 400, 3).unwrap()
        );
        assert!(accuracy(&[]).is_err());
        assert!(brier(&[]).is_err());
        assert!(brier(&[Prediction {
            probability: f64::NAN,
            label: 0,
        }])
        .is_err());
    }
    #[test]
    fn bootstrap_boundaries_and_calibration_are_real_statistics() {
        let correct = [Prediction {
            probability: 1.0,
            label: 1,
        }; 4];
        let wrong = [Prediction {
            probability: 1.0,
            label: 0,
        }; 4];
        assert_eq!(bootstrap_accuracy(&correct, 400, 7).unwrap(), (1.0, 1.0));
        assert_eq!(bootstrap_accuracy(&wrong, 400, 7).unwrap(), (0.0, 0.0));
        let mixed = [correct[0], correct[0], wrong[0], wrong[0]];
        let (lo, hi) = bootstrap_accuracy(&mixed, 400, 7).unwrap();
        assert!(lo < 0.5 && hi > 0.5);
        assert_eq!(calibration_bins(&mixed, 4).unwrap(), vec![(4, 1.0, 0.5)]);
        for probability in [f64::NAN, -0.1, 1.1] {
            assert!(accuracy(&[Prediction {
                probability,
                label: 0
            }])
            .is_err());
        }
        assert!(accuracy(&[Prediction {
            probability: 0.5,
            label: 2
        }])
        .is_err());
    }
}
