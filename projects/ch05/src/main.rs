//! A leakage-aware audit of saved probabilities from an imbalanced binary classifier.

use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
struct EvaluationExample {
    entity: u32,
    target: bool,
    probability: f64,
}

const VALIDATION: [EvaluationExample; 8] = [
    EvaluationExample {
        entity: 11,
        target: true,
        probability: 0.72,
    },
    EvaluationExample {
        entity: 12,
        target: true,
        probability: 0.61,
    },
    EvaluationExample {
        entity: 13,
        target: false,
        probability: 0.58,
    },
    EvaluationExample {
        entity: 14,
        target: false,
        probability: 0.42,
    },
    EvaluationExample {
        entity: 15,
        target: false,
        probability: 0.31,
    },
    EvaluationExample {
        entity: 16,
        target: false,
        probability: 0.20,
    },
    EvaluationExample {
        entity: 17,
        target: false,
        probability: 0.10,
    },
    EvaluationExample {
        entity: 18,
        target: false,
        probability: 0.05,
    },
];

const TEST: [EvaluationExample; 10] = [
    EvaluationExample {
        entity: 21,
        target: true,
        probability: 0.67,
    },
    EvaluationExample {
        entity: 22,
        target: true,
        probability: 0.49,
    },
    EvaluationExample {
        entity: 23,
        target: false,
        probability: 0.63,
    },
    EvaluationExample {
        entity: 24,
        target: false,
        probability: 0.46,
    },
    EvaluationExample {
        entity: 25,
        target: false,
        probability: 0.40,
    },
    EvaluationExample {
        entity: 26,
        target: false,
        probability: 0.35,
    },
    EvaluationExample {
        entity: 27,
        target: false,
        probability: 0.18,
    },
    EvaluationExample {
        entity: 28,
        target: false,
        probability: 0.12,
    },
    EvaluationExample {
        entity: 29,
        target: false,
        probability: 0.08,
    },
    EvaluationExample {
        entity: 30,
        target: false,
        probability: 0.03,
    },
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Confusion {
    true_positive: usize,
    false_positive: usize,
    true_negative: usize,
    false_negative: usize,
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    accuracy: f64,
    precision: f64,
    recall: f64,
    f1: f64,
    balanced_accuracy: f64,
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

impl Confusion {
    fn metrics(self) -> Metrics {
        let positive_recall = ratio(self.true_positive, self.true_positive + self.false_negative);
        let negative_recall = ratio(self.true_negative, self.true_negative + self.false_positive);
        let precision = ratio(self.true_positive, self.true_positive + self.false_positive);
        let f1 = if precision + positive_recall == 0.0 {
            0.0
        } else {
            2.0 * precision * positive_recall / (precision + positive_recall)
        };
        Metrics {
            accuracy: ratio(
                self.true_positive + self.true_negative,
                self.true_positive + self.false_positive + self.true_negative + self.false_negative,
            ),
            precision,
            recall: positive_recall,
            f1,
            balanced_accuracy: (positive_recall + negative_recall) / 2.0,
        }
    }
}

fn predict(probability: f64, threshold: f64) -> bool {
    probability >= threshold
}

fn evaluate(data: &[EvaluationExample], threshold: f64) -> Result<Confusion, &'static str> {
    if data.is_empty() || !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err("evaluation needs examples and a threshold in [0, 1]");
    }
    if data.iter().any(|example| {
        !example.probability.is_finite() || !(0.0..=1.0).contains(&example.probability)
    }) {
        return Err("probabilities must be finite and in [0, 1]");
    }
    Ok(data
        .iter()
        .fold(Confusion::default(), |mut counts, example| {
            match (predict(example.probability, threshold), example.target) {
                (true, true) => counts.true_positive += 1,
                (true, false) => counts.false_positive += 1,
                (false, false) => counts.true_negative += 1,
                (false, true) => counts.false_negative += 1,
            }
            counts
        }))
}

fn choose_threshold(
    validation: &[EvaluationExample],
    candidates: &[f64],
) -> Result<f64, &'static str> {
    candidates
        .iter()
        .copied()
        .try_fold(None, |best, threshold| {
            let f1 = evaluate(validation, threshold)?.metrics().f1;
            Ok(match best {
                Some((best_threshold, best_f1)) if best_f1 >= f1 => Some((best_threshold, best_f1)),
                _ => Some((threshold, f1)),
            })
        })
        .and_then(|best| {
            best.map(|(threshold, _)| threshold)
                .ok_or("at least one threshold is required")
        })
}

fn no_entity_overlap(left: &[EvaluationExample], right: &[EvaluationExample]) -> bool {
    let entities: HashSet<_> = left.iter().map(|example| example.entity).collect();
    right
        .iter()
        .all(|example| !entities.contains(&example.entity))
}

fn print_metrics(label: &str, metrics: Metrics) {
    println!(
        "{label}: accuracy={:.3} precision={:.3} recall={:.3} f1={:.3} balanced_accuracy={:.3}",
        metrics.accuracy, metrics.precision, metrics.recall, metrics.f1, metrics.balanced_accuracy
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !no_entity_overlap(&VALIDATION, &TEST) {
        return Err("entity leakage between validation and test".into());
    }
    let threshold = choose_threshold(&VALIDATION, &[0.3, 0.4, 0.5, 0.6, 0.7])?;
    println!("threshold selected on validation: {threshold:.2}");
    let test_counts = evaluate(&TEST, threshold)?;
    println!("test confusion counts: {test_counts:?}");
    print_metrics("model on untouched test", test_counts.metrics());
    let majority_baseline = Confusion {
        true_negative: 8,
        false_negative: 2,
        ..Confusion::default()
    };
    println!("baseline confusion counts: {majority_baseline:?}");
    print_metrics("always-negative baseline", majority_baseline.metrics());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_produce_consistent_counts_and_metrics() -> Result<(), &'static str> {
        assert!(!predict(0.49, 0.5));
        assert!(predict(0.50, 0.5));
        let counts = evaluate(&TEST, 0.5)?;
        assert_eq!(
            counts,
            Confusion {
                true_positive: 1,
                false_positive: 1,
                true_negative: 7,
                false_negative: 1
            }
        );
        let metrics = counts.metrics();
        assert!((metrics.accuracy - 0.8).abs() < 1e-12);
        assert!((metrics.precision - 0.5).abs() < 1e-12);
        assert!((metrics.recall - 0.5).abs() < 1e-12);
        assert!(evaluate(&[], 0.5).is_err());
        assert!(evaluate(&TEST, f64::NAN).is_err());
        let invalid = [EvaluationExample {
            entity: 31,
            target: false,
            probability: 1.1,
        }];
        assert!(evaluate(&invalid, 0.5).is_err());
        Ok(())
    }

    #[test]
    fn tuning_uses_validation_and_leakage_is_visible() -> Result<(), &'static str> {
        assert_eq!(choose_threshold(&VALIDATION, &[0.4, 0.5, 0.6, 0.7])?, 0.6);
        assert!(no_entity_overlap(&VALIDATION, &TEST));
        let leaked = [EvaluationExample {
            entity: 11,
            target: false,
            probability: 0.1,
        }];
        assert!(!no_entity_overlap(&VALIDATION, &leaked));
        Ok(())
    }
}
