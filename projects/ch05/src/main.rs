//! A leakage-aware audit of an imbalanced binary classifier.

use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
struct ScoredExample {
    entity: u32,
    target: bool,
    score: f64,
}

const VALIDATION: [ScoredExample; 8] = [
    ScoredExample {
        entity: 11,
        target: true,
        score: 0.72,
    },
    ScoredExample {
        entity: 12,
        target: true,
        score: 0.61,
    },
    ScoredExample {
        entity: 13,
        target: false,
        score: 0.58,
    },
    ScoredExample {
        entity: 14,
        target: false,
        score: 0.42,
    },
    ScoredExample {
        entity: 15,
        target: false,
        score: 0.31,
    },
    ScoredExample {
        entity: 16,
        target: false,
        score: 0.20,
    },
    ScoredExample {
        entity: 17,
        target: false,
        score: 0.10,
    },
    ScoredExample {
        entity: 18,
        target: false,
        score: 0.05,
    },
];

const TEST: [ScoredExample; 10] = [
    ScoredExample {
        entity: 21,
        target: true,
        score: 0.67,
    },
    ScoredExample {
        entity: 22,
        target: true,
        score: 0.49,
    },
    ScoredExample {
        entity: 23,
        target: false,
        score: 0.63,
    },
    ScoredExample {
        entity: 24,
        target: false,
        score: 0.46,
    },
    ScoredExample {
        entity: 25,
        target: false,
        score: 0.40,
    },
    ScoredExample {
        entity: 26,
        target: false,
        score: 0.35,
    },
    ScoredExample {
        entity: 27,
        target: false,
        score: 0.18,
    },
    ScoredExample {
        entity: 28,
        target: false,
        score: 0.12,
    },
    ScoredExample {
        entity: 29,
        target: false,
        score: 0.08,
    },
    ScoredExample {
        entity: 30,
        target: false,
        score: 0.03,
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

fn evaluate(data: &[ScoredExample], threshold: f64) -> Result<Confusion, &'static str> {
    if data.is_empty() || !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err("evaluation needs examples and a threshold in [0, 1]");
    }
    if data
        .iter()
        .any(|example| !example.score.is_finite() || !(0.0..=1.0).contains(&example.score))
    {
        return Err("scores must be finite probabilities");
    }
    Ok(data
        .iter()
        .fold(Confusion::default(), |mut counts, example| {
            match (example.score >= threshold, example.target) {
                (true, true) => counts.true_positive += 1,
                (true, false) => counts.false_positive += 1,
                (false, false) => counts.true_negative += 1,
                (false, true) => counts.false_negative += 1,
            }
            counts
        }))
}

fn choose_threshold(validation: &[ScoredExample], candidates: &[f64]) -> Result<f64, &'static str> {
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

fn no_entity_overlap(left: &[ScoredExample], right: &[ScoredExample]) -> bool {
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
    fn confusion_counts_and_metrics_are_consistent() -> Result<(), &'static str> {
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
        Ok(())
    }

    #[test]
    fn tuning_uses_validation_and_leakage_is_visible() -> Result<(), &'static str> {
        assert_eq!(choose_threshold(&VALIDATION, &[0.4, 0.5, 0.6, 0.7])?, 0.6);
        assert!(no_entity_overlap(&VALIDATION, &TEST));
        let leaked = [ScoredExample {
            entity: 11,
            target: false,
            score: 0.1,
        }];
        assert!(!no_entity_overlap(&VALIDATION, &leaked));
        Ok(())
    }
}
