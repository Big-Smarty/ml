#[derive(Clone, Copy)]
struct PredictionRow {
    group: char,
    label: bool,
    predicted: bool,
}

#[cfg(test)]
fn true_positive_rate(rows: &[PredictionRow], group: char) -> Option<f64> {
    // TODO: divide true positives by all positive labels in this group.
    let _ = (rows, group);
    todo!("compute a group-conditional denominator")
}

fn main() {
    let rows = [
        PredictionRow {
            group: 'A',
            label: true,
            predicted: true,
        },
        PredictionRow {
            group: 'A',
            label: false,
            predicted: false,
        },
        PredictionRow {
            group: 'B',
            label: true,
            predicted: false,
        },
    ];
    let correct = rows.iter().filter(|r| r.label == r.predicted).count();
    let group_a = rows.iter().filter(|r| r.group == 'A').count();
    println!("overall accuracy: {correct}/{}", rows.len());
    println!("group A rows: {group_a}");
    println!("Overall accuracy is the checkpoint; the audit needs group denominators.");
}

#[test]
fn group_recall_exposes_difference() {
    let rows = [
        PredictionRow {
            group: 'A',
            label: true,
            predicted: true,
        },
        PredictionRow {
            group: 'B',
            label: true,
            predicted: false,
        },
    ];
    assert_eq!(true_positive_rate(&rows, 'A'), Some(1.0));
    assert_eq!(true_positive_rate(&rows, 'B'), Some(0.0));
}
