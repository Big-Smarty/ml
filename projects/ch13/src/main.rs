//! Leakage-safe fitting of missing-value and categorical preprocessing.
#[derive(Clone, Copy, Debug)]
struct RawRow {
    value: Option<f64>,
    category: &'static str,
    group: &'static str,
    time: u32,
    label: u8,
}

#[derive(Debug)]
struct Preprocessor {
    median: f64,
    categories: Vec<String>,
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle]
    } else {
        values[middle - 1].midpoint(values[middle])
    })
}

impl Preprocessor {
    fn fit(training_rows: &[RawRow]) -> Result<Self, &'static str> {
        if training_rows.is_empty() {
            return Err("fit needs training rows");
        }
        let mut values: Vec<f64> = training_rows
            .iter()
            .filter_map(|raw_row| raw_row.value)
            .collect();
        if values.iter().any(|value| !value.is_finite()) {
            return Err("training numeric values must include a finite value");
        }
        let median =
            median(&mut values).ok_or("training numeric values must include a finite value")?;
        let mut categories: Vec<String> = training_rows
            .iter()
            .map(|raw_row| raw_row.category.to_owned())
            .collect();
        categories.sort();
        categories.dedup();
        Ok(Self { median, categories })
    }
    fn transform(&self, raw_row: RawRow) -> Result<Vec<f64>, &'static str> {
        let value = raw_row.value.unwrap_or(self.median);
        if !value.is_finite() {
            return Err("numeric values must be finite");
        }
        let mut encoded_features = vec![value, raw_row.value.is_none() as u8 as f64];
        encoded_features.extend(
            self.categories
                .iter()
                .map(|category| (category == raw_row.category) as u8 as f64),
        );
        Ok(encoded_features)
    }
}

fn group_split<'a>(
    raw_rows: &'a [RawRow],
    held_out_group: &str,
) -> (Vec<&'a RawRow>, Vec<&'a RawRow>) {
    raw_rows
        .iter()
        .partition(|raw_row| raw_row.group != held_out_group)
}
fn time_split(raw_rows: &[RawRow], cutoff: u32) -> (Vec<&RawRow>, Vec<&RawRow>) {
    raw_rows.iter().partition(|raw_row| raw_row.time < cutoff)
}

fn main() -> Result<(), &'static str> {
    let rows = [
        RawRow {
            value: Some(10.0),
            category: "red",
            group: "A",
            time: 1,
            label: 0,
        },
        RawRow {
            value: None,
            category: "blue",
            group: "A",
            time: 2,
            label: 1,
        },
        RawRow {
            value: Some(14.0),
            category: "red",
            group: "B",
            time: 3,
            label: 0,
        },
        RawRow {
            value: Some(100.0),
            category: "green",
            group: "C",
            time: 4,
            label: 1,
        },
    ];
    let (train, test) = group_split(&rows, "C");
    let training_rows: Vec<RawRow> = train.into_iter().copied().collect();
    let preprocessor = Preprocessor::fit(&training_rows)?;
    println!(
        "group split: train={}, test={}, fitted median={}",
        training_rows.len(),
        test.len(),
        preprocessor.median
    );
    println!(
        "held-out row features: {:?}; label stays separate={}",
        preprocessor.transform(*test[0])?,
        test[0].label
    );
    let (past, future) = time_split(&rows, 4);
    println!("time split: past={}, future={}", past.len(), future.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn held_out_rows_cannot_affect_fit() {
        let rows = [
            RawRow {
                value: Some(2.0),
                category: "a",
                group: "x",
                time: 1,
                label: 0,
            },
            RawRow {
                value: Some(4.0),
                category: "b",
                group: "x",
                time: 2,
                label: 1,
            },
            RawRow {
                value: Some(999.0),
                category: "secret",
                group: "hold",
                time: 3,
                label: 1,
            },
        ];
        let (train, test) = group_split(&rows, "hold");
        let training_rows: Vec<RawRow> = train.into_iter().copied().collect();
        let preprocessor = Preprocessor::fit(&training_rows).unwrap();
        assert_eq!(preprocessor.median, 3.0);
        assert_eq!(preprocessor.categories, ["a", "b"]);
        assert_eq!(
            preprocessor.transform(*test[0]).unwrap(),
            [999.0, 0.0, 0.0, 0.0]
        );
        let (_, future) = time_split(&rows, 3);
        assert_eq!(future.len(), 1);
    }
    #[test]
    fn missing_indicator_and_finite_even_median() {
        let raw_row = RawRow {
            value: Some(f64::MAX),
            category: "a",
            group: "x",
            time: 1,
            label: 0,
        };
        let fitted = Preprocessor::fit(&[raw_row, raw_row]).unwrap();
        assert_eq!(fitted.median, f64::MAX);
        assert_eq!(
            fitted
                .transform(RawRow {
                    value: None,
                    ..raw_row
                })
                .unwrap(),
            [f64::MAX, 1.0, 1.0]
        );
        assert!(Preprocessor::fit(&[RawRow {
            value: None,
            ..raw_row
        }])
        .is_err());
        assert!(fitted
            .transform(RawRow {
                value: Some(f64::NAN),
                ..raw_row
            })
            .is_err());
    }
}
