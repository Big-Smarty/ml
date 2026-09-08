//! Leakage-safe fitting of missing-value and categorical preprocessing.
#[derive(Clone, Copy, Debug)]
struct Row {
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
impl Preprocessor {
    fn fit(rows: &[Row]) -> Result<Self, &'static str> {
        if rows.is_empty() {
            return Err("fit needs training rows");
        }
        let mut values: Vec<f64> = rows.iter().filter_map(|r| r.value).collect();
        if values.is_empty() || values.iter().any(|x| !x.is_finite()) {
            return Err("training numeric values must include a finite value");
        }
        values.sort_by(f64::total_cmp);
        let median = if values.len() % 2 == 1 {
            values[values.len() / 2]
        } else {
            values[values.len() / 2 - 1].midpoint(values[values.len() / 2])
        };
        let mut categories: Vec<String> = rows.iter().map(|r| r.category.to_owned()).collect();
        categories.sort();
        categories.dedup();
        Ok(Self { median, categories })
    }
    fn transform(&self, row: Row) -> Result<Vec<f64>, &'static str> {
        let value = row.value.unwrap_or(self.median);
        if !value.is_finite() {
            return Err("numeric values must be finite");
        }
        let mut features = vec![value, row.value.is_none() as u8 as f64];
        features.extend(
            self.categories
                .iter()
                .map(|c| (c == row.category) as u8 as f64),
        );
        Ok(features)
    }
}

fn group_split<'a>(rows: &'a [Row], held_out_group: &str) -> (Vec<&'a Row>, Vec<&'a Row>) {
    rows.iter().partition(|r| r.group != held_out_group)
}
fn time_split(rows: &[Row], cutoff: u32) -> (Vec<&Row>, Vec<&Row>) {
    rows.iter().partition(|r| r.time < cutoff)
}

fn main() -> Result<(), &'static str> {
    let rows = [
        Row {
            value: Some(10.0),
            category: "red",
            group: "A",
            time: 1,
            label: 0,
        },
        Row {
            value: None,
            category: "blue",
            group: "A",
            time: 2,
            label: 1,
        },
        Row {
            value: Some(14.0),
            category: "red",
            group: "B",
            time: 3,
            label: 0,
        },
        Row {
            value: Some(100.0),
            category: "green",
            group: "C",
            time: 4,
            label: 1,
        },
    ];
    let (train, test) = group_split(&rows, "C");
    let train_rows: Vec<Row> = train.into_iter().copied().collect();
    let pre = Preprocessor::fit(&train_rows)?;
    println!(
        "group split: train={}, test={}, fitted median={}",
        train_rows.len(),
        test.len(),
        pre.median
    );
    println!(
        "held-out row features: {:?}; label stays separate={}",
        pre.transform(*test[0])?,
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
            Row {
                value: Some(2.0),
                category: "a",
                group: "x",
                time: 1,
                label: 0,
            },
            Row {
                value: Some(4.0),
                category: "b",
                group: "x",
                time: 2,
                label: 1,
            },
            Row {
                value: Some(999.0),
                category: "secret",
                group: "hold",
                time: 3,
                label: 1,
            },
        ];
        let (train, test) = group_split(&rows, "hold");
        let owned: Vec<Row> = train.into_iter().copied().collect();
        let p = Preprocessor::fit(&owned).unwrap();
        assert_eq!(p.median, 3.0);
        assert_eq!(p.categories, ["a", "b"]);
        assert_eq!(p.transform(*test[0]).unwrap(), [999.0, 0.0, 0.0, 0.0]);
        let (_, future) = time_split(&rows, 3);
        assert_eq!(future.len(), 1);
    }
    #[test]
    fn missing_indicator_and_finite_even_median() {
        let row = Row {
            value: Some(f64::MAX),
            category: "a",
            group: "x",
            time: 1,
            label: 0,
        };
        let fitted = Preprocessor::fit(&[row, row]).unwrap();
        assert_eq!(fitted.median, f64::MAX);
        assert_eq!(
            fitted.transform(Row { value: None, ..row }).unwrap(),
            [f64::MAX, 1.0, 1.0]
        );
        assert!(Preprocessor::fit(&[Row { value: None, ..row }]).is_err());
        assert!(fitted
            .transform(Row {
                value: Some(f64::NAN),
                ..row
            })
            .is_err());
    }
}
