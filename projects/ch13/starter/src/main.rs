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
    // TODO: sort present training values and return the odd or even median.
    let _ = values;
    todo!("sort the values and return their median, or None when empty")
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

fn main() -> Result<(), &'static str> {
    // An already-fitted checkpoint keeps `cargo run` useful while `fit` is the guided task.
    let _guided_fit: fn(&[RawRow]) -> Result<Preprocessor, &'static str> = Preprocessor::fit;
    let preprocessor = Preprocessor {
        median: 4.0,
        categories: vec!["blue".to_owned(), "red".to_owned()],
    };
    let raw_row = RawRow {
        value: None,
        category: "green",
        group: "held-out",
        time: 5,
        label: 1,
    };
    println!(
        "raw group={}, time={} -> encoded features {:?}; class label stays separate={}",
        raw_row.group,
        raw_row.time,
        preprocessor.transform(raw_row)?,
        raw_row.label
    );
    Ok(())
}

#[test]
fn fit_stores_training_median_and_vocabulary() {
    let training_rows = [
        RawRow {
            value: Some(2.0),
            category: "red",
            group: "train",
            time: 1,
            label: 0,
        },
        RawRow {
            value: None,
            category: "blue",
            group: "train",
            time: 2,
            label: 1,
        },
        RawRow {
            value: Some(6.0),
            category: "red",
            group: "train",
            time: 3,
            label: 0,
        },
        RawRow {
            value: Some(4.0),
            category: "blue",
            group: "train",
            time: 4,
            label: 1,
        },
    ];
    let preprocessor = Preprocessor::fit(&training_rows).unwrap();
    assert_eq!(preprocessor.median, 4.0);
    assert_eq!(preprocessor.categories, ["blue", "red"]);
    assert_eq!(median(&mut [2.0, 4.0]), Some(3.0));
    assert_eq!(median(&mut []), None);
    assert!(Preprocessor::fit(&[RawRow {
        value: None,
        ..training_rows[0]
    }])
    .is_err());
    assert_eq!(
        preprocessor
            .transform(RawRow {
                value: None,
                category: "green",
                group: "held-out",
                time: 5,
                label: 1,
            })
            .unwrap(),
        [4.0, 1.0, 0.0, 0.0]
    );
}
