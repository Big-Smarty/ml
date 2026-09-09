use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
};
pub const COURSE:&str="Title: Rust notes\nLicense: course-authored CC0\n\nRust makes ownership visible.  Rust makes errors visible.\n\n===DOC===\nTitle: Duplicate\nLicense: course-authored CC0\n\nRust makes ownership visible. Rust makes errors visible.\n\n===DOC===\nTitle: Model notes\nLicense: course-authored CC0\n\nA language model predicts the next token from earlier tokens.\n\n===DOC===\nTitle: Held-out prose\nLicense: course-authored CC0\n\nEvaluation asks whether a model predicts text it did not train on.";
#[derive(Clone, Debug)]
pub struct ParsedDocument {
    pub title: String,
    pub license: String,
    pub text: String,
    pub source: String,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CleanDocument {
    pub id: u64,
    pub title: String,
    pub license: String,
    pub text: String,
    pub source: String,
}
#[derive(Debug)]
pub struct SplitDocument {
    pub document: CleanDocument,
    pub split: &'static str,
}
pub type RemovedDuplicate = (CleanDocument, u64, f32);
pub type DedupResult = (Vec<CleanDocument>, Vec<RemovedDuplicate>);
pub fn hash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3)
    }
    h
}
pub fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
pub fn parse_documents(input: &str, source: &str) -> Result<Vec<ParsedDocument>, Box<dyn Error>> {
    let normalized = input.replace("\r\n", "\n");
    normalized
        .split("\n===DOC===\n")
        .map(|raw| {
            let mut lines = raw.trim().lines();
            let title = lines
                .next()
                .and_then(|x| x.strip_prefix("Title: "))
                .ok_or("document missing Title")?
                .to_owned();
            let license = lines
                .next()
                .and_then(|x| x.strip_prefix("License: "))
                .ok_or("document missing License")?
                .to_owned();
            if title.trim().is_empty() || license.trim().is_empty() {
                return Err("document title, license, and text must be nonempty".into());
            }
            Ok(ParsedDocument {
                title,
                license,
                text: lines.collect::<Vec<_>>().join("\n"),
                source: source.to_owned(),
            })
        })
        .collect()
}
pub fn clean_documents(
    documents: Vec<ParsedDocument>,
) -> Result<Vec<CleanDocument>, Box<dyn Error>> {
    let cleaned_documents = documents
        .into_iter()
        .map(|document| {
            let text = clean(&document.text);
            if text.is_empty() {
                return Err("document title, license, and text must be nonempty".into());
            }
            Ok(CleanDocument {
                id: hash(format!("{}\n{text}", document.title).as_bytes()),
                title: document.title,
                license: document.license,
                text,
                source: document.source,
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let mut identities = BTreeMap::new();
    for document in &cleaned_documents {
        if let Some(previous) = identities.insert(document.id, (&document.title, &document.text)) {
            if previous != (&document.title, &document.text) {
                return Err("document hash collision".into());
            }
        }
    }
    Ok(cleaned_documents)
}
pub fn shingles(text: &str) -> BTreeSet<Vec<String>> {
    let words = text
        .to_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    words.windows(3).map(|w| w.to_vec()).collect()
}
pub fn similarity(left: &str, right: &str) -> f32 {
    let a = shingles(left);
    let b = shingles(right);
    let union = a.union(&b).count();
    if union == 0 {
        f32::from(clean(left) == clean(right))
    } else {
        a.intersection(&b).count() as f32 / union as f32
    }
}
pub fn deduplicate(
    documents: Vec<CleanDocument>,
    threshold: f32,
) -> Result<DedupResult, Box<dyn Error>> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err("dedup threshold must be finite and within 0..=1".into());
    }
    let mut retained_documents: Vec<CleanDocument> = Vec::new();
    let mut removed = Vec::new();
    'outer: for document in documents {
        for prior in &retained_documents {
            let score = similarity(&document.text, &prior.text);
            if document.text == prior.text || score >= threshold {
                removed.push((document, prior.id, score));
                continue 'outer;
            }
        }
        retained_documents.push(document)
    }
    Ok((retained_documents, removed))
}
pub fn split(document_id: u64) -> &'static str {
    match document_id % 10 {
        0 => "test",
        1 => "validation",
        _ => "train",
    }
}
pub fn assign_splits(documents: Vec<CleanDocument>) -> Vec<SplitDocument> {
    documents
        .into_iter()
        .map(|document| SplitDocument {
            split: split(document.id),
            document,
        })
        .collect()
}
pub fn audit(documents: &[SplitDocument], removed: &[RemovedDuplicate]) -> String {
    let mut counts = BTreeMap::new();
    for document in documents {
        *counts.entry(document.split).or_insert(0usize) += 1;
    }
    let mut s = String::from("CORPUS_AUDIT_V1\n");
    for split_document in documents {
        let document = &split_document.document;
        s.push_str(&format!(
            "doc={:016x} split={} title={:?} license={:?} source={:?} bytes={}\n",
            document.id,
            split_document.split,
            document.title,
            document.license,
            document.source,
            document.text.len()
        ));
    }
    for (document, kept, score) in removed {
        s.push_str(&format!(
            "removed={:016x} duplicate_of={kept:016x} similarity={score:.3} title={:?} license={:?} source={:?} bytes={}\n",
            document.id,
            document.title,
            document.license,
            document.source,
            document.text.len()
        ));
    }
    s.push_str(&format!("counts={counts:?}\n"));
    s
}

pub type Prepare =
    fn(Vec<ParsedDocument>, f32) -> crate::LabResult<(Vec<SplitDocument>, Vec<RemovedDuplicate>)>;
pub fn run_with(args: &[String], prepare: Prepare) -> crate::LabResult {
    let mut path = None;
    let mut threshold = 0.8;
    let mut values = args.iter();
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--corpus" => path = Some(values.next().ok_or("--corpus needs PATH")?),
            "--threshold" => {
                threshold = values
                    .next()
                    .ok_or("--threshold needs VALUE")?
                    .parse::<f32>()
                    .map_err(|e| e.to_string())?
            }
            _ => return Err(format!("unknown option {flag}")),
        }
    }
    crate::ensure(
        threshold.is_finite() && (0.0..=1.0).contains(&threshold),
        "threshold must be in 0..=1",
    )?;
    let source = path.map_or("bundled CC0", String::as_str);
    let raw = if let Some(path) = path {
        crate::ensure(
            std::fs::metadata(path).map_err(|e| e.to_string())?.len() <= 65536,
            "fixture limit: 64 KiB",
        )?;
        std::fs::read_to_string(path).map_err(|e| e.to_string())?
    } else {
        COURSE.to_owned()
    };
    println!("whitespace recipe=v1, trigrams=lowercase sets, Jaccard threshold={threshold}");
    let parsed = parse_documents(&raw, source).map_err(|e| e.to_string())?;
    let count = parsed.len();
    let (docs, removed) = prepare(parsed, threshold)?;
    println!(
        "parsed={count} retained={} removed={}",
        docs.len(),
        removed.len()
    );
    print!("{}", audit(&docs, &removed));
    Ok(())
}
pub fn check_with(prepare: Prepare) -> crate::LabResult {
    let raw="Title: first\nLicense: CC0\n\none two three four five six\n===DOC===\nTitle: near copy\nLicense: CC0\n\none two three four five six seven\n===DOC===\nTitle: other\nLicense: CC0\n\na different short narrative\n===DOC===\nTitle: exact\nLicense: CC0\n\none  two three four five six";
    let parsed = parse_documents(raw, "held-out fixture").map_err(|e| e.to_string())?;
    let cleaned = clean_documents(parsed.clone()).map_err(|e| e.to_string())?;
    let (reference, reference_removed) = deduplicate(cleaned, 0.8).map_err(|e| e.to_string())?;
    let (docs, removed) = prepare(parsed, 0.8)?;
    crate::ensure(
        docs.len() == reference.len() && removed.len() == reference_removed.len(),
        "goal: near-duplicate decisions differ from the reference",
    )?;
    crate::ensure(docs.iter().zip(&reference).all(|(actual,expected)|actual.document==*expected) &&
        removed.iter().zip(&reference_removed).all(|((actual,id,score),(expected,expected_id,expected_score))|
            actual==expected && id==expected_id && (score-expected_score).abs()<1e-6),
        "goal: ordered retained records or removed (document, duplicate_of, similarity) differ from the reference")?;
    crate::ensure(docs.len()==2 && removed.len()==2,format!("goal: retain 2/remove 2 at Jaccard 0.8; got {}/{}. Implement shingle sets, near-duplicate decisions and removal links.",docs.len(),removed.len()))?;
    crate::ensure(
        removed.iter().all(|(d, id, _)| {
            d.source == "held-out fixture" && docs.iter().any(|p| p.document.id == *id)
        }),
        "goal: provenance or duplicate link was lost",
    )?;
    crate::ensure(
        docs.iter().all(|d| d.split == split(d.document.id)),
        "goal: whole-document stable split",
    )?;
    crate::ensure(
        prepare(
            parse_documents(raw, "fixture").map_err(|e| e.to_string())?,
            f32::NAN,
        )
        .is_err(),
        "goal: reject invalid threshold",
    )?;
    let empty = parse_documents("Title: empty\nLicense: CC0\n\n \n", "fixture")
        .map_err(|e| e.to_string())?;
    crate::ensure(
        prepare(empty, 0.8).is_err(),
        "goal: reject empty cleaned content",
    )?;
    println!("37 goal passed: cleaning, exact/near dedup, links, source, split and invalid inputs");
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn baseline_reference_and_solution() {
        super::run_with(&[], crate::ch37::prepare).unwrap();
        super::check_with(crate::solutions::ch37::prepare).unwrap();
        let parsed = super::parse_documents(super::COURSE, "fixture").unwrap();
        let clean = super::clean_documents(parsed).unwrap();
        let (docs, removed) = super::deduplicate(clean, 0.8).unwrap();
        assert_eq!(super::assign_splits(docs).len(), 3);
        assert_eq!(removed.len(), 1);
        assert!(super::parse_documents("Title: x\ntext", "fixture").is_err());
    }
}

#[cfg(test)]
mod cli_tests {
    #[test]
    fn threshold_flag_changes_the_actual_near_copy_decision() {
        let path = std::env::temp_dir().join(format!("s07-threshold-{}.txt", std::process::id()));
        std::fs::write(&path,"Title: first\nLicense: CC0\n\none two three four five six\n===DOC===\nTitle: near\nLicense: CC0\n\none two three four five six seven").unwrap();
        fn low(
            docs: Vec<super::ParsedDocument>,
            threshold: f32,
        ) -> crate::LabResult<(Vec<super::SplitDocument>, Vec<super::RemovedDuplicate>)> {
            assert_eq!(threshold, 0.5);
            let result = crate::solutions::ch37::prepare(docs, threshold)?;
            assert_eq!(result.0.len(), 1);
            assert_eq!(result.1.len(), 1);
            Ok(result)
        }
        fn high(
            docs: Vec<super::ParsedDocument>,
            threshold: f32,
        ) -> crate::LabResult<(Vec<super::SplitDocument>, Vec<super::RemovedDuplicate>)> {
            assert_eq!(threshold, 0.99);
            let result = crate::solutions::ch37::prepare(docs, threshold)?;
            assert_eq!(result.0.len(), 2);
            assert!(result.1.is_empty());
            Ok(result)
        }
        let args = |threshold: &str| {
            vec![
                "--corpus".to_owned(),
                path.to_string_lossy().into_owned(),
                "--threshold".to_owned(),
                threshold.to_owned(),
            ]
        };
        super::run_with(&args("0.5"), low).unwrap();
        super::run_with(&args("0.99"), high).unwrap();
        std::fs::remove_file(path).unwrap();
    }
}

#[cfg(test)]
mod goal_regression_tests {
    #[test]
    fn wrong_survivor_link_cannot_pass_on_counts_alone() {
        fn wrong_link(
            docs: Vec<super::ParsedDocument>,
            threshold: f32,
        ) -> crate::LabResult<(Vec<super::SplitDocument>, Vec<super::RemovedDuplicate>)> {
            let (docs, mut removed) = crate::solutions::ch37::prepare(docs, threshold)?;
            if let (Some(last), Some(removal)) = (docs.last(), removed.first_mut()) {
                removal.1 = last.document.id;
            }
            Ok((docs, removed))
        }
        let error = super::check_with(wrong_link).unwrap_err();
        assert!(error.starts_with("GOAL_NOT_MET:"));
        assert!(error.contains("ordered retained"));
    }
}
