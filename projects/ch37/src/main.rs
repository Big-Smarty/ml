use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::Path,
};
const COURSE:&str="Title: Rust notes\nLicense: course-authored CC0\n\nRust makes ownership visible.  Rust makes errors visible.\n\n===DOC===\nTitle: Duplicate\nLicense: course-authored CC0\n\nRust makes ownership visible. Rust makes errors visible.\n\n===DOC===\nTitle: Model notes\nLicense: course-authored CC0\n\nA language model predicts the next token from earlier tokens.\n\n===DOC===\nTitle: Held-out prose\nLicense: course-authored CC0\n\nEvaluation asks whether a model predicts text it did not train on.";
#[derive(Clone, Debug)]
struct ParsedDocument {
    title: String,
    license: String,
    text: String,
    source: String,
}
#[derive(Clone, Debug)]
struct CleanDocument {
    id: u64,
    title: String,
    license: String,
    text: String,
    source: String,
}
#[derive(Debug)]
struct SplitDocument {
    document: CleanDocument,
    split: &'static str,
}
type RemovedDuplicate = (CleanDocument, u64, f32);
type DedupResult = (Vec<CleanDocument>, Vec<RemovedDuplicate>);
fn hash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3)
    }
    h
}
fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn parse_documents(input: &str, source: &str) -> Result<Vec<ParsedDocument>, Box<dyn Error>> {
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
fn clean_documents(documents: Vec<ParsedDocument>) -> Result<Vec<CleanDocument>, Box<dyn Error>> {
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
fn shingles(text: &str) -> BTreeSet<Vec<String>> {
    let words = text
        .to_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    words.windows(3).map(|w| w.to_vec()).collect()
}
fn similarity(left: &str, right: &str) -> f32 {
    let a = shingles(left);
    let b = shingles(right);
    let union = a.union(&b).count();
    if union == 0 {
        f32::from(clean(left) == clean(right))
    } else {
        a.intersection(&b).count() as f32 / union as f32
    }
}
fn deduplicate(
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
fn split(document_id: u64) -> &'static str {
    match document_id % 10 {
        0 => "test",
        1 => "validation",
        _ => "train",
    }
}
fn assign_splits(documents: Vec<CleanDocument>) -> Vec<SplitDocument> {
    documents
        .into_iter()
        .map(|document| SplitDocument {
            split: split(document.id),
            document,
        })
        .collect()
}
fn audit(documents: &[SplitDocument], removed: &[RemovedDuplicate]) -> String {
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
fn main() -> Result<(), Box<dyn Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    let (raw, source) = if let Some(path) = args.get(1) {
        (
            fs::read_to_string(path)?,
            Path::new(path).canonicalize()?.display().to_string(),
        )
    } else {
        (
            COURSE.to_owned(),
            "bundled course-authored fixture".to_owned(),
        )
    };
    let parsed_documents = parse_documents(&raw, &source)?;
    let original = parsed_documents.len();
    let cleaned_documents = clean_documents(parsed_documents)?;
    let (unique_documents, removed) = deduplicate(cleaned_documents, 0.8)?;
    let split_documents = assign_splits(unique_documents);
    let report = audit(&split_documents, &removed);
    println!(
        "parsed {original} documents; retained {}; removed {} duplicates",
        split_documents.len(),
        removed.len()
    );
    print!("{report}");
    if let Some(path) = args.get(2) {
        fs::write(path, report)?;
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn windows_newlines_and_removed_provenance_are_preserved() {
        let parsed = parse_documents(&COURSE.replace('\n', "\r\n"), "windows source").unwrap();
        let cleaned = clean_documents(parsed).unwrap();
        let (documents, removed) = deduplicate(cleaned, 0.8).unwrap();
        let split_documents = assign_splits(documents);
        assert_eq!(split_documents.len(), 3);
        let report = audit(&split_documents, &removed);
        let line = report.lines().find(|l| l.starts_with("removed=")).unwrap();
        assert!(
            line.contains("title=\"Duplicate\"")
                && line.contains("license=\"course-authored CC0\"")
                && line.contains("source=\"windows source\"")
        );
        assert!(
            (similarity(
                "one two three four five six",
                "one two three four five six seven"
            ) - 0.8)
                .abs()
                < 1e-6
        );
        assert_eq!(similarity("one two", "three four"), 0.0);
    }

    #[test]
    fn cleaning_and_dedup_are_document_level() {
        let parsed = parse_documents(COURSE, "fixture").unwrap();
        assert!(parsed[0].text.contains("  "));
        let cleaned = clean_documents(parsed).unwrap();
        assert!(!cleaned[0].text.contains("  "));
        let (documents, removed) = deduplicate(cleaned, 0.8).unwrap();
        assert_eq!(documents.len(), 3);
        assert_eq!(removed.len(), 1);
        let ids = documents
            .iter()
            .map(|document| document.id)
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), documents.len());
        let split_documents = assign_splits(documents);
        assert!(split_documents
            .iter()
            .all(|document| document.split == split(document.document.id)));
    }
    #[test]
    fn manifest_keeps_provenance() {
        let parsed = parse_documents(COURSE, "fixture").unwrap();
        let cleaned = clean_documents(parsed).unwrap();
        let documents = assign_splits(cleaned);
        let text = audit(&documents, &[]);
        assert!(text.contains("license=\"course-authored CC0\""));
        assert!(text.contains("source=\"fixture\""));
    }
    #[test]
    fn rejects_empty_metadata_and_invalid_threshold() {
        assert!(parse_documents("Title: \nLicense: CC0\n\ntext", "fixture").is_err());
        let parsed = parse_documents("Title: Empty\nLicense: CC0\n\n \n", "fixture").unwrap();
        assert!(clean_documents(parsed).is_err());
        let parsed = parse_documents(COURSE, "fixture").unwrap();
        assert!(deduplicate(clean_documents(parsed).unwrap(), f32::NAN).is_err());
    }
}
