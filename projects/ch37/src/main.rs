use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::Path,
};
const COURSE:&str="Title: Rust notes\nLicense: course-authored CC0\n\nRust makes ownership visible.  Rust makes errors visible.\n\n===DOC===\nTitle: Duplicate\nLicense: course-authored CC0\n\nRust makes ownership visible. Rust makes errors visible.\n\n===DOC===\nTitle: Model notes\nLicense: course-authored CC0\n\nA language model predicts the next token from earlier tokens.\n\n===DOC===\nTitle: Held-out prose\nLicense: course-authored CC0\n\nEvaluation asks whether a model predicts text it did not train on.";
#[derive(Clone, Debug)]
struct Doc {
    id: u64,
    title: String,
    license: String,
    text: String,
    source: String,
}
type DedupResult = (Vec<Doc>, Vec<(Doc, u64, f32)>);
fn hash(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3)
    }
    h
}
fn clean(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}
fn parse(input: &str, source: &str) -> Result<Vec<Doc>, Box<dyn Error>> {
    let normalized = input.replace("\r\n", "\n");
    let docs: Vec<Doc> = normalized
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
            let text = clean(&lines.collect::<Vec<_>>().join(" "));
            if title.trim().is_empty() || license.trim().is_empty() || text.is_empty() {
                return Err("document title, license, and text must be nonempty".into());
            }
            Ok(Doc {
                id: hash(format!("{title}\n{text}").as_bytes()),
                title,
                license,
                text,
                source: source.to_owned(),
            })
        })
        .collect::<Result<_, Box<dyn Error>>>()?;
    let mut identities = BTreeMap::new();
    for doc in &docs {
        if let Some(previous) = identities.insert(doc.id, (&doc.title, &doc.text)) {
            if previous != (&doc.title, &doc.text) {
                return Err("document hash collision".into());
            }
        }
    }
    Ok(docs)
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
fn dedup(docs: Vec<Doc>, threshold: f32) -> Result<DedupResult, Box<dyn Error>> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err("dedup threshold must be finite and within 0..=1".into());
    }
    let mut keep: Vec<Doc> = Vec::new();
    let mut removed = Vec::new();
    'outer: for d in docs {
        for prior in &keep {
            let score = similarity(&d.text, &prior.text);
            if d.text == prior.text || score >= threshold {
                removed.push((d, prior.id, score));
                continue 'outer;
            }
        }
        keep.push(d)
    }
    Ok((keep, removed))
}
fn split(id: u64) -> &'static str {
    match id % 10 {
        0 => "test",
        1 => "validation",
        _ => "train",
    }
}
fn audit(docs: &[Doc], removed: &[(Doc, u64, f32)]) -> String {
    let mut counts = BTreeMap::new();
    for d in docs {
        *counts.entry(split(d.id)).or_insert(0usize) += 1;
    }
    let mut s = String::from("CORPUS_AUDIT_V1\n");
    for d in docs {
        s.push_str(&format!(
            "doc={:016x} split={} title={:?} license={:?} source={:?} bytes={}\n",
            d.id,
            split(d.id),
            d.title,
            d.license,
            d.source,
            d.text.len()
        ));
    }
    for (doc, kept, score) in removed {
        s.push_str(&format!(
            "removed={:016x} duplicate_of={kept:016x} similarity={score:.3} title={:?} license={:?} source={:?} bytes={}\n",
            doc.id, doc.title, doc.license, doc.source, doc.text.len()
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
    let docs = parse(&raw, &source)?;
    let original = docs.len();
    let (docs, removed) = dedup(docs, 0.8)?;
    let report = audit(&docs, &removed);
    println!(
        "parsed {original} documents; retained {}; removed {} duplicates",
        docs.len(),
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
        let (docs, removed) = dedup(
            parse(&COURSE.replace('\n', "\r\n"), "windows source").unwrap(),
            0.8,
        )
        .unwrap();
        assert_eq!(docs.len(), 3);
        let report = audit(&docs, &removed);
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
        let docs = parse(COURSE, "fixture").unwrap();
        let (d, r) = dedup(docs, 0.8).unwrap();
        assert_eq!(d.len(), 3);
        assert_eq!(r.len(), 1);
        let ids = d.iter().map(|x| x.id).collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), d.len());
    }
    #[test]
    fn manifest_keeps_provenance() {
        let d = parse(COURSE, "fixture").unwrap();
        let text = audit(&d, &[]);
        assert!(text.contains("license=\"course-authored CC0\""));
        assert!(text.contains("source=\"fixture\""));
    }
    #[test]
    fn rejects_empty_metadata_and_invalid_threshold() {
        assert!(parse("Title: \nLicense: CC0\n\ntext", "fixture").is_err());
        assert!(dedup(parse(COURSE, "fixture").unwrap(), f32::NAN).is_err());
    }
}
