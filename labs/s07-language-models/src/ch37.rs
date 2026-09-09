//! Learner: extend exact-text deduplication into a transparent near-duplicate document pipeline.
use crate::{
    corpus::{self, ParsedDocument, RemovedDuplicate, SplitDocument},
    LabResult,
};
pub fn prepare(
    documents: Vec<ParsedDocument>,
    threshold: f32,
) -> LabResult<(Vec<SplitDocument>, Vec<RemovedDuplicate>)> {
    crate::ensure(
        threshold.is_finite() && (0.0..=1.0).contains(&threshold),
        "threshold must be in 0..=1",
    )?;
    let clean = corpus::clean_documents(documents).map_err(|e| e.to_string())?;
    let mut retained: Vec<corpus::CleanDocument> = Vec::new();
    let mut removed = Vec::new();
    for doc in clean {
        if let Some(prior) = retained.iter().find(|p| p.text == doc.text) {
            removed.push((doc, prior.id, 1.));
        } else {
            retained.push(doc);
        }
    }
    Ok((corpus::assign_splits(retained), removed))
}
pub fn run(args: &[String]) -> LabResult {
    corpus::run_with(args, prepare)
}
pub fn check() -> LabResult {
    corpus::check_with(prepare)
}
