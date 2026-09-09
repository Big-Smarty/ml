//! Count pairs inside documents, select the smallest tied pair, replace without overlap,
//! then recount. Encoding replays ranks; decoding concatenates bytes before UTF-8 display.
use crate::{
    tokenizer::{self, Bpe},
    LabResult,
};
use std::collections::BTreeMap;
fn merge(ids: &[u16], pair: (u16, u16), new: u16) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            out.push(new);
            i += 2;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    out
}
pub fn fit(documents: &[&[u8]], merges: usize) -> LabResult<Bpe> {
    crate::ensure(
        merges <= 768 && documents.iter().map(|d| d.len()).sum::<usize>() <= 65536,
        "tiny fitter limit: 768 merges, 64 KiB",
    )?;
    let mut model = Bpe::bytes();
    let mut docs: Vec<Vec<u16>> = documents
        .iter()
        .map(|d| d.iter().map(|&b| u16::from(b)).collect())
        .collect();
    // ponytail: O(merges * bytes) rescans; index pair occurrences for a measured larger corpus.
    for _ in 0..merges {
        let mut counts = BTreeMap::new();
        for doc in &docs {
            for pair in doc.windows(2) {
                *counts.entry((pair[0], pair[1])).or_insert(0usize) += 1;
            }
        }
        let Some((&pair, &count)) = counts
            .iter()
            .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
        else {
            break;
        };
        if count < 2 {
            break;
        }
        let id = model.add(pair)?;
        for doc in &mut docs {
            *doc = merge(doc, pair, id);
        }
    }
    Ok(model)
}
pub fn encode(tokenizer: &Bpe, bytes: &[u8]) -> Vec<u16> {
    let mut ids: Vec<u16> = bytes.iter().map(|&b| u16::from(b)).collect();
    for (rank, &pair) in tokenizer.rules.iter().enumerate() {
        ids = merge(&ids, pair, (256 + rank) as u16);
    }
    ids
}
pub fn decode(tokenizer: &Bpe, ids: &[u16]) -> LabResult<Vec<u8>> {
    let mut bytes = Vec::new();
    for &id in ids {
        let token = tokenizer.vocab.get(id as usize).ok_or("invalid token ID")?;
        crate::ensure(
            bytes.len() + token.len() <= 1_048_576,
            "decode exceeds 1 MiB",
        )?;
        bytes.extend_from_slice(token);
    }
    Ok(bytes)
}
pub fn run(args: &[String]) -> LabResult {
    tokenizer::run_with(args, fit, encode, decode)
}
pub fn check() -> LabResult {
    tokenizer::check_with(fit, encode, decode)
}
