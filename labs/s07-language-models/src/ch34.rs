//! Learner: implement complete deterministic document-aware BPE. I/O stays supplied.
use crate::{
    tokenizer::{self, Bpe},
    LabResult,
};
pub fn fit(documents: &[&[u8]], merges: usize) -> LabResult<Bpe> {
    crate::ensure(
        merges <= 768 && documents.iter().map(|d| d.len()).sum::<usize>() <= 65536,
        "tiny fitter limit: 768 merges, 64 KiB",
    )?;
    // Working baseline is an exact byte tokenizer with no learned merges.
    Ok(Bpe::bytes())
}
pub fn encode(_tokenizer: &Bpe, bytes: &[u8]) -> Vec<u16> {
    bytes.iter().map(|&b| u16::from(b)).collect()
}
pub fn decode(tokenizer: &Bpe, ids: &[u16]) -> LabResult<Vec<u8>> {
    let mut bytes = Vec::new();
    for &id in ids {
        crate::ensure(bytes.len() <= 1_048_576, "decode exceeds 1 MiB")?;
        bytes.extend_from_slice(tokenizer.vocab.get(id as usize).ok_or("invalid token ID")?);
    }
    crate::ensure(bytes.len() <= 1_048_576, "decode exceeds 1 MiB")?;
    Ok(bytes)
}
pub fn run(args: &[String]) -> LabResult {
    tokenizer::run_with(args, fit, encode, decode)
}
pub fn check() -> LabResult {
    tokenizer::check_with(fit, encode, decode)
}
