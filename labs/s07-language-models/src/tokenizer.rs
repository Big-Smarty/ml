//! Supplied serialization and boundary checks; learning algorithms live in ch34.rs.
use crate::{ensure, LabResult};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Bpe {
    pub rules: Vec<(u16, u16)>,
    pub vocab: Vec<Vec<u8>>,
}
impl Bpe {
    pub fn bytes() -> Self {
        Self {
            rules: vec![],
            vocab: (0..256).map(|x| vec![x as u8]).collect(),
        }
    }
    pub fn add(&mut self, pair: (u16, u16)) -> LabResult<u16> {
        ensure(self.vocab.len() < 1024, "teaching vocabulary limit is 1024")?;
        let a = self
            .vocab
            .get(pair.0 as usize)
            .ok_or("invalid left parent")?;
        let b = self
            .vocab
            .get(pair.1 as usize)
            .ok_or("invalid right parent")?;
        ensure(
            a.len() + b.len() <= 4096,
            "expanded token exceeds 4096 bytes",
        )?;
        let mut bytes = a.clone();
        bytes.extend_from_slice(b);
        let id = self.vocab.len() as u16;
        self.vocab.push(bytes);
        self.rules.push(pair);
        Ok(id)
    }
    pub fn serialize(&self) -> String {
        let mut out = String::from("S07BPE1\n");
        for (a, b) in &self.rules {
            out.push_str(&format!("{a} {b}\n"));
        }
        out
    }
    pub fn load(text: &str) -> LabResult<Self> {
        ensure(text.len() <= 16384, "tokenizer text exceeds 16 KiB")?;
        let mut lines = text.lines();
        ensure(lines.next() == Some("S07BPE1"), "invalid tokenizer version")?;
        let mut model = Self::bytes();
        for line in lines {
            let values: Vec<_> = line.split_whitespace().collect();
            ensure(values.len() == 2, "merge needs two IDs")?;
            let a = values[0].parse::<u16>().map_err(|e| e.to_string())?;
            let b = values[1].parse::<u16>().map_err(|e| e.to_string())?;
            model.add((a, b))?;
        }
        Ok(model)
    }
}
pub type Fit = fn(&[&[u8]], usize) -> LabResult<Bpe>;
pub type Encode = fn(&Bpe, &[u8]) -> Vec<u16>;
pub type Decode = fn(&Bpe, &[u16]) -> LabResult<Vec<u8>>;
pub fn run_with(args: &[String], fit: Fit, encode: Encode, decode: Decode) -> LabResult {
    let merges = match args {
        [] => 24,
        [flag, value] if flag == "--merges" => value.parse::<usize>().map_err(|e| e.to_string())?,
        _ => return Err("usage: 34 [--merges N]".into()),
    };
    ensure(merges <= 768, "at most 768 merges")?;
    let train = "naïve café 咖啡 — café 咖啡";
    let held = "café 🙂 unseen";
    let tokenizer = fit(&[train.as_bytes()], merges)?;
    let loaded = Bpe::load(&tokenizer.serialize())?;
    println!(
        "rules={:?}; vocabulary={}",
        loaded.rules,
        loaded.vocab.len()
    );
    for text in [train, held] {
        let ids = encode(&loaded, text.as_bytes());
        ensure(
            decode(&loaded, &ids)? == text.as_bytes(),
            "byte roundtrip failed",
        )?;
        println!(
            "{text:?}: scalars={} bytes={} tokens={} bytes/token={:.3} roundtrip=true",
            text.chars().count(),
            text.len(),
            ids.len(),
            text.len() as f64 / ids.len() as f64
        );
    }
    Ok(())
}
pub fn check_with(fit: Fit, encode: Encode, decode: Decode) -> LabResult {
    let t = fit(&[b"banana"], 1)?;
    ensure(t.rules==[(97,110)],format!("goal: banana's tied pair should select (97,110); got {:?}. Implement count/select/replace/repeat.",t.rules))?;
    ensure(
        encode(&t, b"banana") == [98, 256, 256, 97],
        "goal: ranked non-overlapping encoding",
    )?;
    let boundary = fit(&[b"ab", b"ab"], 8)?;
    ensure(
        boundary.rules == [(97, 98)],
        "goal: pair counts must not cross documents",
    )?;
    let overlaps = fit(&[b"aaaaa"], 1)?;
    ensure(
        encode(&overlaps, b"aaaaa") == [256, 256, 97],
        "goal: consume each input only once",
    )?;
    let trained = fit(&["café café café 🙂".as_bytes()], 12)?;
    let loaded = Bpe::load(&trained.serialize())?;
    for bytes in ["café 咖啡 🙂".as_bytes(), &[0, 255, 195, 169], b"", b"zzzz"] {
        let ids = encode(&loaded, bytes);
        ensure(
            decode(&loaded, &ids)? == bytes,
            "goal: exact arbitrary-byte roundtrip",
        )?;
        ensure(
            ids == encode(&trained, bytes),
            "goal: save/load preserves IDs",
        )?;
    }
    ensure(
        decode(&loaded, &[u16::MAX]).is_err(),
        "goal: reject unknown token IDs",
    )?;
    println!(
        "34 goal passed: full BPE, ties, overlap, boundaries, serialization and unfamiliar bytes"
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn bytes_are_working_and_solution_roundtrips() {
        super::run_with(
            &[],
            crate::ch34::fit,
            crate::ch34::encode,
            crate::ch34::decode,
        )
        .unwrap();
        super::check_with(
            crate::solutions::ch34::fit,
            crate::solutions::ch34::encode,
            crate::solutions::ch34::decode,
        )
        .unwrap();
        assert!(super::Bpe::load("S07BPE1\n256 97\n").is_err());
        assert!(super::Bpe::load("wrong").is_err());
    }
}
