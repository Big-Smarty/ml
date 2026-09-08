use std::{collections::HashMap, error::Error, fs, io::Write};
const MAX_MERGES: usize = u16::MAX as usize + 1 - 256;
const MAX_FILE_BYTES: u64 = 1_048_576;
const MAX_TOKEN_BYTES: usize = 1_048_576;
const MAX_TOTAL_VOCAB_BYTES: usize = 16_777_216;
const MAX_DECODE_BYTES: usize = 67_108_864;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Bpe {
    vocab: Vec<Vec<u8>>,
    merges: Vec<(u16, u16)>,
}
impl Bpe {
    fn train(text: &[u8], merges: usize) -> Result<Self, String> {
        if merges > MAX_MERGES {
            return Err("merge count exceeds u16 vocabulary capacity".to_owned());
        }
        let mut vocab = (0..=255).map(|b| vec![b as u8]).collect::<Vec<_>>();
        let mut words = vec![text.iter().map(|&b| b as u16).collect::<Vec<_>>()];
        let mut rules = Vec::new();
        for _ in 0..merges {
            let mut counts = HashMap::<(u16, u16), usize>::new();
            for w in &words {
                for p in w.windows(2) {
                    *counts.entry((p[0], p[1])).or_default() += 1;
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
            let id = vocab.len() as u16;
            let mut bytes = vocab[pair.0 as usize].clone();
            bytes.extend_from_slice(&vocab[pair.1 as usize]);
            vocab.push(bytes);
            rules.push(pair);
            for w in &mut words {
                *w = merge_pair(w, pair, id);
            }
        }
        Ok(Self {
            vocab,
            merges: rules,
        })
    }
    fn encode(&self, bytes: &[u8]) -> Vec<u16> {
        let mut ids = bytes.iter().map(|&b| b as u16).collect::<Vec<_>>();
        for (i, &pair) in self.merges.iter().enumerate() {
            ids = merge_pair(&ids, pair, (256 + i) as u16);
        }
        ids
    }
    fn decode(&self, ids: &[u16]) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        for &id in ids {
            let token = self
                .vocab
                .get(id as usize)
                .ok_or("token id outside vocabulary")?;
            if out
                .len()
                .checked_add(token.len())
                .filter(|&n| n <= MAX_DECODE_BYTES)
                .is_none()
            {
                return Err("decoded output exceeds safety limit".to_owned());
            }
            out.extend_from_slice(token);
        }
        Ok(out)
    }
    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let mut s = String::from("BYTE_BPE_V1\n");
        for &(a, b) in &self.merges {
            s.push_str(&format!("{a} {b}\n"));
        }
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(s.as_bytes())?;
        Ok(())
    }
    fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        if fs::metadata(path)?.len() > MAX_FILE_BYTES {
            return Err("tokenizer file exceeds safety limit".into());
        }
        let text = fs::read_to_string(path)?;
        let mut lines = text.lines();
        if lines.next() != Some("BYTE_BPE_V1") {
            return Err("bad tokenizer header".into());
        }
        let mut vocab = (0..=255).map(|b| vec![b as u8]).collect::<Vec<_>>();
        let mut merges = Vec::new();
        let mut total_bytes = 256usize;
        for line in lines {
            if vocab.len() > u16::MAX as usize {
                return Err("too many merge rules".into());
            }
            let mut p = line.split_whitespace();
            let a: u16 = p.next().ok_or("missing left id")?.parse()?;
            let b: u16 = p.next().ok_or("missing right id")?.parse()?;
            if p.next().is_some() || a as usize >= vocab.len() || b as usize >= vocab.len() {
                return Err("invalid merge rule".into());
            }
            let mut bytes = vocab[a as usize].clone();
            let new_len = bytes
                .len()
                .checked_add(vocab[b as usize].len())
                .ok_or("token length overflow")?;
            if new_len > MAX_TOKEN_BYTES {
                return Err("expanded token exceeds safety limit".into());
            }
            total_bytes = total_bytes
                .checked_add(new_len)
                .filter(|&n| n <= MAX_TOTAL_VOCAB_BYTES)
                .ok_or("expanded vocabulary exceeds safety limit")?;
            bytes.extend_from_slice(&vocab[b as usize]);
            vocab.push(bytes);
            merges.push((a, b));
        }
        Ok(Self { vocab, merges })
    }
}
fn merge_pair(ids: &[u16], pair: (u16, u16), new_id: u16) -> Vec<u16> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < ids.len() {
        if i + 1 < ids.len() && (ids[i], ids[i + 1]) == pair {
            out.push(new_id);
            i += 2
        } else {
            out.push(ids[i]);
            i += 1
        }
    }
    out
}
fn main() -> Result<(), Box<dyn Error>> {
    let text = "naïve café 咖啡 — café 咖啡";
    let bpe = Bpe::train(text.as_bytes(), 24)?;
    let ids = bpe.encode(text.as_bytes());
    let restored = String::from_utf8(bpe.decode(&ids)?)?;
    let path = std::env::temp_dir().join(format!("ch34-tokenizer-{}.txt", std::process::id()));
    bpe.save(path.to_str().ok_or("non-UTF-8 temporary path")?)?;
    let loaded = Bpe::load(path.to_str().ok_or("non-UTF-8 temporary path")?)?;
    println!(
        "{} UTF-8 bytes -> {} BPE tokens; vocabulary {}",
        text.len(),
        ids.len(),
        bpe.vocab.len()
    );
    fs::remove_file(path)?;
    println!("round trip: {}", restored == text);
    println!(
        "serialized tokenizer preserves encoding: {}",
        loaded.encode(text.as_bytes()) == ids
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_frequency_pairs_use_lexicographically_smallest_pair() {
        let b = Bpe::train(b"banana", 1).unwrap();
        assert_eq!(b.merges[0], (b'a' as u16, b'n' as u16));
    }

    #[test]
    fn multilingual_round_trip() {
        for s in ["hello", "café", "咖啡", "🙂a"] {
            let b = Bpe::train(s.as_bytes(), 10).unwrap();
            assert_eq!(b.decode(&b.encode(s.as_bytes())).unwrap(), s.as_bytes());
        }
    }
    #[test]
    fn serialization_is_stable() {
        let b = Bpe::train(b"banana banana", 8).unwrap();
        let p = std::env::temp_dir().join(format!("ch34-{}.txt", std::process::id()));
        b.save(p.to_str().unwrap()).unwrap();
        assert_eq!(Bpe::load(p.to_str().unwrap()).unwrap(), b);
        let original = fs::read(&p).unwrap();
        assert!(b.save(p.to_str().unwrap()).is_err());
        assert_eq!(fs::read(&p).unwrap(), original);
        fs::remove_file(&p).unwrap();
    }
    #[test]
    fn rejects_invalid_and_expanding_files() {
        let p = std::env::temp_dir().join(format!("ch34-bad-{}.txt", std::process::id()));
        fs::write(&p, "BYTE_BPE_V1\n999 0\n").unwrap();
        assert!(Bpe::load(p.to_str().unwrap()).is_err());
        fs::write(&p, "wrong\n").unwrap();
        assert!(Bpe::load(p.to_str().unwrap()).is_err());
        fs::remove_file(&p).unwrap();
        assert!(Bpe::train(b"x", MAX_MERGES + 1).is_err());
        let b = Bpe::train(b"abc", 0).unwrap();
        assert!(b.decode(&[999]).is_err());
        let mut doubling = String::from("BYTE_BPE_V1\n97 97\n");
        for id in 256..280 {
            doubling.push_str(&format!("{id} {id}\n"));
        }
        fs::write(&p, doubling).unwrap();
        assert!(Bpe::load(p.to_str().unwrap()).is_err());
        fs::remove_file(&p).unwrap();
    }
}
