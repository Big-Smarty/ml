//! A tiny local RAG pipeline: BM25, trained embeddings, exact search, context, extractive answering, attribution, and abstention.
use std::collections::{BTreeSet, HashMap};

#[derive(Clone)]
struct Fact {
    id: &'static str,
    text: &'static str,
}
const FACTS: [Fact; 3] = [
    Fact {
        id: "planet",
        text: "Mars has two moons named Phobos and Deimos.",
    },
    Fact {
        id: "france",
        text: "Paris is the capital city of France.",
    },
    Fact {
        id: "rust",
        text: "Rust ownership enables memory safety without a garbage collector.",
    },
];
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|x| !x.is_empty())
        .map(|x| x.to_ascii_lowercase())
        .collect()
}
fn content_words(s: &str) -> Vec<String> {
    words(s)
        .into_iter()
        .filter(|w| !["a", "and", "is", "of", "the", "what", "which", "who"].contains(&w.as_str()))
        .collect()
}

fn bm25(query: &str, docs: &[Fact]) -> Vec<f32> {
    let q = content_words(query);
    let tokenized: Vec<Vec<String>> = docs.iter().map(|d| words(d.text)).collect();
    let avg = tokenized.iter().map(Vec::len).sum::<usize>() as f32 / docs.len() as f32;
    tokenized
        .iter()
        .map(|doc| {
            q.iter()
                .map(|term| {
                    let n = tokenized.iter().filter(|d| d.contains(term)).count() as f32;
                    let idf = ((docs.len() as f32 - n + 0.5) / (n + 0.5) + 1.0).ln();
                    let tf = doc.iter().filter(|w| *w == term).count() as f32;
                    let denom = tf + 1.2 * (1.0 - 0.75 + 0.75 * doc.len() as f32 / avg);
                    idf * tf * 2.2 / denom
                })
                .sum()
        })
        .collect()
}
fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}
fn cross_entropy_from_logits(logits: &[f32], target: usize) -> f32 {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (maximum - logits[target])
        + logits
            .iter()
            .map(|logit| (logit - maximum).exp())
            .sum::<f32>()
            .ln()
}
fn argmax(x: &[f32]) -> usize {
    (1..x.len()).fold(0, |best, i| if x[i] > x[best] { i } else { best })
}

#[derive(Clone)]
struct DenseRetriever {
    ids: HashMap<String, usize>,
    query_embeddings: Vec<f32>,
    document_embeddings: Vec<f32>,
    dim: usize,
    docs: Vec<Vec<usize>>,
}
impl DenseRetriever {
    fn new(facts: &[Fact], train_data: &[(&str, usize)], dim: usize) -> Self {
        let mut vocab = BTreeSet::new();
        for f in facts {
            vocab.extend(words(f.text));
        }
        for (query, _) in train_data {
            vocab.extend(words(query));
        }
        let ids: HashMap<_, _> = vocab.into_iter().enumerate().map(|(i, w)| (w, i)).collect();
        let n = ids.len();
        let init = |salt: usize| {
            (0..n * dim)
                .map(|i| (((i * 37 + salt) % 101) as f32 - 50.0) / 500.0)
                .collect()
        };
        let docs = facts
            .iter()
            .map(|f| {
                words(f.text)
                    .iter()
                    .filter_map(|w| ids.get(w).copied())
                    .collect()
            })
            .collect();
        Self {
            ids,
            query_embeddings: init(3),
            document_embeddings: init(19),
            dim,
            docs,
        }
    }
    fn encode_with(&self, text: &str, table: &[f32]) -> Vec<f32> {
        let ids: Vec<_> = words(text)
            .iter()
            .filter_map(|w| self.ids.get(w).copied())
            .collect();
        if ids.is_empty() {
            return vec![0.0; self.dim];
        }
        let mut out = vec![0.0; self.dim];
        for id in &ids {
            for k in 0..self.dim {
                out[k] += table[id * self.dim + k];
            }
        }
        for x in &mut out {
            *x /= ids.len() as f32;
        }
        out
    }
    fn doc_vectors(&self) -> Vec<Vec<f32>> {
        self.docs
            .iter()
            .map(|ids| {
                let mut v = vec![0.0; self.dim];
                for id in ids {
                    for (k, value) in v.iter_mut().enumerate() {
                        *value += self.document_embeddings[id * self.dim + k];
                    }
                }
                for x in &mut v {
                    *x /= ids.len() as f32;
                }
                v
            })
            .collect()
    }
    fn scores(&self, query: &str) -> Vec<f32> {
        let query_vector = self.encode_with(query, &self.query_embeddings);
        self.doc_vectors()
            .iter()
            .map(|document_vector| dot(&query_vector, document_vector))
            .collect()
    }
    fn loss(&self, data: &[(&str, usize)]) -> f32 {
        data.iter()
            .map(|(query, target)| cross_entropy_from_logits(&self.scores(query), *target))
            .sum::<f32>()
            / data.len() as f32
    }
    fn train(&mut self, train_data: &[(&str, usize)], steps: usize, learning_rate: f32) {
        for step in 0..steps {
            let (text, target) = train_data[step % train_data.len()];
            let qids: Vec<_> = words(text)
                .iter()
                .filter_map(|w| self.ids.get(w).copied())
                .collect();
            let query_vector = self.encode_with(text, &self.query_embeddings);
            let document_vectors = self.doc_vectors();
            let scores: Vec<f32> = document_vectors
                .iter()
                .map(|document_vector| dot(&query_vector, document_vector))
                .collect();
            let m = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut probabilities: Vec<f32> = scores.iter().map(|x| (x - m).exp()).collect();
            let z = probabilities.iter().sum::<f32>();
            probabilities.iter_mut().for_each(|x| *x /= z);
            let mut logit_gradients = probabilities;
            logit_gradients[target] -= 1.0;
            let mut dq = vec![0.0; self.dim];
            for (j, document_vector) in document_vectors.iter().enumerate() {
                for (value, &document_value) in dq.iter_mut().zip(document_vector) {
                    *value += logit_gradients[j] * document_value;
                }
            }
            for id in &qids {
                for (k, &gradient) in dq.iter().enumerate() {
                    self.query_embeddings[id * self.dim + k] -=
                        learning_rate * gradient / qids.len() as f32;
                }
            }
            for (j, ids) in self.docs.iter().enumerate() {
                for id in ids {
                    for (k, &query_value) in query_vector.iter().enumerate() {
                        self.document_embeddings[id * self.dim + k] -=
                            learning_rate * logit_gradients[j] * query_value / ids.len() as f32;
                    }
                }
            }
        }
    }
}

struct Answer {
    text: String,
    source: String,
    context: String,
}
fn should_abstain(best: f32, second: f32, known_terms: usize) -> bool {
    // ponytail: fixture thresholds do not establish support; calibrate on held-out answerable and unanswerable queries.
    known_terms == 0 || best < 0.05 || best - second < 0.02
}
fn rag(query: &str, retriever: &DenseRetriever, facts: &[Fact]) -> Option<Answer> {
    let scores = retriever.scores(query);
    let best = argmax(&scores);
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let known_terms = content_words(query)
        .iter()
        .filter(|word| retriever.ids.contains_key(*word))
        .count();
    if should_abstain(
        sorted[0],
        sorted.get(1).copied().unwrap_or(f32::NEG_INFINITY),
        known_terms,
    ) {
        return None;
    }
    let fact = &facts[best];
    Some(Answer {
        text: fact.text.to_string(),
        source: fact.id.into(),
        context: format!("[{}] {}", fact.id, fact.text),
    })
}

fn main() {
    let train_data = [
        ("red planet satellites", 0),
        ("martian natural satellites", 0),
        ("french seat government", 1),
        ("city governing france", 1),
        ("safe systems language ownership", 2),
        ("memory management without gc", 2),
    ];
    let mut dense = DenseRetriever::new(&FACTS, &train_data, 8);
    let before_loss = dense.loss(&train_data);
    dense.train(&train_data, 900, 0.08);
    let after_loss = dense.loss(&train_data);
    println!("dense training cross-entropy: {before_loss:.4} -> {after_loss:.4}");
    let held_out = [
        ("two satellites", 0),
        ("city france government", 1),
        ("ownership garbage safety", 2),
    ];
    let hits = held_out
        .iter()
        .filter(|(q, want)| argmax(&dense.scores(q)) == *want)
        .count();
    println!("dense held-out retrieval hit@1: {hits}/{}", held_out.len());
    let queries = [
        ("Which satellites orbit the red planet?", Some("planet")),
        ("What is the French capital?", Some("france")),
        ("Who won the lunar chess final?", None),
    ];
    let mut answered = 0;
    let mut correct = 0;
    let mut attributed = 0;
    for (query, expected) in queries {
        let lexical = bm25(query, &FACTS);
        println!(
            "query: {query}\n  BM25 top={} scores={lexical:?}",
            FACTS[argmax(&lexical)].id
        );
        match rag(query, &dense, &FACTS) {
            Some(a) => {
                answered += 1;
                correct += usize::from(Some(a.source.as_str()) == expected);
                if a.context.starts_with(&format!("[{}]", a.source)) {
                    attributed += 1;
                }
                println!(
                    "  context={}\n  extracted={} [{}]",
                    a.context, a.text, a.source
                )
            }
            None => println!("  abstain: retrieved evidence is unsupported or ambiguous"),
        }
    }
    println!("grounded evaluation: dense hit@1={hits}/3, correct supported source={correct}/2, answered={answered}/3, answers with matching source label={attributed}/{answered}, unsupported abstained={}/1",usize::from(rag(queries[2].0,&dense,&FACTS).is_none()));
}

#[cfg(test)]
mod tests {
    use super::*;
    fn trained() -> DenseRetriever {
        let train_data = [
            ("red planet satellites", 0),
            ("french seat government", 1),
            ("safe systems language ownership", 2),
        ];
        let mut r = DenseRetriever::new(&FACTS, &train_data, 8);
        r.train(&train_data, 900, 0.08);
        r
    }
    #[test]
    fn both_embedding_tables_match_finite_differences() {
        let train_data = [("red planet satellites", 0), ("french seat government", 1)];
        let original = DenseRetriever::new(&FACTS, &train_data, 3);
        let mut updated = original.clone();
        let learning_rate = 0.1;
        updated.train(&train_data, 1, learning_rate);
        for (query_table, word) in [(true, "red"), (false, "moons")] {
            let index = original.ids[word] * original.dim;
            let h = 0.01;
            let mut plus = original.clone();
            let mut minus = original.clone();
            let (before, after) = if query_table {
                plus.query_embeddings[index] += h;
                minus.query_embeddings[index] -= h;
                (
                    original.query_embeddings[index],
                    updated.query_embeddings[index],
                )
            } else {
                plus.document_embeddings[index] += h;
                minus.document_embeddings[index] -= h;
                (
                    original.document_embeddings[index],
                    updated.document_embeddings[index],
                )
            };
            let numerical =
                (plus.loss(&train_data[..1]) - minus.loss(&train_data[..1])) / (2.0 * h);
            let analytic = (before - after) / learning_rate;
            assert!(
                (numerical - analytic).abs() < 1e-5 + 1e-3 * numerical.abs(),
                "{word}: {numerical} vs {analytic}"
            );
        }
    }
    #[test]
    fn lexical_baseline_rewards_exact_terms() {
        let s = bm25("capital France", &FACTS);
        assert_eq!(argmax(&s), 1);
    }
    #[test]
    fn learned_exact_index_recovers_paraphrases() {
        let r = trained();
        for (q, want) in [
            ("red planet satellites", 0),
            ("french seat government", 1),
            ("safe systems language ownership", 2),
        ] {
            assert_eq!(argmax(&r.scores(q)), want);
        }
        let train_words: BTreeSet<_> = content_words("red planet satellites").into_iter().collect();
        let held_words: BTreeSet<_> = content_words("two satellites").into_iter().collect();
        assert_ne!(train_words, held_words);
        assert_eq!(argmax(&r.scores("two satellites")), 0);
    }
    #[test]
    fn pipeline_attributes_and_abstains() {
        let r = trained();
        let a = rag("french seat government", &r, &FACTS).unwrap();
        assert_eq!(a.source, "france");
        assert!(a.context.contains("[france]"));
        assert!(rag("unicorn weather prophecy", &r, &FACTS).is_none());
        let mut changed = FACTS.clone();
        changed[1].text = "Lyon is the supplied answer.";
        assert_eq!(
            rag("french seat government", &r, &changed).unwrap().text,
            "Lyon is the supplied answer."
        );
    }
}
