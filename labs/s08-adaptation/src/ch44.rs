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
    // Working lexical overlap baseline. LEARNER: replace with BM25's full
    // corpus-frequency, saturation and length-normalization calculation.
    let query = content_words(query);
    docs.iter()
        .map(|doc| {
            let terms = words(doc.text);
            query.iter().filter(|w| terms.contains(w)).count() as f32
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

fn experiment() -> Result<(), &'static str> {
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
    let mut hits = 0;
    for (q, want) in held_out {
        let top = exact_top(
            &dense.encode_with(q, &dense.query_embeddings),
            &dense.doc_vectors(),
        )?;
        hits += usize::from(top == want);
    }
    println!("dense held-out retrieval hit@1: {hits}/{}", held_out.len());
    let queries = [
        ("Which satellites orbit the red planet?", Some("planet")),
        ("What is the French capital?", Some("france")),
        ("Who won the lunar chess final?", None),
        ("France lunar chess champion", None),
    ];
    let mut answered = 0;
    let mut correct = 0;
    let mut attributed = 0;
    for (query, expected) in queries {
        let lexical = bm25(query, &FACTS);
        println!(
            "query: {query}\n  lexical overlap top={} scores={lexical:?}",
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
    println!("grounded evaluation: dense hit@1={hits}/3, correct supported source={correct}/2, answered={answered}/4, answers with matching source label={attributed}/{answered}, unsupported abstained={}/2",queries.iter().filter(|(q,expected)|expected.is_none() && rag(q,&dense,&FACTS).is_none()).count());
    Ok(())
}

/// Run the useful baseline; --check separately verifies the learning goal.
pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("unexpected experiment argument".into());
    }
    experiment().map_err(str::to_string)?;
    evaluation_experiment().map_err(str::to_string)
}

/// Exhaustive search has a deterministic first-index tie break and validates vector shape.
fn exact_top(query: &[f32], docs: &[Vec<f32>]) -> Result<usize, &'static str> {
    // Working nearest-vector baseline by L2 distance. Replace with exhaustive
    // inner-product ranking; these metrics differ for unnormalized embeddings.
    if query.is_empty()
        || docs.is_empty()
        || query.iter().any(|x| !x.is_finite())
        || docs
            .iter()
            .any(|d| d.len() != query.len() || d.iter().any(|x| !x.is_finite()))
    {
        return Err("invalid vectors");
    }
    let scores: Vec<_> = docs
        .iter()
        .map(|d| {
            -query
                .iter()
                .zip(d)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
        })
        .collect();
    Ok(argmax(&scores))
}
/// Coverage denominator is all questions; risk denominator is answered questions.
fn coverage_risk(rows: &[(f32, bool)], threshold: f32) -> Result<(f32, Option<f32>), &'static str> {
    // Baseline reports coverage and unconditional error rate. Replace the
    // second denominator with answered count; define zero-answer behavior.
    if rows.is_empty() || !threshold.is_finite() || rows.iter().any(|(s, _)| !s.is_finite()) {
        return Err("invalid evaluation rows");
    }
    let answered = rows.iter().filter(|(s, _)| *s >= threshold).count();
    let wrong = rows.iter().filter(|(s, c)| *s >= threshold && !c).count();
    Ok((
        answered as f32 / rows.len() as f32,
        Some(wrong as f32 / rows.len() as f32),
    ))
}
fn evaluation_experiment() -> Result<(), &'static str> {
    // These labels deliberately include an unsupported query with familiar words.
    let rows = [(0.9, true), (0.8, true), (0.7, false), (0.1, false)];
    for threshold in [0., 0.5, 0.85, 1.] {
        let (c, r) = coverage_risk(&rows, threshold)?;
        println!("threshold={threshold}: coverage={c:.2}, selective risk={r:?}");
    }
    println!("Overlapping unsupported example: 'France lunar chess champion'. A source ID proves provenance, not relevance.");
    Ok(())
}
pub fn check() -> Result<(), String> {
    let verify = || -> Result<(), &'static str> {
        let docs = [
            Fact {
                id: "a",
                text: "rare",
            },
            Fact {
                id: "b",
                text: "rare common common common",
            },
            Fact {
                id: "c",
                text: "common",
            },
        ];
        let scores = bm25("rare", &docs);
        if !(scores[0] > scores[1] && scores[1] > scores[2]) {
            return Err("GOAL_NOT_MET: implement BM25 saturation and document-length normalization; overlap alone ties these documents");
        }
        if exact_top(&[1., 0.], &[vec![-3., 0.], vec![-1., 0.], vec![-2., 0.]])? != 1 {
            return Err("GOAL_NOT_MET: exact search must consider every candidate, including negative scores");
        }
        if exact_top(&[1., 0.], &[vec![1., 0.], vec![3., 0.]])? != 1 {
            return Err("GOAL_NOT_MET: inner-product search must not silently substitute L2 on unequal-norm vectors");
        }
        if coverage_risk(&[(0.9, true), (0.8, false), (0.1, false)], 0.5)? != (2. / 3., Some(0.5)) {
            return Err("GOAL_NOT_MET: coverage and risk use different denominators");
        }
        if coverage_risk(&[(0.1, false)], 1.)? != (0., None) {
            return Err("GOAL_NOT_MET: zero coverage has undefined risk, not zero error");
        }
        let train = [
            ("red planet satellites", 0),
            ("french seat government", 1),
            ("safe systems language ownership", 2),
        ];
        let mut r = DenseRetriever::new(&FACTS, &train, 8);
        let before = r.loss(&train);
        r.train(&train, 900, 0.08);
        if r.loss(&train) >= before || argmax(&r.scores("two satellites")) != 0 {
            return Err(
                "GOAL_NOT_MET: trained dense retriever must lower loss and retrieve an unseen combination",
            );
        }
        let answer =
            rag("french seat government", &r, &FACTS).ok_or("supported answer abstained")?;
        if answer.source != "france"
            || answer.text != FACTS[1].text
            || !answer.context.contains("[france]")
        {
            return Err("GOAL_NOT_MET: assemble the actual retrieved text with a stable citation");
        }
        if rag("unicorn weather prophecy", &r, &FACTS).is_some() {
            return Err("GOAL_NOT_MET: unknown-content query must abstain");
        }
        evaluation_experiment()?;
        println!("goal: BM25, exact search, dense transfer, context and selective evaluation pass");
        Ok(())
    };
    verify().map_err(str::to_string)
}

#[cfg(test)]
mod baseline_tests {
    #[test]
    fn supplied_baseline_runs() {
        assert!(super::run(&[]).is_ok());
    }
}
