fn cosine(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1];
    let norms = (a[0].powi(2) + a[1].powi(2)).sqrt() * (b[0].powi(2) + b[1].powi(2)).sqrt();
    dot / norms
}

#[cfg(test)]
fn best_caption(image: [f64; 2], captions: &[[f64; 2]]) -> usize {
    // TODO: return the index with greatest cosine similarity.
    let _ = (image, captions);
    todo!("compare every caption embedding")
}

fn main() {
    let image = [0.9, 0.1];
    for (name, text) in [("vertical", [1.0, 0.0]), ("horizontal", [0.0, 1.0])] {
        println!("{name:10} similarity={:.3}", cosine(image, text));
    }
    println!("Normalized similarity is the retrieval checkpoint before learned encoders.");
}

#[test]
fn retrieves_nearest_caption() {
    assert_eq!(best_caption([0.9, 0.1], &[[0.0, 1.0], [1.0, 0.0]]), 1);
}
