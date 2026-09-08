#[derive(Clone, Copy)]
struct Gradient {
    dw: f64,
    db: f64,
    count: usize,
}

fn local_gradient(batch: &[(f64, f64)], weight: f64, bias: f64) -> Gradient {
    let (dw, db) = batch.iter().fold((0.0, 0.0), |(dw, db), &(x, y)| {
        let error = weight * x + bias - y;
        (dw + 2.0 * error * x, db + 2.0 * error)
    });
    Gradient {
        dw,
        db,
        count: batch.len(),
    }
}

#[cfg(test)]
fn average_worker_gradients(parts: &[Gradient]) -> (f64, f64) {
    // TODO: sum worker gradients, then divide by the total example count.
    let _ = parts;
    todo!("aggregate gradients by examples, not by workers")
}

fn main() {
    let data = [(-1.0, -1.0), (0.0, 1.0), (1.0, 3.0), (2.0, 5.0)];
    let left = local_gradient(&data[..2], 0.0, 0.0);
    let right = local_gradient(&data[2..], 0.0, 0.0);
    println!(
        "worker 0: sum_dw={:.1}, sum_db={:.1}, examples={}",
        left.dw, left.db, left.count
    );
    println!(
        "worker 1: sum_dw={:.1}, sum_db={:.1}, examples={}",
        right.dw, right.db, right.count
    );
    println!("Two real worker messages are next; complete the aggregation test.");
}

#[test]
fn worker_aggregation_matches_full_batch() {
    let parts = [
        Gradient {
            dw: -2.0,
            db: 0.0,
            count: 1,
        },
        Gradient {
            dw: -6.0,
            db: -2.0,
            count: 3,
        },
    ];
    assert_eq!(average_worker_gradients(&parts), (-2.0, -0.5));
}
