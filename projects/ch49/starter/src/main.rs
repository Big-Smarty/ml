//! Keep a sequential state oracle while implementing affine composition.

#[derive(Clone, Copy, Debug, PartialEq)]
struct Transition {
    a: f64,
    b: f64,
}

fn recurrence(transitions: &[Transition], mut state: f64) -> Vec<f64> {
    transitions
        .iter()
        .map(|t| {
            state = t.a * state + t.b;
            state
        })
        .collect()
}

fn compose(later: Transition, earlier: Transition) -> Transition {
    // TODO: compose later after earlier without changing time order.
    let _ = (later, earlier);
    todo!("compose the two affine state transitions")
}

fn main() {
    let _guided_todo: fn(Transition, Transition) -> Transition = compose;
    let transitions = [
        Transition { a: 0.5, b: 1.0 },
        Transition { a: 0.2, b: -0.5 },
    ];
    println!("sequential states: {:?}", recurrence(&transitions, 1.0));
    println!("Now run cargo test and implement compose.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composed_transition_matches_two_steps() {
        let earlier = Transition { a: 0.5, b: 1.0 };
        let later = Transition { a: 0.2, b: -0.5 };
        assert_eq!(compose(later, earlier), Transition { a: 0.1, b: -0.3 });
    }
}
