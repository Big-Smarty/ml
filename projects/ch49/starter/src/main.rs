//! Keep a sequential state oracle while implementing affine composition.

#[derive(Clone, Copy, Debug, PartialEq)]
struct Transition {
    a: f64,
    b: f64,
}

impl Transition {
    /// Apply `earlier`, then `self`.
    fn after(self, earlier: Self) -> Self {
        // TODO: compose self after earlier without changing time order.
        let _ = earlier;
        todo!("compose the two affine state transitions")
    }

    fn apply(self, state: f64) -> f64 {
        self.a * state + self.b
    }
}

fn recurrent(transitions: &[Transition], initial: f64) -> Vec<f64> {
    let mut state = initial;
    transitions
        .iter()
        .map(|&transition| {
            state = transition.apply(state);
            state
        })
        .collect()
}

fn main() {
    let _guided_todo: fn(Transition, Transition) -> Transition = Transition::after;
    let transitions = [
        Transition { a: 0.5, b: 1.0 },
        Transition { a: 0.2, b: -0.5 },
    ];
    println!("sequential states: {:?}", recurrent(&transitions, 1.0));
    println!("Now run cargo test and implement Transition::after.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composed_transition_matches_two_steps() {
        let earlier = Transition { a: 0.5, b: 1.0 };
        let later = Transition { a: 0.2, b: -0.5 };
        assert_eq!(later.after(earlier), Transition { a: 0.1, b: -0.3 });
    }
}
