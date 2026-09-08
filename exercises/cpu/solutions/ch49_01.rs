#[derive(Clone, Copy, Debug, PartialEq)]
struct Transition {
    a: f64,
    b: f64,
}

impl Transition {
    fn after(self, earlier: Self) -> Self {
        Self {
            a: self.a * earlier.a,
            b: self.a * earlier.b + self.b,
        }
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
    let _after: fn(Transition, Transition) -> Transition = Transition::after;
    let _apply: fn(Transition, f64) -> f64 = Transition::apply;
    let _recurrent: fn(&[Transition], f64) -> Vec<f64> = recurrent;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composition_matches_two_steps() {
        let earlier = Transition { a: 0.5, b: 1.0 };
        let later = Transition { a: 0.2, b: -0.5 };
        let composed = later.after(earlier);
        assert_eq!(composed, Transition { a: 0.1, b: -0.3 });
        let direct = recurrent(&[earlier, later], 1.0)[1];
        assert!((composed.apply(1.0) - direct).abs() < 1e-12);
    }
}
