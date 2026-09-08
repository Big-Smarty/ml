#[derive(Clone, Copy, Debug, PartialEq)]
struct Transition {
    a: f64,
    b: f64,
}

fn compose(later: Transition, earlier: Transition) -> Transition {
    Transition {
        a: later.a * earlier.a,
        b: later.a * earlier.b + later.b,
    }
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composition_matches_two_steps() {
        assert_eq!(
            compose(
                Transition { a: 0.2, b: -0.5 },
                Transition { a: 0.5, b: 1.0 }
            ),
            Transition { a: 0.1, b: -0.3 }
        );
    }
}
