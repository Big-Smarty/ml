#[derive(Clone, Copy, Debug, PartialEq)]
struct Transition {
    a: f64,
    b: f64,
}

fn compose(later: Transition, earlier: Transition) -> Transition {
    // TODO: make one affine map equivalent to applying earlier and then later.
    let _ = (later, earlier);
    todo!()
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
