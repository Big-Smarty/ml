//! Scalar reverse-mode automatic differentiation with shared-node handling.
use std::{
    cell::RefCell,
    collections::HashSet,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(Clone)]
struct Value(Rc<RefCell<Node>>);
#[derive(Clone)]
enum Op {
    Leaf,
    Add(Value, Value),
    Mul(Value, Value),
    Tanh(Value),
}
struct Node {
    data: f64,
    grad: f64,
    op: Op,
}

impl Value {
    fn new(data: f64) -> Self {
        Self(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op: Op::Leaf,
        })))
    }
    fn data(&self) -> f64 {
        self.0.borrow().data
    }
    fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    fn tanh(self) -> Self {
        let data = self.data().tanh();
        Self(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op: Op::Tanh(self),
        })))
    }
    fn add_grad(&self, g: f64) {
        self.0.borrow_mut().grad += g;
    }
    fn id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }
    // ponytail: recursion suits tiny scalar graphs; use an explicit stack for deep graphs.
    fn topo(&self, seen: &mut HashSet<usize>, out: &mut Vec<Value>) {
        if !seen.insert(self.id()) {
            return;
        }
        match self.0.borrow().op.clone() {
            Op::Leaf => {}
            Op::Add(a, b) | Op::Mul(a, b) => {
                a.topo(seen, out);
                b.topo(seen, out)
            }
            Op::Tanh(a) => a.topo(seen, out),
        }
        out.push(self.clone());
    }
    fn backward(&self) -> Result<(), &'static str> {
        if !self.data().is_finite() {
            return Err("cannot differentiate a nonfinite result");
        }
        let mut order = Vec::new();
        self.topo(&mut HashSet::new(), &mut order);
        if order.iter().any(|v| !v.data().is_finite()) {
            return Err("cannot differentiate nonfinite intermediate values");
        }
        for v in &order {
            v.0.borrow_mut().grad = 0.0;
        }
        self.0.borrow_mut().grad = 1.0;
        for v in order.into_iter().rev() {
            let (g, op) = {
                let n = v.0.borrow();
                (n.grad, n.op.clone())
            };
            if !g.is_finite() {
                return Err("gradient overflowed");
            }
            match op {
                Op::Leaf => {}
                Op::Add(a, b) => {
                    a.add_grad(g);
                    b.add_grad(g);
                }
                Op::Mul(a, b) => {
                    let ad = a.data();
                    let bd = b.data();
                    a.add_grad(g * bd);
                    b.add_grad(g * ad);
                }
                Op::Tanh(a) => a.add_grad(g * (1.0 - v.data() * v.data())),
            }
        }
        Ok(())
    }
}

impl Add for Value {
    type Output = Value;
    fn add(self, rhs: Value) -> Value {
        let data = self.data() + rhs.data();
        Value(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op: Op::Add(self, rhs),
        })))
    }
}
impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        let data = self.data() * rhs.data();
        Value(Rc::new(RefCell::new(Node {
            data,
            grad: 0.0,
            op: Op::Mul(self, rhs),
        })))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = Value::new(3.0);
    let y = x.clone() * x.clone() + x.clone(); // x is shared by three incoming paths.
    y.backward()?;
    println!("y = x*x+x at x=3: {:.1}", y.data());
    println!("dy/dx: {:.1} (expected 7)", x.grad());
    let z = (Value::new(0.5) * Value::new(2.0)).tanh();
    println!("tanh(1): {:.6}", z.data());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shared_node_accumulates_every_path() {
        let x = Value::new(3.0);
        let y = x.clone() * x.clone() + x.clone();
        y.backward().unwrap();
        assert_eq!(y.data(), 12.0);
        assert_eq!(x.grad(), 7.0);
    }
    #[test]
    fn chain_rule_matches_finite_difference() {
        let x = Value::new(0.4);
        let y = (x.clone() * x.clone() + x.clone()).tanh();
        y.backward().unwrap();
        let h = 1e-5;
        let f = |q: f64| (q * q + q).tanh();
        let numeric = (f(0.4 + h) - f(0.4 - h)) / (2.0 * h);
        assert!((x.grad() - numeric).abs() < 1e-6 + 1e-4 * numeric.abs());
    }
    #[test]
    fn backward_clears_old_gradients() {
        let x = Value::new(2.0);
        let y = x.clone() * x.clone();
        y.backward().unwrap();
        y.backward().unwrap();
        assert_eq!(x.grad(), 4.0);
    }
    #[test]
    fn finite_output_does_not_hide_nonfinite_graph_or_gradient() {
        let y = (Value::new(f64::MAX) * Value::new(2.0)).tanh();
        assert!(y.data().is_finite());
        assert!(y.backward().is_err());
        let x = Value::new(0.0);
        let y = (x * Value::new(f64::MAX)) * Value::new(2.0);
        assert_eq!(y.data(), 0.0);
        assert!(y.backward().is_err());
    }
}
