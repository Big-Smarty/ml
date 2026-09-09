//! Supplied append-only graph storage. Earlier node IDs are a topological order.
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Leaf,
    Add(usize, usize),
    Mul(usize, usize),
    Tanh(usize),
}
#[derive(Clone, Debug)]
pub struct Node {
    pub value: f64,
    pub grad: f64,
    pub op: Op,
}
#[derive(Clone, Debug, Default)]
pub struct Tape {
    pub nodes: Vec<Node>,
}
impl Tape {
    pub fn leaf(&mut self, value: f64) -> Result<usize, String> {
        self.push(value, Op::Leaf)
    }
    fn push(&mut self, value: f64, op: Op) -> Result<usize, String> {
        if !value.is_finite() {
            return Err("graph value is not finite".into());
        }
        let id = self.nodes.len();
        self.nodes.push(Node {
            value,
            grad: 0.,
            op,
        });
        Ok(id)
    }
    pub fn value(&self, id: usize) -> Result<f64, String> {
        self.nodes
            .get(id)
            .map(|n| n.value)
            .ok_or_else(|| format!("node {id} does not exist"))
    }
    pub fn add(&mut self, a: usize, b: usize) -> Result<usize, String> {
        self.push(self.value(a)? + self.value(b)?, Op::Add(a, b))
    }
    pub fn mul(&mut self, a: usize, b: usize) -> Result<usize, String> {
        self.push(self.value(a)? * self.value(b)?, Op::Mul(a, b))
    }
    pub fn tanh(&mut self, a: usize) -> Result<usize, String> {
        self.push(self.value(a)?.tanh(), Op::Tanh(a))
    }
    pub fn prepare(&mut self, root: usize) -> Result<Vec<usize>, String> {
        self.value(root)?;
        let mut reachable = vec![false; self.nodes.len()];
        reachable[root] = true;
        let mut order = Vec::new();
        for id in (0..=root).rev() {
            if !reachable[id] {
                continue;
            }
            if !self.nodes[id].value.is_finite() {
                return Err("nonfinite graph value".into());
            }
            let parents = match self.nodes[id].op {
                Op::Leaf => vec![],
                Op::Add(a, b) | Op::Mul(a, b) => vec![a, b],
                Op::Tanh(a) => vec![a],
            };
            for parent in parents {
                if parent >= id {
                    return Err("graph parents must be earlier nodes".into());
                }
                reachable[parent] = true;
            }
            order.push(id);
        }
        for node in &mut self.nodes {
            node.grad = 0.;
        }
        self.nodes[root].grad = 1.;
        Ok(order)
    }
    pub fn accumulate(&mut self, id: usize, contribution: f64) -> Result<(), String> {
        let node = self
            .nodes
            .get_mut(id)
            .ok_or("unknown gradient destination")?;
        let grad = node.grad + contribution;
        if !grad.is_finite() {
            return Err("gradient overflow".into());
        }
        node.grad = grad;
        Ok(())
    }
}
