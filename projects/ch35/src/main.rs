#[derive(Debug)]
struct Attention {
    output: Vec<f64>,
    weights: Vec<f64>,
}
#[derive(Debug)]
struct Backward {
    dq: Vec<f64>,
    dk: Vec<f64>,
    dv: Vec<f64>,
}
fn causal_attention(q: &[f64], k: &[f64], v: &[f64], t: usize, d: usize) -> Attention {
    assert!(t > 0 && d > 0);
    let len = t.checked_mul(d).expect("attention shape overflow");
    assert_eq!(q.len(), len);
    assert_eq!(k.len(), len);
    assert_eq!(v.len(), len);
    let mut a = vec![0.0; t * t];
    let mut out = vec![0.0; t * d];
    let scale = (d as f64).sqrt().recip();
    for i in 0..t {
        let mut max = f64::NEG_INFINITY;
        for j in 0..=i {
            let s = (0..d).map(|z| q[i * d + z] * k[j * d + z]).sum::<f64>() * scale;
            a[i * t + j] = s;
            max = max.max(s)
        }
        let mut sum = 0.0;
        for j in 0..=i {
            a[i * t + j] = (a[i * t + j] - max).exp();
            sum += a[i * t + j]
        }
        for j in 0..=i {
            a[i * t + j] /= sum;
            for z in 0..d {
                out[i * d + z] += a[i * t + j] * v[j * d + z]
            }
        }
    }
    Attention {
        output: out,
        weights: a,
    }
}
fn backward(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    a: &[f64],
    dout: &[f64],
    t: usize,
    d: usize,
) -> Backward {
    assert!(t > 0 && d > 0);
    let len = t.checked_mul(d).expect("attention shape overflow");
    assert_eq!(q.len(), len);
    assert_eq!(k.len(), len);
    assert_eq!(v.len(), len);
    assert_eq!(dout.len(), len);
    assert_eq!(
        a.len(),
        t.checked_mul(t).expect("probability shape overflow")
    );
    let mut dq = vec![0.0; t * d];
    let mut dk = vec![0.0; t * d];
    let mut dv = vec![0.0; t * d];
    let scale = (d as f64).sqrt().recip();
    for i in 0..t {
        let mut da = vec![0.0; i + 1];
        for j in 0..=i {
            for z in 0..d {
                da[j] += dout[i * d + z] * v[j * d + z];
                dv[j * d + z] += a[i * t + j] * dout[i * d + z]
            }
        }
        let dot = (0..=i).map(|j| da[j] * a[i * t + j]).sum::<f64>();
        for j in 0..=i {
            let ds = a[i * t + j] * (da[j] - dot) * scale;
            for z in 0..d {
                dq[i * d + z] += ds * k[j * d + z];
                dk[j * d + z] += ds * q[i * d + z]
            }
        }
    }
    Backward { dq, dk, dv }
}
#[cfg(test)]
fn loss(q: &[f64], k: &[f64], v: &[f64], t: usize, d: usize, upstream: &[f64]) -> f64 {
    let output = causal_attention(q, k, v, t, d).output;
    assert_eq!(output.len(), upstream.len());
    output.iter().zip(upstream).map(|(a, b)| a * b).sum()
}
fn main() {
    let q = [1.0, 0.0, 0.0, 1.0];
    let k = [1.0, 0.0, 1.0, 1.0];
    let v = [2.0, 0.0, 0.0, 4.0];
    let a = causal_attention(&q, &k, &v, 2, 2);
    println!(
        "causal weights [q0->k0,q0->k1,q1->k0,q1->k1] = {:.3?}",
        a.weights
    );
    println!("outputs = {:.3?}", a.output);
    let g = backward(&q, &k, &v, &a.weights, &[1.0; 4], 2, 2);
    println!(
        "gradient norms: Q={:.4} K={:.4} V={:.4}",
        norm(&g.dq),
        norm(&g.dk),
        norm(&g.dv)
    );
}
fn norm(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn future_is_exactly_masked() {
        let q = [1.0, 0.5, 0.2, -0.4, 0.3, 0.7];
        let k = [0.2, 0.1, 0.4, 0.3, 9.0, 9.0];
        let v = [1.0, 2.0, 3.0, 4.0, 99.0, 99.0];
        let a = causal_attention(&q, &k, &v, 3, 2);
        assert_eq!(&a.output[..2], &v[..2]);
        assert_eq!(a.weights[1], 0.0);
        assert_eq!(a.weights[2], 0.0);
        for row in a.weights.chunks_exact(3) {
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let mut changed_k = k;
        changed_k[4] = -500.0;
        changed_k[5] = 700.0;
        let mut changed_v = v;
        changed_v[4] = -900.0;
        changed_v[5] = 600.0;
        let changed = causal_attention(&q, &changed_k, &changed_v, 3, 2);
        assert_eq!(&a.output[..4], &changed.output[..4]);
    }
    fn numeric(q: &[f64], k: &[f64], v: &[f64], group: usize, index: usize, up: &[f64]) -> f64 {
        let eps = 1e-5;
        let (mut qp, mut kp, mut vp) = (q.to_vec(), k.to_vec(), v.to_vec());
        [&mut qp, &mut kp, &mut vp][group][index] += eps;
        let plus = loss(&qp, &kp, &vp, 3, 2, up);
        [&mut qp, &mut kp, &mut vp][group][index] -= 2.0 * eps;
        let minus = loss(&qp, &kp, &vp, 3, 2, up);
        (plus - minus) / (2.0 * eps)
    }
    #[test]
    fn qkv_gradients_match_central_differences() {
        let q = vec![0.7, -0.2, 0.1, 0.9, -0.5, 0.3];
        let k = vec![0.4, 0.6, -0.8, 0.2, 0.3, -0.7];
        let v = vec![1.0, -0.5, 0.2, 0.8, -0.4, 0.9];
        let up = [0.3, -0.7, 1.2, 0.4, -0.2, 0.6];
        let a = causal_attention(&q, &k, &v, 3, 2);
        let g = backward(&q, &k, &v, &a.weights, &up, 3, 2);
        for (group, analytic) in [g.dq, g.dk, g.dv].iter().enumerate() {
            for (i, &a) in analytic.iter().enumerate() {
                assert!((a - numeric(&q, &k, &v, group, i, &up)).abs() < 1e-6);
            }
        }
    }
}
