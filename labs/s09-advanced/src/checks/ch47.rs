pub fn check() -> Result<(), String> {
    let pruned = prune_to_density(&[-1.0, 0.2, 3.0, -2.0], 0.5)?;
    crate::ensure(
        pruned == [0.0, 0.0, 3.0, -2.0],
        &format!("magnitude budget: got {pruned:?}; expected [0,0,3,-2]"),
    )?;
    crate::ensure(
        prune_to_density(&[2.0, -2.0, 0.5], 0.34)? == [2.0, 0.0, 0.0],
        "ties must retain the first original index",
    )?;
    let w = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, -3.0, 0.0, 4.0];
    let csr = CsrMatrix::from_dense(&w, 3, 3)?;
    crate::ensure(
        csr.row_ptr == [0, 1, 1, 3] && csr.col_idx == [1, 0, 2] && csr.values == [2.0, -3.0, 4.0],
        "CSR must omit zeros and preserve the empty middle row",
    )?;
    for x in [[1.0, 2.0, 3.0], [-2.0, 0.5, 1.5]] {
        let expected = dense_matvec_reference(&w, 3, 3, &x)?;
        let mut into = [0.0; 3];
        csr.matvec_into(&x, &mut into)?;
        crate::ensure(
            close(&expected, &csr.matvec(&x)?) && close(&expected, &into),
            "sparse kernel differs on unfamiliar signed input",
        )?;
    }
    crate::ensure(
        CsrMatrix::from_dense(&[1.0], 1, 2).is_err(),
        "shape boundary accepted malformed matrix",
    )
}

pub fn run(args: &[String]) -> Result<(), String> {
    if !args.is_empty() {
        return Err("this experiment takes no extra arguments".into());
    }
    demo().map_err(|e| e.to_string())
}
