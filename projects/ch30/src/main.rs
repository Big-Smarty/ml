use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let gpu = ch30::Gpu::new()?;
    let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let got = gpu.matmul(&a, &b, 2, 3, 2)?;
    println!("GPU [2,3] × [3,2] = {got:?}");
    println!(
        "CPU oracle             = {:?}",
        ch30::cpu_matmul(&a, &b, 2, 3, 2)?
    );
    println!(
        "GPU sum([1,2,3,4,5])   = {}",
        gpu.reduce_sum(&[1.0, 2.0, 3.0, 4.0, 5.0])?
    );
    Ok(())
}
