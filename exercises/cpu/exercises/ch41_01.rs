fn pack(_low: i8, _high: i8) -> u8 {
    // TODO: store two signed four-bit values as low and high nibbles.
    todo!("pack int4")
}
fn main() {
    println!("packed: {:02x}", pack(-1, 3));
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packs_twos_complement_nibbles() {
        assert_eq!(pack(-1, 3), 0x3f);
    }
}
