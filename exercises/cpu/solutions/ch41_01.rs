fn pack(low: i8, high: i8) -> u8 {
    (low as u8 & 0x0f) | ((high as u8 & 0x0f) << 4)
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
