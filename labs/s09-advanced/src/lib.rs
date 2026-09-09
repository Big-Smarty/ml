pub mod ch47;
pub mod ch48;
pub mod ch49;
pub mod ch50;
pub mod ch51;
pub mod ch52;
pub mod ch53;
pub mod ch54;
pub mod ch55;
pub mod ch56;
pub mod solutions {
    pub mod ch47;
    pub mod ch48;
    pub mod ch49;
    pub mod ch50;
    pub mod ch51;
    pub mod ch52;
    pub mod ch53;
    pub mod ch54;
    pub mod ch55;
    pub mod ch56;
}

pub fn ensure(ok: bool, message: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(format!("GOAL_NOT_MET: {message}"))
    }
}
pub fn close(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-10 + 1e-8 * a.abs().max(b.abs())
}
