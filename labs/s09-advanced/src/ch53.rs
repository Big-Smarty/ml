//! Chapter 53 learner algorithms. Baseline rejects every candidate and monitors the cumulative mean. Implement a canary release decision and a four-request rolling window; keep all request/artifact validation intact.
include!("common/ch53.rs");
include!("checks/ch53.rs");
fn choose_release<'a>(
    candidate: &'a Model,
    previous: &'a Model,
    canary: &[f64],
    tolerance: f64,
) -> Result<&'a Model, &'static str> {
    if canary.is_empty() || !tolerance.is_finite() || tolerance < 0.0 {
        return Err("canary inputs and tolerance are required");
    }
    // Conservative baseline validates the canary but never promotes a candidate.
    for &input in canary {
        candidate.predict(input)?;
        previous.predict(input)?;
    }
    Ok(previous)
}

impl Monitor {
    fn observe(&mut self, value: f64) -> Result<(), &'static str> {
        if !value.is_finite() {
            return Err("monitor values must be finite");
        }
        let count = self.count.checked_add(1).ok_or("monitor count overflow")?;
        let sum = self.sum + value;
        if !sum.is_finite() {
            return Err("monitor sum overflow");
        }
        self.count = count;
        self.sum = sum;
        self.window.push_back(value);
        // Baseline retains the complete history; implement a bounded four-request window.
        Ok(())
    }
}

impl Monitor {
    fn drift_from(&self, baseline: f64, threshold: f64) -> Result<bool, &'static str> {
        if !baseline.is_finite() || !threshold.is_finite() || threshold < 0.0 {
            return Err("drift settings must be finite and threshold nonnegative");
        }
        Ok(self
            .mean()
            .is_some_and(|mean| (mean - baseline).abs() > threshold))
    }
}
