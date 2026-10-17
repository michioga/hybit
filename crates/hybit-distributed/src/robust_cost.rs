//! H1-C3I: opt-in paired-epoch cost admission before root Graph reselection.
//!
//! This evaluator is deterministic and MPI-independent. The caller supplies
//! identical, representative max-rank epoch medians and budget estimates on
//! every participating rank, or computes the decision only on rank 0.
//! The *minimum observed paired gain* is NOT a statistical lower confidence
//! bound and is not proof of future speedup. There is NO automatic routing.

/// The reason an experimental Graph candidate is admitted or rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RobustCostDecision {
    InsufficientEpochs,
    InvalidTimings,
    NoConsistentPairedGain,
    UnamortizedTotalCost,
    Admit,
}

/// All costs are integer nanoseconds. `epochs` contains paired
/// `(baseline_max_rank_spmv_ns, graph_max_rank_spmv_ns)` medians; each pair
/// must come from the SAME calibration epoch (never from held-out results).
#[derive(Clone, Copy, Debug)]
pub struct RobustCostInputs<'a> {
    pub epochs: &'a [(u64, u64)],
    pub expected_spmv_calls: u64,
    /// Wall time from the beginning of the probe, including its initial
    /// baseline, Graph discovery, both native preparations, and MPI samples.
    pub probe_wall_ns: u64,
    /// Baseline-only setup, which is not charged to the baseline twice.
    pub baseline_native_setup_ns: u64,
    /// Setup when preparing the selected Graph AFTER the final decision.
    pub graph_native_setup_ns: u64,
    /// Preregistered conservative budget for a future second root selector;
    /// actual final selection time must be checked independently afterward.
    pub second_selector_budget_ns: u64,
    /// Additional positive net benefit required beyond all preparation costs.
    pub minimum_net_savings_ns: u64,
}

/// Integer economic audit. `required_calls` is an empirical threshold under
/// the observed minimum paired saving; it is NOT an expected solver iteration
/// count, a confidence interval, or an actual full-N execution time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RobustCostReport {
    pub decision: RobustCostDecision,
    pub minimum_paired_gain_ns: Option<u64>,
    pub conservative_extra_budget_ns: u128,
    pub estimated_gross_savings_ns: u128,
    pub required_calls: Option<u64>,
}

/// Reject on ANY nonpositive paired-epoch gain. Account for the whole probe,
/// forecast selector budget and final Graph setup, subtract baseline-only setup
/// exactly once. A conservative 25% budget contingency is charged to the
/// positive incremental preparation cost. `minimum_net_savings_ns` is extra.
///
/// Comparisons use u128 so even extreme u64 budgets/call counts cannot
/// overflow. This is a *preflight veto*; structural/Halo/scoped-MPI gates
/// still apply afterward, and production routing is unchanged.
pub fn evaluate_robust_cost(inputs: RobustCostInputs<'_>) -> RobustCostReport {
    let mut result = RobustCostReport {
        decision: RobustCostDecision::InsufficientEpochs,
        minimum_paired_gain_ns: None,
        conservative_extra_budget_ns: 0,
        estimated_gross_savings_ns: 0,
        required_calls: None,
    };
    if inputs.epochs.len() < 3 {
        return result;
    }
    if inputs
        .epochs
        .iter()
        .any(|&(baseline, graph)| baseline == 0 || graph == 0)
    {
        result.decision = RobustCostDecision::InvalidTimings;
        return result;
    }
    let minimum_gain = inputs
        .epochs
        .iter()
        .map(|&(baseline, graph)| baseline.checked_sub(graph).filter(|&delta| delta > 0))
        .collect::<Option<Vec<_>>>()
        .and_then(|values| values.into_iter().min());
    let Some(gain) = minimum_gain else {
        result.decision = RobustCostDecision::NoConsistentPairedGain;
        return result;
    };
    result.minimum_paired_gain_ns = Some(gain);
    let raw_extra = (u128::from(inputs.probe_wall_ns)
        + u128::from(inputs.second_selector_budget_ns)
        + u128::from(inputs.graph_native_setup_ns))
    .saturating_sub(u128::from(inputs.baseline_native_setup_ns));
    let extra_with_contingency = raw_extra + raw_extra.div_ceil(4);
    result.conservative_extra_budget_ns = extra_with_contingency;
    result.estimated_gross_savings_ns = u128::from(gain) * u128::from(inputs.expected_spmv_calls);
    let threshold = extra_with_contingency + u128::from(inputs.minimum_net_savings_ns);
    let required_calls = threshold / u128::from(gain) + 1; // strict net benefit
    result.required_calls = u64::try_from(required_calls).ok();
    result.decision = if result.estimated_gross_savings_ns > threshold {
        RobustCostDecision::Admit
    } else {
        RobustCostDecision::UnamortizedTotalCost
    };
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inputs<'a>(epochs: &'a [(u64, u64)], calls: u64) -> RobustCostInputs<'a> {
        RobustCostInputs {
            epochs,
            expected_spmv_calls: calls,
            probe_wall_ns: 1_000_000,
            baseline_native_setup_ns: 100_000,
            graph_native_setup_ns: 200_000,
            second_selector_budget_ns: 300_000,
            minimum_net_savings_ns: 0,
        }
    }

    #[test]
    fn consistent_large_gain_admits() {
        let report = evaluate_robust_cost(inputs(
            &[(10_000, 9_000), (11_000, 9_500), (12_000, 10_000)],
            10_000,
        ));
        assert_eq!(report.decision, RobustCostDecision::Admit);
        assert_eq!(report.minimum_paired_gain_ns, Some(1_000));
        assert_eq!(report.conservative_extra_budget_ns, 1_750_000);
        assert_eq!(report.required_calls, Some(1_751));
    }

    #[test]
    fn one_noisy_reversal_denies_despite_good_median() {
        let report = evaluate_robust_cost(inputs(
            &[(10_000, 8_000), (10_000, 8_000), (10_000, 10_100)],
            100_000,
        ));
        assert_eq!(report.decision, RobustCostDecision::NoConsistentPairedGain);
        assert_eq!(report.required_calls, None);
    }

    #[test]
    fn equality_denies() {
        let report = evaluate_robust_cost(inputs(&[(10, 9), (10, 9), (10, 10)], 100_000));
        assert_eq!(report.decision, RobustCostDecision::NoConsistentPairedGain);
    }

    #[test]
    fn all_positive_but_unamortized_denies() {
        let report = evaluate_robust_cost(inputs(&[(10_000, 9_900); 3], 10_000));
        assert_eq!(report.decision, RobustCostDecision::UnamortizedTotalCost);
        assert_eq!(report.required_calls, Some(17_501));
    }

    #[test]
    fn short_or_invalid_evidence_denies() {
        assert_eq!(
            evaluate_robust_cost(inputs(&[(5, 4); 2], 100)).decision,
            RobustCostDecision::InsufficientEpochs
        );
        assert_eq!(
            evaluate_robust_cost(inputs(&[(5, 4), (5, 0), (5, 4)], 100)).decision,
            RobustCostDecision::InvalidTimings
        );
    }

    #[test]
    fn threshold_is_strict_and_minimum_savings_is_charged() {
        let epochs = [(2_000, 1_000); 3];
        let mut value = inputs(&epochs, 1_750);
        assert_eq!(
            evaluate_robust_cost(value).decision,
            RobustCostDecision::UnamortizedTotalCost
        );
        value.expected_spmv_calls = 1_751;
        assert_eq!(
            evaluate_robust_cost(value).decision,
            RobustCostDecision::Admit
        );
        value.minimum_net_savings_ns = 1_000;
        assert_eq!(
            evaluate_robust_cost(value).decision,
            RobustCostDecision::UnamortizedTotalCost
        );
    }

    #[test]
    fn baseline_setup_is_subtracted_once() {
        let epochs = [(20_000, 10_000); 3];
        let mut value = inputs(&epochs, 100);
        value.baseline_native_setup_ns = 2_000_000;
        let report = evaluate_robust_cost(value);
        assert_eq!(report.conservative_extra_budget_ns, 0);
        assert_eq!(report.required_calls, Some(1));
        assert_eq!(report.decision, RobustCostDecision::Admit);
    }

    #[test]
    fn extreme_u64_values_do_not_overflow() {
        let epochs = [(u64::MAX, 1); 3];
        let mut value = inputs(&epochs, u64::MAX);
        value.probe_wall_ns = u64::MAX;
        value.second_selector_budget_ns = u64::MAX;
        value.graph_native_setup_ns = u64::MAX;
        let report = evaluate_robust_cost(value);
        assert_eq!(report.decision, RobustCostDecision::Admit);
        assert_eq!(report.minimum_paired_gain_ns, Some(u64::MAX - 1));
    }

    #[test]
    fn zero_calls_never_admits() {
        let report = evaluate_robust_cost(inputs(&[(10_000, 9_000); 3], 0));
        assert_eq!(report.decision, RobustCostDecision::UnamortizedTotalCost);
    }
}
