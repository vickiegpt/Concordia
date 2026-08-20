#![cfg(all(feature = "nvidia", feature = "evaluation"))]

use nvcuda::hetgpu_v3_batch_plan_for_evaluation;

#[test]
fn exact_and_remainder_plans_use_the_real_scheduler_contract() {
    assert_eq!(
        hetgpu_v3_batch_plan_for_evaluation(8, 4, None).unwrap(),
        vec![(0, 4), (4, 4)]
    );
    assert_eq!(
        hetgpu_v3_batch_plan_for_evaluation(10, 4, None).unwrap(),
        vec![(0, 4), (4, 4), (8, 2)]
    );
}

#[test]
fn configured_limit_may_lower_but_never_raise_live_capability() {
    assert_eq!(
        hetgpu_v3_batch_plan_for_evaluation(5, 4, Some("2")).unwrap(),
        vec![(0, 2), (2, 2), (4, 1)]
    );

    let error = hetgpu_v3_batch_plan_for_evaluation(5, 4, Some("5")).unwrap_err();
    assert!(error.contains("outside 1..=4"), "{error}");
}

#[test]
fn zero_and_malformed_inputs_are_rejected() {
    for result in [
        hetgpu_v3_batch_plan_for_evaluation(0, 4, None),
        hetgpu_v3_batch_plan_for_evaluation(1, 0, None),
        hetgpu_v3_batch_plan_for_evaluation(1, 4, Some("0")),
        hetgpu_v3_batch_plan_for_evaluation(1, 4, Some("malformed")),
    ] {
        assert!(result.is_err());
    }
}
