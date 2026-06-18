use ptx::sass::fuzz::{run_sass_lifter_fuzzer, SassLifterFuzzConfig};

#[test]
fn sass_lifter_fuzzer_is_deterministic_and_generates_parseable_ptx() {
    let config = SassLifterFuzzConfig {
        seed: 0x5a55_1200,
        cases: 24,
        max_instructions: 12,
        sm_version: 120,
        parse_lifted_ptx: true,
    };

    let first = run_sass_lifter_fuzzer(config.clone()).expect("fuzz run should pass");
    let second = run_sass_lifter_fuzzer(config).expect("fuzz run should be repeatable");

    assert_eq!(first, second);
    assert_eq!(first.cases, 24);
    assert!(first.instructions >= 24);
    assert_eq!(first.lift_diagnostics, 0);
    assert_eq!(first.parse_failures, 0);
}

#[test]
fn sass_lifter_fuzzer_rejects_empty_runs() {
    let config = SassLifterFuzzConfig {
        seed: 1,
        cases: 0,
        max_instructions: 8,
        sm_version: 120,
        parse_lifted_ptx: true,
    };

    let error = run_sass_lifter_fuzzer(config).expect_err("zero cases should be rejected");
    assert!(error.to_string().contains("cases must be greater than zero"));
}
