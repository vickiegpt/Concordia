//! Hardware smoke test: open device 0, verify XRT bindings link and run.
//! Gated behind `hw-test` feature so CI skips it.
//!
//! Run with: `cargo test -p aie_runtime_sys --test smoke --features hw-test -- --ignored`

#[cfg(not(feature = "hw-test"))]
#[test]
#[ignore = "requires Strix NPU and hw-test feature"]
fn open_device_zero() {
    // Intentionally empty — when hw-test feature is off, this test is
    // ignored and does nothing.
}

#[cfg(feature = "hw-test")]
#[test]
fn open_device_zero() {
    use aie_runtime_sys::*;
    unsafe {
        let dev = xrtDeviceOpen(0);
        assert!(!dev.is_null(), "xrtDeviceOpen(0) returned null — is amdxdna loaded?");
        let rc = xrtDeviceClose(dev);
        assert_eq!(rc, 0, "xrtDeviceClose returned {rc}");
    }
}
