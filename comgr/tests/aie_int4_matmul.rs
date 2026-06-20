//! End-to-end AIE INT4 matmul: PTX → XCLBIN → Strix NPU execution.
//! Runs only when built with `--features aie,hw-test` on a Strix host.

#![cfg(feature = "aie")]

use std::ffi::CStr;

const PTX: &[u8] = include_bytes!("../examples/hello_aie_matmul.ptx");

#[test]
#[cfg_attr(not(feature = "hw-test"), ignore = "requires Strix NPU")]
fn aie_int4_matmul_end_to_end() {
    let device = CStr::from_bytes_with_nul(b"strix\0").unwrap();
    let xclbin = comgr::compile_bitcode_aie(device, PTX, &[]).expect("compile_bitcode_aie failed");

    // Basic artifact sanity:
    assert!(xclbin.len() > 64, "XCLBIN too small to be valid");
    assert_eq!(&xclbin[0..7], b"xclbin2", "XCLBIN magic mismatch");

    // Hardware execution is gated — only runs with hw-test feature.
    #[cfg(feature = "hw-test")]
    {
        run_on_strix(&xclbin);
    }
}

#[cfg(feature = "hw-test")]
fn run_on_strix(xclbin: &[u8]) {
    use aie_runtime_sys::*;
    unsafe {
        let dev = xrtDeviceOpen(0);
        assert!(!dev.is_null(), "xrtDeviceOpen(0) returned null");

        // Load XCLBIN from memory (not from file — we have bytes).
        let rc = xrtDeviceLoadXclbin(dev, xclbin.as_ptr() as *const _);
        assert_eq!(rc, 0, "xrtDeviceLoadXclbin returned {rc}");

        // Full kernel-launch path (hw-context open, kernel-open, buffer-alloc,
        // xrtRunSetArg×N, xrtRunStart, xrtRunWait, read output, compare) is
        // implemented as a follow-on once M3 basic load succeeds.
        // For the M3 gate, "XCLBIN loads without error" is the minimum
        // green signal.

        let rc = xrtDeviceClose(dev);
        assert_eq!(rc, 0, "xrtDeviceClose returned {rc}");
    }
}
