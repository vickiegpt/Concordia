//! Subprocess-driven pipeline from TOSA MLIR to XCLBIN.
//! Populated in Task 2.

use crate::{AieCompileConfig, AieComgrError};

pub(crate) fn run(
    _tosa_mlir: &str,
    _config: &AieCompileConfig,
) -> Result<Vec<u8>, AieComgrError> {
    Err(AieComgrError::InvalidInput(
        "pipeline::run not yet implemented".to_string(),
    ))
}
