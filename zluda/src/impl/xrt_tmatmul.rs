use std::collections::HashMap;
use std::fmt;

const AXI_INSTANCE_STRIDE: u32 = 0x4000;
const MM2S_DMACR: u32 = 0x0000;
const MM2S_SA: u32 = 0x0018;
const MM2S_LENGTH: u32 = 0x0028;
const STALL: u32 = 0x1000;
const INSTRUCTION_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InstanceRegisters {
    dma_control: u32,
    dma_source_lo: u32,
    dma_source_hi: u32,
    dma_length: u32,
    stall: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum XrtTmatmulError {
    Config(String),
    DynamicLoad(String),
    Xrt { operation: &'static str, code: i32 },
    NullHandle(&'static str),
    InvalidProgram(String),
    InvalidBuffer(String),
    Assemble(String),
    Timeout { timeout_ms: u32 },
}

impl fmt::Display for XrtTmatmulError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(message) => write!(f, "XRT tmatmul configuration failed: {message}"),
            Self::DynamicLoad(message) => write!(f, "XRT runtime loading failed: {message}"),
            Self::Xrt { operation, code } => {
                write!(f, "XRT {operation} failed with code {code}")
            }
            Self::NullHandle(operation) => write!(f, "XRT {operation} returned a null handle"),
            Self::InvalidProgram(message) => write!(f, "invalid tmatmul program: {message}"),
            Self::InvalidBuffer(message) => write!(f, "invalid tmatmul buffer: {message}"),
            Self::Assemble(message) => write!(f, "tmatmul assembly failed: {message}"),
            Self::Timeout { timeout_ms } => {
                write!(f, "XRT tmatmul timed out after {timeout_ms} ms")
            }
        }
    }
}

impl std::error::Error for XrtTmatmulError {}

fn instance_registers(instance: u32) -> Result<InstanceRegisters, XrtTmatmulError> {
    let base = instance.checked_mul(AXI_INSTANCE_STRIDE).ok_or_else(|| {
        XrtTmatmulError::Config(format!(
            "instance {instance} overflows the register aperture"
        ))
    })?;
    let add = |offset: u32| {
        base.checked_add(offset).ok_or_else(|| {
            XrtTmatmulError::Config(format!("instance {instance} register offset overflow"))
        })
    };

    Ok(InstanceRegisters {
        dma_control: add(MM2S_DMACR)?,
        dma_source_lo: add(MM2S_SA)?,
        dma_source_hi: add(MM2S_SA + 4)?,
        dma_length: add(MM2S_LENGTH)?,
        stall: add(STALL)?,
    })
}

fn validate_program(program: &[u8]) -> Result<(), XrtTmatmulError> {
    if program.is_empty() {
        return Err(XrtTmatmulError::InvalidProgram(
            "program is empty".to_string(),
        ));
    }
    if program.len() % INSTRUCTION_BYTES != 0 {
        return Err(XrtTmatmulError::InvalidProgram(format!(
            "{} bytes is not a multiple of {INSTRUCTION_BYTES}",
            program.len()
        )));
    }
    u32::try_from(program.len()).map_err(|_| {
        XrtTmatmulError::InvalidProgram("program length does not fit MM2S_LENGTH".to_string())
    })?;
    Ok(())
}

fn bind_labels(
    matrix_label: &str,
    input_label: &str,
    output_label: &str,
    matrix_address: u64,
    input_address: u64,
    output_address: u64,
) -> Result<HashMap<String, u64>, XrtTmatmulError> {
    if matrix_label.is_empty() || input_label.is_empty() || output_label.is_empty() {
        return Err(XrtTmatmulError::Config(
            "BO labels must not be empty".to_string(),
        ));
    }
    if matrix_label == input_label || matrix_label == output_label || input_label == output_label {
        return Err(XrtTmatmulError::Config(
            "matrix, input, and output labels must be distinct".to_string(),
        ));
    }
    if matrix_address == 0 || input_address == 0 || output_address == 0 {
        return Err(XrtTmatmulError::InvalidBuffer(
            "XRT returned a zero BO address".to_string(),
        ));
    }

    Ok(HashMap::from([
        (matrix_label.to_owned(), matrix_address),
        (input_label.to_owned(), input_address),
        (output_label.to_owned(), output_address),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instance_registers_match_ternary_matmul_contract() {
        assert_eq!(
            instance_registers(0).unwrap(),
            InstanceRegisters {
                dma_control: 0x0000,
                dma_source_lo: 0x0018,
                dma_source_hi: 0x001c,
                dma_length: 0x0028,
                stall: 0x1000,
            }
        );
        assert_eq!(instance_registers(2).unwrap().stall, 0x9000);
    }

    #[test]
    fn program_requires_complete_128_bit_words() {
        assert!(validate_program(&[0; 16]).is_ok());
        assert!(matches!(
            validate_program(&[]),
            Err(XrtTmatmulError::InvalidProgram(_))
        ));
        assert!(matches!(
            validate_program(&[0; 17]),
            Err(XrtTmatmulError::InvalidProgram(_))
        ));
    }

    #[test]
    fn labels_bind_to_real_bo_addresses() {
        let labels = bind_labels("PARAM_0", "PARAM_1", "PARAM_2", 0x1000, 0x2000, 0x3000).unwrap();
        assert_eq!(labels["PARAM_0"], 0x1000);
        assert_eq!(labels["PARAM_1"], 0x2000);
        assert_eq!(labels["PARAM_2"], 0x3000);
    }

    #[test]
    fn duplicate_labels_are_rejected() {
        let error = bind_labels("PARAM_0", "PARAM_0", "PARAM_2", 1, 2, 3).unwrap_err();
        assert!(error.to_string().contains("distinct"));
    }
}
