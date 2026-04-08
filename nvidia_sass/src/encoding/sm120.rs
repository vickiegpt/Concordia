use crate::types::*;

pub fn encode(inst: &SassInst) -> Result<[u8; 16], NvSassError> {
    let inst_lo = encode_instruction_bits(inst)?;
    let ctrl_hi = super::control_codes::encode(&inst.control);
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&inst_lo.to_le_bytes());
    bytes[8..].copy_from_slice(&ctrl_hi.to_le_bytes());
    Ok(bytes)
}

fn encode_instruction_bits(_inst: &SassInst) -> Result<u64, NvSassError> {
    // Stub - will be fully implemented in Task 2
    Ok(0x7FF_u64 << 52 | 7u64 << 16) // NOP with PT
}

pub fn lookup_opcode_bits(_mnemonic: &str) -> Result<u16, NvSassError> {
    // Stub - will be fully implemented in Task 2
    Ok(0x7FF)
}
