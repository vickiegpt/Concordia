#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_smoke_program_matches_v2_runner() {
        let prog = encode_smoke_program();

        assert_eq!(prog.len(), 96);
        assert_eq!(&prog[0..8], &TMATMUL_DPA_INPUT.to_le_bytes());
        assert_eq!(&prog[32..40], &TMATMUL_DPA_MATRIX.to_le_bytes());
        assert_eq!(&prog[64..72], &TMATMUL_DPA_OUTPUT.to_le_bytes());
    }

    #[test]
    fn layout_requires_program_end() {
        assert_eq!(
            required_dax_len(),
            TMATMUL_DPA_PROGRAM as usize + TMATMUL_PROGRAM_BYTES
        );
    }

    #[test]
    fn validates_supported_sizes() {
        let dim = 512;

        assert_eq!(matrix_bytes(dim).unwrap(), 512 * 512 / 4);
        assert_eq!(vector_bytes(dim).unwrap(), 512 * 2);
        assert!(validate_allocations(dim, 1024, 1024, 65536).is_ok());
        assert!(validate_allocations(dim, 1023, 1024, 65536).is_err());
    }
}
