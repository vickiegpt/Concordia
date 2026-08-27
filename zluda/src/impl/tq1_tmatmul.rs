use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock, RwLock};
use std::time::UNIX_EPOCH;

pub(crate) const TQ1_VALUES: usize = 256;
pub(crate) const TQ1_BLOCK_BYTES: usize = 54;
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ExpertRole {
    Gate,
    Up,
    Down,
    GateUp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Tq1TensorRegistration {
    pub(crate) path: PathBuf,
    pub(crate) file_offset: u64,
    pub(crate) nbytes: u64,
    pub(crate) name: String,
    pub(crate) ne: [u64; 4],
    pub(crate) nb: [u64; 4],
    pub(crate) role: ExpertRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct TensorIdentity {
    pub(crate) canonical_path: PathBuf,
    pub(crate) file_offset: u64,
    pub(crate) nbytes: u64,
    pub(crate) name: String,
    pub(crate) ne: [u64; 4],
    pub(crate) nb: [u64; 4],
    pub(crate) role: ExpertRole,
    pub(crate) device: u64,
    pub(crate) inode: u64,
    pub(crate) modified_ns: u128,
}

#[derive(Debug)]
pub(crate) struct Tq1TensorSource {
    pub(crate) identity: TensorIdentity,
    file: File,
}

#[derive(Debug, Default)]
pub(crate) struct TensorRegistry {
    sources: RwLock<HashMap<String, Arc<Tq1TensorSource>>>,
}

static TENSOR_REGISTRY: OnceLock<TensorRegistry> = OnceLock::new();

pub(crate) fn classify_tensor_name(bytes: &[u8]) -> Result<ExpertRole, String> {
    let name = std::str::from_utf8(bytes)
        .map_err(|_| "TQ1_0 tensor name is not valid UTF-8".to_string())?;
    let remainder = name
        .strip_prefix("blk.")
        .ok_or_else(|| "invalid TQ1_0 expert tensor name".to_string())?;
    let (block, projection) = remainder
        .split_once(".ffn_")
        .ok_or_else(|| "invalid TQ1_0 expert tensor name".to_string())?;
    if block.is_empty() || !block.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("invalid TQ1_0 expert tensor name".to_string());
    }
    let role = match projection {
        "gate_exps.weight" => ExpertRole::Gate,
        "up_exps.weight" => ExpertRole::Up,
        "down_exps.weight" => ExpertRole::Down,
        "gate_up_exps.weight" => ExpertRole::GateUp,
        _ => return Err("invalid TQ1_0 expert tensor name".to_string()),
    };
    Ok(role)
}

fn validate_tensor_name(bytes: &[u8], role: ExpertRole) -> Result<(), String> {
    let expected_role = classify_tensor_name(bytes)?;
    if role != expected_role {
        return Err("TQ1_0 expert tensor role does not agree with its name".to_string());
    }
    Ok(())
}

fn checked_layout(registration: &Tq1TensorRegistration) -> Result<(), String> {
    validate_tensor_name(registration.name.as_bytes(), registration.role)?;
    if registration.ne[0] == 0 || !registration.ne[0].is_multiple_of(TQ1_VALUES as u64) {
        return Err("TQ1_0 K must be a nonzero multiple of 256".to_string());
    }
    if registration.ne[1] == 0 || registration.ne[2] == 0 {
        return Err("TQ1_0 row and expert dimensions must be positive".to_string());
    }
    if registration.ne[3] != 1 {
        return Err("TQ1_0 ne[3] must equal 1".to_string());
    }
    if registration.nb[0] != TQ1_BLOCK_BYTES as u64 {
        return Err("TQ1_0 nb[0] must equal 54".to_string());
    }

    let blocks_per_row = registration.ne[0] / TQ1_VALUES as u64;
    let packed_row_bytes = blocks_per_row
        .checked_mul(registration.nb[0])
        .ok_or_else(|| "TQ1_0 row size overflow".to_string())?;
    if registration.nb[1] < packed_row_bytes {
        return Err("TQ1_0 row stride does not contain all K blocks".to_string());
    }
    let packed_expert_bytes = registration.ne[1]
        .checked_mul(registration.nb[1])
        .ok_or_else(|| "TQ1_0 expert size overflow".to_string())?;
    if registration.nb[2] < packed_expert_bytes {
        return Err("TQ1_0 expert stride does not contain all rows".to_string());
    }
    let packed_outer_bytes = registration.ne[2]
        .checked_mul(registration.nb[2])
        .ok_or_else(|| "TQ1_0 outer size overflow".to_string())?;
    if registration.nb[3] < packed_outer_bytes {
        return Err("TQ1_0 outer stride does not contain all experts".to_string());
    }

    let required = registration.ne[2]
        .checked_sub(1)
        .and_then(|expert| expert.checked_mul(registration.nb[2]))
        .and_then(|prefix| {
            registration.ne[1]
                .checked_sub(1)
                .and_then(|row| row.checked_mul(registration.nb[1]))
                .and_then(|row_offset| prefix.checked_add(row_offset))
        })
        .and_then(|prefix| {
            blocks_per_row
                .checked_sub(1)
                .and_then(|block| block.checked_mul(registration.nb[0]))
                .and_then(|block_offset| prefix.checked_add(block_offset))
        })
        .and_then(|prefix| prefix.checked_add(TQ1_BLOCK_BYTES as u64))
        .ok_or_else(|| "TQ1_0 tensor span overflow".to_string())?;
    if required > registration.nbytes {
        return Err("TQ1_0 tensor span is smaller than its strided layout".to_string());
    }
    Ok(())
}

impl TensorRegistry {
    pub(crate) fn register(
        &self,
        registration: Tq1TensorRegistration,
    ) -> Result<Arc<Tq1TensorSource>, String> {
        checked_layout(&registration)?;
        let canonical_path = registration
            .path
            .canonicalize()
            .map_err(|error| format!("canonicalize TQ1_0 tensor file: {error}"))?;
        let file = OpenOptions::new()
            .read(true)
            .open(&canonical_path)
            .map_err(|error| format!("open TQ1_0 tensor file read-only: {error}"))?;
        let metadata = file
            .metadata()
            .map_err(|error| format!("inspect TQ1_0 tensor file: {error}"))?;
        if !metadata.is_file() {
            return Err("TQ1_0 tensor source is not a regular file".to_string());
        }
        let file_end = registration
            .file_offset
            .checked_add(registration.nbytes)
            .ok_or_else(|| "TQ1_0 file range overflow".to_string())?;
        if file_end > metadata.len() {
            return Err("TQ1_0 tensor range exceeds file size".to_string());
        }
        let modified_ns = metadata
            .modified()
            .map_err(|error| format!("read TQ1_0 modification time: {error}"))?
            .duration_since(UNIX_EPOCH)
            .map_err(|_| "TQ1_0 modification time predates the Unix epoch".to_string())?
            .as_nanos();
        let identity = TensorIdentity {
            canonical_path,
            file_offset: registration.file_offset,
            nbytes: registration.nbytes,
            name: registration.name,
            ne: registration.ne,
            nb: registration.nb,
            role: registration.role,
            device: metadata.dev(),
            inode: metadata.ino(),
            modified_ns,
        };

        let mut sources = self
            .sources
            .write()
            .map_err(|_| "TQ1_0 tensor registry lock poisoned".to_string())?;
        if let Some(existing) = sources.get(&identity.name) {
            if existing.identity == identity {
                return Ok(Arc::clone(existing));
            }
            return Err(format!(
                "conflicting TQ1_0 registration for {}",
                identity.name
            ));
        }
        let source = Arc::new(Tq1TensorSource { identity, file });
        sources.insert(source.identity.name.clone(), Arc::clone(&source));
        Ok(source)
    }

    pub(crate) fn get(&self, name: &str) -> Result<Option<Arc<Tq1TensorSource>>, String> {
        let sources = self
            .sources
            .read()
            .map_err(|_| "TQ1_0 tensor registry lock poisoned".to_string())?;
        Ok(sources.get(name).cloned())
    }
}

impl Tq1TensorSource {
    pub(crate) fn register(registration: Tq1TensorRegistration) -> Result<Arc<Self>, String> {
        TENSOR_REGISTRY
            .get_or_init(TensorRegistry::default)
            .register(registration)
    }

    pub(crate) fn lookup(name: &str) -> Result<Option<Arc<Self>>, String> {
        TENSOR_REGISTRY
            .get_or_init(TensorRegistry::default)
            .get(name)
    }

    pub(crate) fn read_exact(&self, relative: u64, output: &mut [u8]) -> Result<(), String> {
        let output_len = u64::try_from(output.len())
            .map_err(|_| "TQ1_0 read length does not fit u64".to_string())?;
        let end = relative
            .checked_add(output_len)
            .ok_or_else(|| "TQ1_0 relative file range overflow".to_string())?;
        if end > self.identity.nbytes {
            return Err("TQ1_0 read exceeds registered tensor span".to_string());
        }
        let absolute = self
            .identity
            .file_offset
            .checked_add(relative)
            .ok_or_else(|| "TQ1_0 absolute file range overflow".to_string())?;
        self.file
            .read_exact_at(output, absolute)
            .map_err(|error| format!("read TQ1_0 tensor {}: {error}", self.identity.name))
    }

    pub(crate) fn read_row_blocks(
        &self,
        expert: u64,
        row: u64,
        first_block: u64,
        block_count: u64,
    ) -> Result<Vec<u8>, String> {
        if expert >= self.identity.ne[2] {
            return Err("TQ1_0 expert index is out of bounds".to_string());
        }
        if row >= self.identity.ne[1] {
            return Err("TQ1_0 row index is out of bounds".to_string());
        }
        let blocks_per_row = self.identity.ne[0] / TQ1_VALUES as u64;
        let block_end = first_block
            .checked_add(block_count)
            .ok_or_else(|| "TQ1_0 block range overflow".to_string())?;
        if block_count == 0 || block_end > blocks_per_row {
            return Err("TQ1_0 block range is out of bounds".to_string());
        }
        let relative = expert
            .checked_mul(self.identity.nb[2])
            .and_then(|value| {
                row.checked_mul(self.identity.nb[1])
                    .and_then(|offset| value.checked_add(offset))
            })
            .and_then(|value| {
                first_block
                    .checked_mul(self.identity.nb[0])
                    .and_then(|offset| value.checked_add(offset))
            })
            .ok_or_else(|| "TQ1_0 row address overflow".to_string())?;
        let byte_count = block_count
            .checked_mul(TQ1_BLOCK_BYTES as u64)
            .ok_or_else(|| "TQ1_0 row byte count overflow".to_string())?;
        let mut output = vec![
            0u8;
            usize::try_from(byte_count).map_err(|_| {
                "TQ1_0 row byte count does not fit usize".to_string()
            })?
        ];
        self.read_exact(relative, &mut output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Tq1Block {
    pub(crate) trits: [i8; TQ1_VALUES],
    pub(crate) scale: f32,
}

fn decode_digit(byte: u8, power: u8) -> i8 {
    let q = byte.wrapping_mul(power);
    ((((q as u16) * 3) >> 8) as i16 - 1) as i8
}

impl Tq1Block {
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != TQ1_BLOCK_BYTES {
            return Err(format!(
                "TQ1_0 block has {} bytes, expected 54",
                bytes.len()
            ));
        }

        let mut trits = [0i8; TQ1_VALUES];
        let mut out = 0;
        for j in (0..32).step_by(32) {
            for n in 0..5 {
                for m in 0..32 {
                    trits[out] = decode_digit(bytes[j + m], POW3[n]);
                    out += 1;
                }
            }
        }
        for j in (32..48).step_by(16) {
            for n in 0..5 {
                for m in 0..16 {
                    trits[out] = decode_digit(bytes[j + m], POW3[n]);
                    out += 1;
                }
            }
        }
        for n in 0..4 {
            for j in 0..4 {
                trits[out] = decode_digit(bytes[48 + j], POW3[n]);
                out += 1;
            }
        }
        if out != TQ1_VALUES {
            return Err(format!("TQ1_0 decoded {out} values, expected 256"));
        }

        let scale = super::iq1s_tmatmul::half_to_f32(u16::from_le_bytes([bytes[52], bytes[53]]));
        if !scale.is_finite() {
            return Err("TQ1_0 scale is not finite".to_string());
        }
        Ok(Self { trits, scale })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Q8KBlock {
    pub(crate) qs: [i8; TQ1_VALUES],
    pub(crate) scale: f32,
}

fn nearest_int(value: f32) -> Result<i32, String> {
    if !value.is_finite() || value.abs() > 4_194_303.0 {
        return Err("Q8_K rounding input is outside the upstream bound".to_string());
    }
    let bits = (value + 12_582_912.0).to_bits() as i32;
    Ok((bits & 0x007f_ffff) - 0x0040_0000)
}

impl Q8KBlock {
    pub(crate) fn quantize(values: &[f32]) -> Result<Self, String> {
        if values.len() != TQ1_VALUES || values.iter().any(|value| !value.is_finite()) {
            return Err("Q8_K requires 256 finite f32 values".to_string());
        }

        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &value in values {
            if value.abs() > amax {
                amax = value.abs();
                max = value;
            }
        }
        if amax == 0.0 {
            return Ok(Self {
                qs: [0; TQ1_VALUES],
                scale: 0.0,
            });
        }

        let iscale = -127.0 / max;
        let mut qs = [0i8; TQ1_VALUES];
        for (dst, &value) in qs.iter_mut().zip(values) {
            *dst = nearest_int(iscale * value)?.min(127) as i8;
        }
        Ok(Self {
            qs,
            scale: 1.0 / iscale,
        })
    }
}

pub(crate) fn reference_dot(weights: &[Tq1Block], activations: &[f32]) -> Result<f32, String> {
    if activations.is_empty() || !activations.len().is_multiple_of(TQ1_VALUES) {
        return Err("activation K must be a nonzero multiple of 256".to_string());
    }
    if weights.len() != activations.len() / TQ1_VALUES {
        return Err("TQ1_0 block count does not match activation K".to_string());
    }

    let mut result = 0.0f32;
    for (weight, values) in weights.iter().zip(activations.chunks_exact(TQ1_VALUES)) {
        let activation = Q8KBlock::quantize(values)?;
        let integer_dot = weight
            .trits
            .iter()
            .zip(activation.qs)
            .map(|(&weight, activation)| i32::from(weight) * i32::from(activation))
            .sum::<i32>();
        result += integer_dot as f32 * weight.scale * activation.scale;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn fixture_registration(
        path: &std::path::Path,
        name: &str,
        role: ExpertRole,
    ) -> Tq1TensorRegistration {
        Tq1TensorRegistration {
            path: path.to_path_buf(),
            file_offset: 0,
            nbytes: 2 * 2 * 4 * TQ1_BLOCK_BYTES as u64,
            name: name.to_string(),
            ne: [1024, 2, 2, 1],
            nb: [54, 216, 432, 864],
            role,
        }
    }

    fn fixture_tensor_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        for expert in 0..2u8 {
            for row in 0..2u8 {
                for block in 0..4u8 {
                    file.write_all(&[expert * 64 + row * 16 + block; TQ1_BLOCK_BYTES])
                        .unwrap();
                }
            }
        }
        file.flush().unwrap();
        file
    }

    fn fixture_digit(byte: u8, power: u8) -> i8 {
        let q = byte.wrapping_mul(power);
        ((((q as u16) * 3) >> 8) as i16 - 1) as i8
    }

    fn upstream_decode_fixture(bytes: &[u8; 54]) -> [i8; 256] {
        const POWERS: [u8; 5] = [1, 3, 9, 27, 81];
        let mut expected = [0i8; 256];
        let mut out = 0;
        for n in 0..5 {
            for m in 0..32 {
                expected[out] = fixture_digit(bytes[m], POWERS[n]);
                out += 1;
            }
        }
        for n in 0..5 {
            for m in 0..16 {
                expected[out] = fixture_digit(bytes[32 + m], POWERS[n]);
                out += 1;
            }
        }
        for n in 0..4 {
            for j in 0..4 {
                expected[out] = fixture_digit(bytes[48 + j], POWERS[n]);
                out += 1;
            }
        }
        assert_eq!(out, 256);
        expected
    }

    fn upstream_q8_fixture(values: &[f32]) -> [i8; 256] {
        let max = values
            .iter()
            .copied()
            .max_by(|left, right| left.abs().total_cmp(&right.abs()))
            .unwrap();
        if max == 0.0 {
            return [0; 256];
        }
        let iscale = -127.0 / max;
        let mut expected = [0i8; 256];
        for (dst, value) in expected.iter_mut().zip(values) {
            let bits = (iscale * value + 12_582_912.0).to_bits() as i32;
            *dst = ((bits & 0x007f_ffff) - 0x0040_0000).min(127) as i8;
        }
        expected
    }

    #[test]
    fn tq1_decode_covers_payload_tail_and_scale() {
        let mut bytes = [0u8; TQ1_BLOCK_BYTES];
        for (index, byte) in bytes[..48].iter_mut().enumerate() {
            *byte = (index as u8).wrapping_mul(37).wrapping_add(11);
        }
        bytes[48..52].copy_from_slice(&[3, 81, 127, 242]);
        bytes[52..54].copy_from_slice(&0x3800u16.to_le_bytes());

        let block = Tq1Block::decode(&bytes).unwrap();

        assert_eq!(block.scale, 0.5);
        assert_eq!(block.trits.len(), 256);
        assert!(block.trits.iter().all(|value| (-1..=1).contains(value)));
        assert_eq!(block.trits, upstream_decode_fixture(&bytes));
    }

    #[test]
    fn tq1_decode_rejects_wrong_size_and_nonfinite_scale() {
        assert!(Tq1Block::decode(&[0; 53])
            .unwrap_err()
            .contains("expected 54"));
        assert!(Tq1Block::decode(&[0; 55])
            .unwrap_err()
            .contains("expected 54"));

        let mut bytes = [0u8; TQ1_BLOCK_BYTES];
        bytes[52..54].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(Tq1Block::decode(&bytes)
            .unwrap_err()
            .contains("scale is not finite"));
    }

    #[test]
    fn q8_k_matches_upstream_rounding_and_zero_block() {
        let values: Vec<f32> = (0..256)
            .map(|index| ((index as i32 - 127) as f32) / 31.0)
            .collect();
        let quantized = Q8KBlock::quantize(&values).unwrap();

        assert_eq!(quantized.qs, upstream_q8_fixture(&values));
        assert!(quantized.scale.is_finite());
        assert_eq!(Q8KBlock::quantize(&[0.0; 256]).unwrap().scale, 0.0);

        let positive = Q8KBlock::quantize(&[1.0; 256]).unwrap();
        assert_eq!(positive.qs, [-127; 256]);
        assert_eq!(positive.scale, -1.0 / 127.0);
        let negative = Q8KBlock::quantize(&[-1.0; 256]).unwrap();
        assert_eq!(negative.qs, [-127; 256]);
        assert_eq!(negative.scale, 1.0 / 127.0);
    }

    #[test]
    fn q8_k_rejects_wrong_size_and_nonfinite_values() {
        assert!(Q8KBlock::quantize(&[0.0; 255]).is_err());
        let mut values = [0.0; 256];
        values[7] = f32::NAN;
        assert!(Q8KBlock::quantize(&values).is_err());
        values[7] = f32::INFINITY;
        assert!(Q8KBlock::quantize(&values).is_err());
    }

    #[test]
    fn tq1_q8_dot_uses_per_block_scales() {
        let weights = [0.5, 1.0, 2.0, 4.0].map(|scale| Tq1Block {
            trits: [-1; TQ1_VALUES],
            scale,
        });
        let activations = [1.0; 1024];

        assert_eq!(reference_dot(&weights, &activations).unwrap(), -1920.0);
        assert!(reference_dot(&weights[..1], &activations[..255])
            .unwrap_err()
            .contains("multiple of 256"));
    }

    #[test]
    fn registry_reads_only_the_registered_expert_range() {
        let file = fixture_tensor_file();
        let registration =
            fixture_registration(file.path(), "blk.0.ffn_down_exps.weight", ExpertRole::Down);
        let source = Tq1TensorSource::register(registration).unwrap();

        let first = source.read_row_blocks(0, 0, 0, 4).unwrap();
        let second = source.read_row_blocks(1, 0, 0, 4).unwrap();

        assert_ne!(first, second);
        assert_eq!(first.len(), 4 * TQ1_BLOCK_BYTES);
        assert_eq!(&first[..TQ1_BLOCK_BYTES], &[0; TQ1_BLOCK_BYTES]);
        assert_eq!(&second[..TQ1_BLOCK_BYTES], &[64; TQ1_BLOCK_BYTES]);
        assert!(source.read_row_blocks(2, 0, 0, 1).is_err());
        assert!(source.read_row_blocks(0, 2, 0, 1).is_err());
        assert!(source.read_row_blocks(0, 0, 3, 2).is_err());
    }

    #[test]
    fn registry_rejects_changed_metadata_and_accepts_identical_duplicate() {
        let file = fixture_tensor_file();
        let registry = TensorRegistry::default();
        let registration =
            fixture_registration(file.path(), "blk.1.ffn_up_exps.weight", ExpertRole::Up);
        let first = registry.register(registration.clone()).unwrap();
        let second = registry.register(registration.clone()).unwrap();
        assert!(Arc::ptr_eq(&first, &second));

        let mut changed = registration;
        changed.nb[3] += 1;
        let error = registry.register(changed).unwrap_err();
        assert!(error.contains("conflicting TQ1_0 registration"));
    }

    #[cfg(unix)]
    #[test]
    fn registry_canonicalizes_symlink_and_opens_read_only() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let target = directory.path().join("weights.gguf");
        std::fs::write(&target, vec![0u8; 864]).unwrap();
        let link = directory.path().join("weights-link.gguf");
        symlink(&target, &link).unwrap();
        let registry = TensorRegistry::default();
        let source = registry
            .register(fixture_registration(
                &link,
                "blk.2.ffn_gate_exps.weight",
                ExpertRole::Gate,
            ))
            .unwrap();

        assert_eq!(
            source.identity.canonical_path,
            target.canonicalize().unwrap()
        );
        assert!(source.file.metadata().unwrap().is_file());
        let mut opened = source.file.try_clone().unwrap();
        assert!(opened.write_all(b"must fail").is_err());
    }

    #[test]
    fn registry_rejects_bad_ranges_layout_names_and_roles() {
        let file = fixture_tensor_file();
        let registry = TensorRegistry::default();
        let valid = fixture_registration(
            file.path(),
            "blk.3.ffn_gate_up_exps.weight",
            ExpertRole::GateUp,
        );

        let mut case = valid.clone();
        case.file_offset = u64::MAX;
        assert!(registry.register(case).unwrap_err().contains("overflow"));
        let mut case = valid.clone();
        case.nbytes += 1;
        assert!(registry.register(case).unwrap_err().contains("file size"));
        let mut case = valid.clone();
        case.ne[3] = 2;
        assert!(registry.register(case).unwrap_err().contains("ne[3]"));
        let mut case = valid.clone();
        case.ne[0] = 1000;
        assert!(registry
            .register(case)
            .unwrap_err()
            .contains("multiple of 256"));
        let mut case = valid.clone();
        case.nb[0] = 53;
        assert!(registry.register(case).unwrap_err().contains("nb[0]"));
        let mut case = valid.clone();
        case.nb[2] = case.nb[1];
        assert!(registry
            .register(case)
            .unwrap_err()
            .contains("expert stride"));
        let mut case = valid.clone();
        case.name = "blk.x.ffn_gate_up_exps.weight".to_string();
        assert!(registry.register(case).unwrap_err().contains("tensor name"));
        let mut case = valid;
        case.role = ExpertRole::Down;
        assert!(registry.register(case).unwrap_err().contains("role"));

        assert!(validate_tensor_name(&[0xff], ExpertRole::Down)
            .unwrap_err()
            .contains("UTF-8"));
        assert!(registry
            .register(fixture_registration(
                file.path().parent().unwrap(),
                "blk.4.ffn_down_exps.weight",
                ExpertRole::Down,
            ))
            .unwrap_err()
            .contains("regular file"));
    }
}
