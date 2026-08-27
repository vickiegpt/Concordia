pub(crate) const TQ1_VALUES: usize = 256;
pub(crate) const TQ1_BLOCK_BYTES: usize = 54;
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

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
}
