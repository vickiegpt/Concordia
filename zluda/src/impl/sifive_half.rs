// CUDA __float2half uses round-to-nearest, ties-to-even, including subnormals.
pub(super) fn from_f32(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | if frac == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 112;
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    if half_exp < -10 {
        return sign;
    }
    let (base, remainder, halfway) = if half_exp <= 0 {
        let mant = frac | 0x80_0000;
        let shift = (14 - half_exp) as u32;
        (mant >> shift, mant & ((1 << shift) - 1), 1 << (shift - 1))
    } else {
        (((half_exp as u32) << 10) | (frac >> 13), frac & 0x1fff, 0x1000)
    };
    let round_up = remainder > halfway || (remainder == halfway && base & 1 != 0);
    sign | (base + u32::from(round_up)) as u16
}

#[cfg(test)]
mod tests {
    use super::from_f32;

    // Arithmetic reference independent of the conversion's bit-shift algorithm.
    fn positive_half(h: u16) -> f32 {
        let exp = h >> 10;
        let frac = h & 1023;
        if exp == 0 { (frac as f32) * 2.0f32.powi(-24) }
        else { (1.0 + frac as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15) }
    }

    #[test]
    fn all_finite_half_values_and_rounding_boundaries() {
        for h in 0u16..=0x7bff {
            let x = positive_half(h);
            assert_eq!(from_f32(x), h);
            assert_eq!(from_f32(-x), h | 0x8000);
            if h == 0x7bff { continue; }
            let mid = (x + positive_half(h + 1)) * 0.5;
            let even = h + (h & 1);
            assert_eq!(from_f32(mid), even, "midpoint {h:#x}");
            assert_eq!(from_f32(-mid), even | 0x8000);
            assert_eq!(from_f32(f32::from_bits(mid.to_bits() - 1)), h);
            assert_eq!(from_f32(f32::from_bits(mid.to_bits() + 1)), h + 1);
        }
    }

    #[test]
    fn overflow_specials_and_quantization_scale() {
        assert_eq!(from_f32(65520.0), 0x7c00);
        assert_eq!(from_f32(-65520.0), 0xfc00);
        assert_eq!(from_f32(f32::from_bits(65520.0f32.to_bits() - 1)), 0x7bff);
        assert_eq!(from_f32(f32::INFINITY), 0x7c00);
        assert_eq!(from_f32(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(from_f32(f32::NAN) & 0x7fff, 0x7e00);
        assert_eq!(from_f32(1.0 / 127.0), 0x2008);
        assert_eq!(from_f32(f32::from_bits(1)), 0);
    }
}
