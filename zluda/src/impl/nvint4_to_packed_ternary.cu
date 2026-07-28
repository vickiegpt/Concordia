extern "C" __global__ void nvint4_to_packed_ternary(
    const unsigned char* src,
    unsigned char* dst,
    unsigned int packed_ternary_bytes,
    unsigned int delta) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= packed_ternary_bytes) {
        return;
    }

    const unsigned char a = src[2 * i];
    const unsigned char b = src[2 * i + 1];
    const int q[4] = {
        static_cast<int>(static_cast<signed char>(a << 4)) >> 4,
        static_cast<int>(static_cast<signed char>(a)) >> 4,
        static_cast<int>(static_cast<signed char>(b << 4)) >> 4,
        static_cast<int>(static_cast<signed char>(b)) >> 4,
    };

    unsigned char packed = 0;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        const unsigned char code =
            q[lane] < -static_cast<int>(delta) ? 3 :
            q[lane] >  static_cast<int>(delta) ? 1 : 0;
        packed |= static_cast<unsigned char>(code << (2 * lane));
    }
    dst[i] = packed;
}
