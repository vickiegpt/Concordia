extern "C" __global__ void add_one(float *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] + 1.0f;
    }
}
