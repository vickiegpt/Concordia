#ifndef NVINT4_CUDA_GLIBC_COMPAT_H
#define NVINT4_CUDA_GLIBC_COMPAT_H
#define __MATH_FUNCTIONS_H__
#if defined(__CUDACC__)
template <typename Left, typename Right>
__attribute__((host)) __attribute__((device)) inline auto min(
    Left left, Right right) -> decltype(left + right) {
    return static_cast<decltype(left + right)>(left < right ? left : right);
}
template <typename Left, typename Right>
__attribute__((host)) __attribute__((device)) inline auto max(
    Left left, Right right) -> decltype(left + right) {
    return static_cast<decltype(left + right)>(left > right ? left : right);
}
#endif
#endif
