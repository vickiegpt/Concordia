#include <stddef.h>
#include <stdint.h>

typedef int cudaError_t;
typedef void *cudaGraph_t;
typedef void *cudaGraphNode_t;
typedef void *cudaGraphConditionalHandle;
typedef void *cudaStream_t;
typedef int cudaStreamCaptureMode;

cudaError_t cudaGraphAddNode_v2(cudaGraphNode_t *pGraphNode,
                                cudaGraph_t graph,
                                const cudaGraphNode_t *dependencies,
                                size_t numDependencies,
                                const void *nodeParams) {
    (void)graph;
    (void)dependencies;
    (void)numDependencies;
    (void)nodeParams;
    if (pGraphNode) {
        *pGraphNode = NULL;
    }
    return 0;
}

cudaError_t cudaGraphConditionalHandleCreate(cudaGraphConditionalHandle *pHandle,
                                             cudaGraph_t graph,
                                             unsigned int defaultLaunchValue,
                                             unsigned int flags) {
    (void)graph;
    (void)defaultLaunchValue;
    (void)flags;
    if (pHandle) {
        *pHandle = (cudaGraphConditionalHandle)(uintptr_t)0x1;
    }
    return 0;
}

cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream,
                                          cudaGraph_t graph,
                                          const cudaGraphNode_t *dependencies,
                                          const void *dependencyData,
                                          size_t numDependencies,
                                          cudaStreamCaptureMode mode) {
    (void)stream;
    (void)graph;
    (void)dependencies;
    (void)dependencyData;
    (void)numDependencies;
    (void)mode;
    return 0;
}

void hetgpu_torch_cuda_blas_gemm_half()
    __asm__("_ZN2at4cuda4blas4gemmIN3c104HalfES4_TnPNSt9enable_ifIXntaaoosr3std7is_sameIT_S4_EE5valuesr3std7is_sameIS6_NS3_8BFloat16EEE5valuesr3std7is_sameIT0_fEE5valueES6_E4typeELPS4_0EEEvcclllNS_10OpMathTypeIS6_E4typeEPKS6_lSH_lSF_PS8_l");
void hetgpu_torch_cuda_blas_gemm_half() {}

void hetgpu_torch_cuda_blas_gemm_bfloat16()
    __asm__("_ZN2at4cuda4blas4gemmIN3c108BFloat16ES4_TnPNSt9enable_ifIXntaaoosr3std7is_sameIT_NS3_4HalfEEE5valuesr3std7is_sameIS6_S4_EE5valuesr3std7is_sameIT0_fEE5valueES6_E4typeELPS4_0EEEvcclllNS_10OpMathTypeIS6_E4typeEPKS6_lSH_lSF_PS8_l");
void hetgpu_torch_cuda_blas_gemm_bfloat16() {}

void hetgpu_torch_cuda_blas_gemm_double()
    __asm__("_ZN2at4cuda4blas4gemmIddTnPNSt9enable_ifIXntaaoosr3std7is_sameIT_N3c104HalfEEE5valuesr3std7is_sameIS4_NS5_8BFloat16EEE5valuesr3std7is_sameIT0_fEE5valueES4_E4typeELPd0EEEvcclllNS_10OpMathTypeIS4_E4typeEPKS4_lSH_lSF_PS8_l");
void hetgpu_torch_cuda_blas_gemm_double() {}

void hetgpu_torch_cuda_blas_gemm_float()
    __asm__("_ZN2at4cuda4blas4gemmIffTnPNSt9enable_ifIXntaaoosr3std7is_sameIT_N3c104HalfEEE5valuesr3std7is_sameIS4_NS5_8BFloat16EEE5valuesr3std7is_sameIT0_fEE5valueES4_E4typeELPf0EEEvcclllNS_10OpMathTypeIS4_E4typeEPKS4_lSH_lSF_PS8_l");
void hetgpu_torch_cuda_blas_gemm_float() {}
