#include <torch/extension.h>
//#include "fast_sum.h"

// CUDA forward declarations
torch::Tensor fast_sum_cuda(
    torch::Tensor row_splits,
    torch::Tensor z_values,
    torch::Tensor deps,
    torch::Tensor sensor_start,
    torch::Tensor sensor_end
);

// C++ interface
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor fast_sum_cuda_interface(
    torch::Tensor row_splits,
    torch::Tensor z_values,
    torch::Tensor deps,
    torch::Tensor sensor_start,
    torch::Tensor sensor_end
) {
    CHECK_INPUT(row_splits);
    CHECK_INPUT(z_values);
    CHECK_INPUT(deps);
    CHECK_INPUT(sensor_start);
    CHECK_INPUT(sensor_end);

    return fast_sum_cuda(row_splits,z_values,deps,sensor_start,sensor_end);
}

TORCH_LIBRARY(fast_sum_cuda, m) {
    m.def("fast_sum_cuda", fast_sum_cuda_interface);
}