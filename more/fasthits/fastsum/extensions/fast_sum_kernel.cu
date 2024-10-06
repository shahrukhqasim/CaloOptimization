#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
// #include "fast_sum.h"


// Binary search device function
__device__ int32_t binary_search(const int32_t* row_splits, int32_t n_sims, int32_t idx) {
    int32_t left = 0;
    int32_t right = n_sims;

    while (left <= right) {
        int32_t mid = left + (right - left) / 2;
        if (row_splits[mid] <= idx && idx < row_splits[mid + 1]) {
            return mid;
        } else if (row_splits[mid] > idx) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // If not found (should not happen with proper input), return -1
    return -1;
}


// CUDA kernel
__global__ void fast_sum_kernel(
    const int32_t* row_splits,
    const float* z_values,
    const float* deps,
    const float* sensor_start,
    const float* sensor_end,
    float* output,
    int32_t n_sims,
    int32_t n_sensors,
    int32_t n_hits_total
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_hits_total) {
        // Determine the simulation index (sim_idx) using row_splits
        int32_t sim_idx = binary_search(row_splits, n_sims, idx);

        // Get the z_value for this hit
        float z_val = z_values[idx];

        // Determine which sensor this z_value falls into
        int32_t sensor_idx = -1;
        for (int32_t i = 0; i < n_sensors; ++i) {
            if (sensor_start[i] <= z_val && z_val < sensor_end[i]) {
                sensor_idx = i;
                break;
            }
        }

        // If a valid sensor is found, sum up the deps value into the output
        if (sensor_idx != -1) {
            atomicAdd(&output[sim_idx * n_sensors + sensor_idx], deps[idx]);
        }
    }
}


torch::Tensor fast_sum_cuda(
    torch::Tensor row_splits,
    torch::Tensor z_values,
    torch::Tensor deps,
    torch::Tensor sensor_start,
    torch::Tensor sensor_end
) {
    const auto n_sims = row_splits.size(0) -1;
    const auto n_sensors = sensor_start.size(0);
    const auto n_hits_total = z_values.size(0);

    auto output = torch::zeros({ n_sims, n_sensors }, torch::TensorOptions().dtype(torch::kFloat32).device(row_splits.device()));


    const int32_t threads_per_block = 1024;
    const int32_t num_blocks = (n_hits_total + threads_per_block - 1) / threads_per_block;

    // Launch CUDA kernel
    fast_sum_kernel<<<num_blocks, threads_per_block>>>(
        row_splits.data_ptr<int32_t>(),
        z_values.data_ptr<float>(),
        deps.data_ptr<float>(),
        sensor_start.data_ptr<float>(),
        sensor_end.data_ptr<float>(),
        output.data_ptr<float>(),
        n_sims,
        n_sensors,
        n_hits_total
    );


    return output;
}
