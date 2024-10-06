import time
import os.path as osp

import numpy as np
import torch
import unittest

import fastsum.extensions
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(fastsum.extensions.__file__)), 'fast_sum.so'))

def generate_uniform_values(x, y, size):
    return x + (y - x) * torch.rand(size)


class TestFastSumCuda(unittest.TestCase):
    def test_fast_sum_cuda(self):
        # Input data
        row_splits = [0, 10]
        z_values = [1, 1, 4, 4, 6, 6, 7, 8, 9, 9]
        deps = [0.1, 0.11, 0.13, 0.4, 0.7, 0.8, 0.8, 0.1, 0.1, 0.1]
        sensor_start = [3.5, 5.5]
        sensor_end = [5.7, 6.5]

        # Expected output

        # Convert to torch tensors and move to CUDA
        row_splits_tensor = torch.tensor(row_splits, device='cuda', dtype=torch.int32)
        z_values_tensor = torch.tensor(z_values, device='cuda', dtype=torch.float)
        deps_tensor = torch.tensor(deps, device='cuda', dtype=torch.float)
        sensor_start_tensor = torch.tensor(sensor_start, device='cuda', dtype=torch.float)
        sensor_end_tensor = torch.tensor(sensor_end, device='cuda', dtype=torch.float)


        expected_output = torch.tensor([[0.5300, 1.5000]]).to(row_splits_tensor.device)

        # Perform the operation
        output = torch.ops.fast_sum_cuda.fast_sum_cuda(
            row_splits_tensor,
            z_values_tensor,
            deps_tensor,
            sensor_start_tensor,
            sensor_end_tensor
        )

        # Assert that the output is as expected
        self.assertTrue(torch.allclose(output, expected_output), "The output does not match the expected result.")

    def test_fast_sum_timing_cuda(self):
        for i in range(100):
            num_sims = 5000
            sensor_start = torch.tensor([1, 2, 3, 5, 6, 7], dtype=torch.float).cuda()
            sensor_end = torch.tensor([1.1, 2.1, 3.1, 5.1, 6.1, 7.1], dtype=torch.float).cuda()

            row_splits = [0]

            z_values = []
            deps = []

            for i in range(num_sims):
                size = np.random.randint(30000, 300000)

                z_values_ = torch.randuniform_values = generate_uniform_values(0, 10, size=(size,)).cuda()
                deps_ = torch.normal(0.1, 0.4, size=(size,)).cuda()

                z_values.append(z_values_)
                deps.append(deps_)

                row_splits.append(size + row_splits[-1])

            z_values = torch.cat(z_values, dim=0)
            deps = torch.cat(deps, dim=0)
            row_splits = torch.tensor(row_splits, device='cuda', dtype=torch.int32)

            t1 = time.time()
            output = torch.ops.fast_sum_cuda.fast_sum_cuda(
                row_splits,
                z_values,
                deps,
                sensor_start,
                sensor_end
            )
            print(output[0,0])
            print("Took", time.time() - t1,"seconds")

    def test_fast_sum_bigger_cuda(self):
        num_sims = 10

        sensor_start = torch.tensor([2,5,8], dtype=torch.float).cuda()
        sensor_end = torch.tensor([3,6,9], dtype=torch.float).cuda()


        row_splits = [0]

        z_values = []
        deps = []

        output_baseline = torch.zeros((num_sims, len(sensor_start)), dtype=torch.float).cuda()

        for i in range(num_sims):
            size = np.random.randint(400,900)

            z_values_ = torch.randuniform_values = generate_uniform_values(0, 10, size=(size,)).cuda()
            deps_ = torch.normal(0.1, 0.4, size=(size,)).cuda()

            z_values.append(z_values_)
            deps.append(deps_)

            row_splits.append(size+row_splits[-1])

            for k in range(len(z_values_)):
                for j in range(len(sensor_start)):
                    if z_values_[k] > sensor_start[j] and z_values_[k] < sensor_end[j]:
                        output_baseline[i][j] += deps_[k]

        z_values = torch.cat(z_values, dim=0)
        deps = torch.cat(deps, dim=0)
        row_splits = torch.tensor(row_splits, device='cuda', dtype=torch.int32)

        output = torch.ops.fast_sum_cuda.fast_sum_cuda(
            row_splits,
            z_values,
            deps,
            sensor_start,
            sensor_end
        )

        self.assertTrue(torch.allclose(output, output_baseline), "The output does not match the expected result.")



