import pickle
import time
from heterogeneous_sampling_calorimeter import get_ref_alpha_design, get_ref_beta_design, get_ref_gamma_design, get_ref_slab_design
import numpy as np
from heterogeneous_sampling_calorimeter import um
import torch
import os.path as osp
import matplotlib.pyplot as plt



import fastsum.extensions
torch.ops.load_library(osp.join(osp.dirname(osp.realpath(fastsum.extensions.__file__)), 'fast_sum.so'))



class ResamplingSimulation:
    def __init__(self):
        pass

    def set_sensors(self, sensors_start, sensors_end):
        self.sensors_start_cpu = sensors_start
        self.sensors_end_cpu = sensors_end

        self.sensors_start_gpu = torch.tensor(self.sensors_start_cpu, dtype=torch.float32).cuda()
        self.sensors_end_gpu = torch.tensor(self.sensors_end_cpu, dtype=torch.float32).cuda()

        layer_weighting = []
        prev = 0
        for i in range(len(sensors_start)):
            tmp = (sensors_start[i] - prev) / (sensors_end[i] - sensors_start[i])
            prev = sensors_end[i]
            layer_weighting.append(tmp)
        self.layer_weighting_cpu = np.array(layer_weighting)
        self.layer_weighting_gpu = torch.tensor(self.layer_weighting_cpu).cuda()

    def simulate(self, z_values, deposits, true):
        row_splits = [0]
        for i in range(len(z_values)):
            row_splits.append(row_splits[-1] + len(z_values[i]))

        row_splits_tensor = torch.tensor(row_splits, dtype=torch.int32).cuda()
        all_zs_tensor = [torch.tensor(arr, dtype=torch.float32) for arr in z_values]
        all_deps_tensor = [torch.tensor(arr, dtype=torch.float32) for arr in deposits]

        z_values = torch.cat(all_zs_tensor, dim=0).cuda()
        deps = torch.cat(all_deps_tensor, dim=0).cuda()

        output = torch.ops.fast_sum_cuda.fast_sum_cuda(
            row_splits_tensor,
            z_values,
            deps,
            self.sensors_start_gpu,
            self.sensors_end_gpu,
        )

        output = torch.sum(output * self.layer_weighting_gpu, dim=1)

        return output.cpu().numpy()

    def calibrate_layer_weighting(self, factor):
        self.layer_weighting_cpu = self.layer_weighting_cpu * factor
        self.layer_weighting_gpu = self.layer_weighting_gpu * factor



def main():
    design = get_ref_alpha_design()
    sensors_start = np.array([x['z_center']  for x in design['layers'] if x['active']]) * 1000
    sensors_end = sensors_start + 0.12

    simulation = ResamplingSimulation()
    simulation.set_sensors(sensors_start, sensors_end)


    t1 = time.time()
    print("Loading...")
    with open('calo_data_6.pkl', 'rb') as f:
        all_results = pickle.load(f)
    print("Took", time.time() - t1,"seconds to load.")

    em_zs = all_results[1][-2]
    em_deps = all_results[1][-1]
    em_true = [x['pz'] for x in all_results[1][0]]


    had_zs = all_results[0][-2]
    had_deps = all_results[0][-1]
    had_true = [x['pz'] for x in all_results[0][0]]

    all_zs = em_zs + had_zs
    all_deps = em_deps + had_deps
    all_true = np.array(em_true + had_true)
    output = simulation.simulate(all_zs, all_deps, all_true)

    a, _, _, _ = np.linalg.lstsq(output[:, np.newaxis], all_true[:, np.newaxis])
    a = a[0][0]
    simulation.calibrate_layer_weighting(a)
    output = simulation.simulate(all_zs, all_deps, all_true)

    plt.scatter(all_true, output, s=0.1)
    plt.savefig('plotsx/scatter.pdf')
    plt.close()

    x = 2
    output = simulation.simulate(all_results[x][-2], all_results[x][-1], [x['pz'] for x in all_results[x][0]])
    plt.hist(output, histtype='step')
    plt.savefig('plotsx/hist2.pdf')
    plt.close()

    x = 3
    output = simulation.simulate(all_results[x][-2], all_results[x][-1], [x['pz'] for x in all_results[x][0]])
    plt.hist(output, histtype='step')
    plt.savefig('plotsx/hist3.pdf')
    plt.close()

    x = 4
    output = simulation.simulate(all_results[x][-2], all_results[x][-1], [x['pz'] for x in all_results[x][0]])
    plt.hist(output, histtype='step')
    plt.savefig('plotsx/hist4.pdf')
    plt.close()

    x = 5
    output = simulation.simulate(all_results[x][-2], all_results[x][-1], [x['pz'] for x in all_results[x][0]])
    plt.hist(output, histtype='step')
    plt.savefig('plotsx/hist5.pdf')
    plt.close()




if __name__ == '__main__':
    main()

