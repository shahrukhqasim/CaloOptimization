import gzip
import os.path
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import argh
from heterogeneous_sampling_calorimeter import get_ref_alpha_design, get_ref_beta_design, get_ref_gamma_design
from sampling_calo import add, initialize, simulate_particle, collect_deposits
import json
from tqdm import tqdm
import numba
from multiprocessing import Pool
from neural_calibration import train_neural_network


def run_sims(design, ll, pid, pz_fn):
    seed = int(time.time()) + os.getpid()
    np.random.seed(seed)

    output_data = initialize(np.random.randint(256), np.random.randint(256), np.random.randint(256),
                             np.random.randint(256), json.dumps(design))

    t1 = time.time()
    num_sims = ll
    true = []
    layer_deps = []
    for i in tqdm(range(num_sims)):
        pz = np.random.uniform(pz_fn[0], pz_fn[1])
        simulate_particle(0, 0, pz, pid, 0, 0, -1)
        charge_deposit = collect_deposits()['charge_deposit']
        true += [pz]
        layer_deps += [charge_deposit]

    return true, layer_deps


def parallel_function_call(func, params, num_cores):
    """
    Call a function in parallel using multiprocessing.

    :param func: The function to be called.
    :param params: A list of tuples, where each tuple contains the parameters for a single function call.
    :param num_cores: The number of cores to use for parallel processing.
    :return: A list with the results of each function call.
    """
    with Pool(processes=num_cores) as pool:
        results = pool.starmap(func, params)

    true = []
    layer_deps = []

    for r in results:
        true += r[0]
        layer_deps += r[1]

    return true, layer_deps


def main():
    design = get_ref_gamma_design()
    design['limits'] = {
        'max_step_length': 0.005,
        'minimum_kinetic_energy': 0.00001,

    }

    initial_weighting = []
    filt = []
    for l in design['layers']:
        filt += [l['active']]
        if l['active'] == True:
            initial_weighting += [l['pre_absorber_width'] / l['dz']]
    filt = np.array(filt)
    initial_weighting = np.array(initial_weighting)

    num_sims = 30
    cores = 10
    pz_fn = (9, 250)
    true, layer_deps = parallel_function_call(run_sims, [(design, int(num_sims / cores), 22, pz_fn)] * cores, cores)
    true2, layer_deps2 = parallel_function_call(run_sims, [(design, int(num_sims / cores), 211, pz_fn)] * cores, cores)
    true = true + true2
    layer_deps = layer_deps + layer_deps2

    true = np.array(true)
    dep = np.array([x[filt]*initial_weighting for x in layer_deps])
    print(true)
    print(dep.shape)

    a, _, _, _ = np.linalg.lstsq(np.sum(dep, axis=1, keepdims=True), true[:, np.newaxis])
    a = float(a[0][0])

    num_sims = 1000
    pz_fn = (49.9999, 50.00)
    true, layer_deps = parallel_function_call(run_sims, [(design, int(num_sims / cores), 211, pz_fn)] * cores, cores)
    dep = np.sum(np.array([x[filt] * initial_weighting * a for x in layer_deps]), axis=1)

    print(dep)

    print(np.std(dep))

    plt.hist(dep, bins=30)
    plt.show()


    0/0



if __name__ == '__main__':
    main()