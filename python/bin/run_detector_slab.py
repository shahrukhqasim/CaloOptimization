import gzip
import os.path
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import argh
from heterogeneous_sampling_calorimeter import get_ref_alpha_design, get_ref_beta_design, get_ref_gamma_design, get_ref_slab_design
from sampling_calo import add, initialize, simulate_particle, collect_deposits, collect_full_deposits
import json
from tqdm import tqdm
import numba
from multiprocessing import Pool
from neural_calibration import train_neural_network


def run_sims(design, ll, pid, pz_fn):

    num_sims = ll
    true = []
    deposit_full_xs = []
    deposit_full_ys = []
    deposit_full_zs = []
    deposit_full_charges = []

    for i in tqdm(range(num_sims)):
        pz = np.random.uniform(pz_fn[0], pz_fn[1])
        simulate_particle(0, 0, pz, pid, 0, 0, -1)

        full_deposits = collect_full_deposits()

        deposit_full_xs.append(full_deposits['deposit_full_x'])
        deposit_full_ys.append(full_deposits['deposit_full_y'])
        deposit_full_zs.append(full_deposits['deposit_full_z'])
        deposit_full_charges.append(full_deposits['deposit_full_charge'])

        print(full_deposits['deposit_full_x'].shape, full_deposits['deposit_full_y'].shape,
              full_deposits['deposit_full_z'].shape, full_deposits['deposit_full_charge'].shape)


        true.append({
            'pz':pz,
            'pid':pid
        })


    return true, deposit_full_xs, deposit_full_ys, deposit_full_zs, deposit_full_charges


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
    design = get_ref_slab_design()
    design['limits'] = {
        'max_step_length': 0.005,
        'minimum_kinetic_energy': 0.00001,
    }
    design['store_full'] = True


    all_results = []

    seed = int(time.time()) + os.getpid()
    np.random.seed(seed)

    output_data = initialize(np.random.randint(256), np.random.randint(256), np.random.randint(256),
                             np.random.randint(256), json.dumps(design))

    res = run_sims(design, ll=500, pid=211, pz_fn=(9, 200.))
    all_results.append(res)
    with open('calo_data_1.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    res = run_sims(design, ll=500, pid=22, pz_fn=(9, 200.))
    all_results.append(res)
    with open('calo_data_2.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    res = run_sims(design, ll=500, pid=22, pz_fn=(49.999, 50.))
    all_results.append(res)
    with open('calo_data_3.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    res = run_sims(design, ll=500, pid=22, pz_fn=(99.999, 100.))
    all_results.append(res)
    with open('calo_data_4.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    res = run_sims(design, ll=500, pid=211, pz_fn=(49.999, 50.))
    all_results.append(res)
    with open('calo_data_5.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    res = run_sims(design, ll=500, pid=211, pz_fn=(99.999, 100.))
    all_results.append(res)
    with open('calo_data_6.pkl', 'wb') as f:
        pickle.dump(all_results, f)



if __name__ == '__main__':
    main()