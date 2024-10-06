import gzip
import os.path
import pickle
import time

import numpy as np
import matplotlib.pyplot as plt
import argh
from heterogeneous_sampling_calorimeter import get_ref_alpha_design
from sampling_calo import add, initialize, simulate_particle, collect_deposits
import json
from tqdm import tqdm
import numba
from multiprocessing import Pool
from neural_calibration import train_neural_network


@numba.jit(nopython=True)
def find_layer_deps(hit_layer, hit_deposit, uncalibrated_weights):
    layer_deps = np.zeros(len(uncalibrated_weights))
    if len(hit_layer) > 0:
        for l, d in zip(hit_layer, hit_deposit):
            layer_deps[l] += d * uncalibrated_weights[l]
    return layer_deps


def run_sims(ll, pid, layer_weighting):
    global design
    
    seed = int(time.time()) + os.getpid()
    np.random.seed(seed)
    output_data = initialize(np.random.randint(256), np.random.randint(256), np.random.randint(256),
                             np.random.randint(256), json.dumps(design))


    all_layer_dep_data = []
    true = []
    for i in tqdm(range(ll)):
        pz = np.random.uniform(9, 250)
        simulate_particle(0, 0, pz, pid, 0, 0, -1)
        deposit_data = collect_deposits()
        if deposit_data['layer'].dtype != np.int32 or deposit_data['layer'].dtype != np.int64:
            deposit_data['layer'] = deposit_data['layer'].astype(np.int64)

        layer_deps = find_layer_deps(deposit_data['layer'], deposit_data['charge_deposit'], layer_weighting)
        all_layer_dep_data.append(layer_deps)
        true.append(pz)
        # print(np.sum(layer_deps))

    return true, all_layer_dep_data


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
    return results




design = get_ref_alpha_design()
design['limits'] = {
            'max_step_length': 0.005,
            'minimum_kinetic_energy' : 0.01,

    }


def main(rerun_sims=False, num_sims=10, cores=2):
    global design
    print("Does it work?", add(1,1))

    layer_weighting = []
    i = 0
    for l in design['layers']:
        if l['active'] == True:
            # r = l['pre_absorber_rad_lengths'] #/ l['sensor_rad_lengths']
            r = 1 + l['pre_absorber_width'] / l['dz']
            # r = l['pre_absorber_rad_lengths'] / l['sensor_rad_lengths']
            # print(r)
            layer_weighting += [r]
            assert l['layer_number'] == i
            i += 1
    layer_weighting = np.array(layer_weighting, np.float64)


    if rerun_sims or not os.path.exists('temp/sims_erst.pkl'):
        all_layer_dep_data = []
        true = []
        t1 = time.time()

        # true, all_layer_dep_data = run_sims(100, lambda: np.random.uniform(9, 250), 22, layer_weighting)
        all_ret = parallel_function_call(run_sims, [(int(num_sims/cores), 22, layer_weighting)]*cores, cores)
        true = []
        all_layer_dep_data = []
        for ar in all_ret:
            true += ar[0]
            all_layer_dep_data += ar[1]

        # all_layer_dep_data = np.sum(np.array(all_layer_dep_data), axis=1)
        true = np.array(true)
        all_layer_dep_data = np.array(all_layer_dep_data)

        em_dep, em_true = all_layer_dep_data, true


        # print(all_layer_dep_data)
        # print(true)
        #
        #

        all_ret = parallel_function_call(run_sims, [(int(num_sims/cores), 211, layer_weighting)]*cores, cores)
        true = []
        all_layer_dep_data = []
        for ar in all_ret:
            true += ar[0]
            all_layer_dep_data += ar[1]

        true = np.array(true)
        all_layer_dep_data = np.array(all_layer_dep_data)


        had_dep, had_true = all_layer_dep_data, true

        outdict = {
            'em_true': em_true,
            'em_dep': em_dep,
            'had_true': had_true,
            'had_dep': had_dep,
        }
        with gzip.open('temp/sims_erst.pkl', 'wb') as f:
            pickle.dump(outdict, f)
    else:
        with gzip.open('temp/sims_erst.pkl', 'rb') as f:
            outdict = pickle.load(f)

    a, _, _, _ = np.linalg.lstsq(np.sum(outdict['em_dep'], axis=1, keepdims=True), outdict['em_true'][:, np.newaxis])


    epochs = 10000
    outdict['em_dep'] = train_neural_network(outdict['em_dep']*a, outdict['em_true'], hidden_dim=-1, num_epochs=epochs, learning_rate=0.0001)
    outdict['had_dep'] = train_neural_network(outdict['had_dep']*a, outdict['had_true'], hidden_dim=-1, num_epochs=epochs, learning_rate=0.0001)
    # outdict['em_dep'] = np.sum(outdict['em_dep'] * a, axis=1)
    # outdict['had_dep'] = np.sum(outdict['had_dep'] * a, axis=1)


    # a, _, _, _ = np.linalg.lstsq(outdict['em_dep'][:, np.newaxis], outdict['em_true'][:, np.newaxis])
    # outdict['em_dep'] = a * outdict['em_dep']

    # outdict['had_dep'] = a * outdict['had_dep']

    plt.scatter(outdict['em_true'], outdict['em_dep'], label='EM', s=0.3, alpha=0.5)
    plt.scatter(outdict['had_true'], outdict['had_dep'], label='Had',  s=0.3, alpha=0.5)
    plt.plot([0,250], [0, 250])
    plt.legend()
    # plt.show()
    plt.savefig('plots/response_scatter.png')


if __name__ == '__main__':
    argh.dispatch_command(main)