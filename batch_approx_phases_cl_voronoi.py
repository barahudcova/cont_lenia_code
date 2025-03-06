import numpy as np
import torch
import pickle
import os

import cProfile, pstats, io
from pstats import SortKey
import time

import torch
from Automaton import BatchLeniaMC
from utils.main_utils import compute_ker, read_params_from_demo, read_params_from_data, around_params, read_file, save_params
import cv2
import pickle as pk
from utils.main_utils import check_mass
import numpy as np, os, random
from torchenhanced.util import showTens
import matplotlib.pyplot as plt
from utils.hash_params_to_names import get_params_name




TIME_BOUND = 200 # in seconds

def classify(phases):
    if phases == {"order"}:
        return "order"
    elif phases == {"chaos"}: #or phases == {"chaos", "max"}:
        return "chaos"
    elif phases == {"order", "chaos", "max"}:
        return "max"
    elif phases == {"order", "max"}:
        return "max"
    elif phases == {"order", "chaos"}:
        return "trans"
    elif phases:
        return "no phase"
    else:
        return "TBA"


def get_batch_phases(auto, mass_centers_x, mass_centers_y, total_masses, std, window_size):
    phases = [None]*auto.batch
    for b in range(auto.batch):
        same_mass = torch.any(total_masses[-1,b]==total_masses[:-1,b])
        same_cent_x = torch.any(mass_centers_x[-1,b]==mass_centers_x[:-1,b])
        same_cent_y = torch.any(mass_centers_y[-1,b]==mass_centers_y[:-1,b])
        if same_mass and same_cent_x and same_cent_y:
            phases[b]="order"
            continue

        cmx_stable = torch.all(torch.max(torch.abs(torch.mean(mass_centers_x[-window_size:, b], axis=0)-mass_centers_x[-window_size:, b]))<std)
        cmy_stable = torch.all(torch.max(torch.abs(torch.mean(mass_centers_y[-window_size:, b], axis=0)-mass_centers_y[-window_size:, b]))<std)

        if cmx_stable and cmy_stable:
            phases[b]="chaos"
        else:
            phases[b]="max"

    return phases

# @timeout(TIME_BOUND)
def get_approx_trans(auto, std, window_size):
    config = auto.state
    array_size = config.shape[-1]
    T_MAX = int(np.round(100*np.log2(array_size)))

    trajectory = torch.empty(T_MAX, *config.shape).to(auto.device)
    mass_centers_x = torch.empty(T_MAX, auto.batch).to(auto.device)
    mass_centers_y = torch.empty(T_MAX, auto.batch).to(auto.device)
    total_masses = torch.empty(T_MAX, auto.batch).to(auto.device)

    for t in range(T_MAX):
        auto.step()
        config = auto.state
        trajectory[t] = config
        cm, tm = auto.get_batch_mass_center(config)
        mass_centers_x[t] = cm[0]
        mass_centers_y[t] = cm[1]
        total_masses[t] = tm

    

    phases = get_batch_phases(auto, mass_centers_x, mass_centers_y, total_masses, std, window_size)

    return phases


def get_approx_data(auto, polygon_size_range, array_size, samples, folder_name, params):
    batch_size = auto.batch

    g_mju = np.round(params["mu"].item(), 4)
    g_sig = np.round(params["sigma"].item(), 4)

    print("rounded: ", g_mju, g_sig)

    path = f'unif_random_voronoi/{folder_name}/data/{g_mju}_{g_sig}.pickle'

    std = 3
    window_size = 200

    PH_TUP = []
    PH = []

    
    for polygon_size in polygon_size_range:
        start = time.time()
        print("polygon size: ", polygon_size)

        try:
            with open(path, 'rb') as handle:
                data = pickle.load(handle)
        except:
            data = {"params": params}

        try:
            _ = data[str(array_size)]
        except:
            data[str(array_size)] = {}

        try:
            ph = [t for t in data[str(array_size)][str(polygon_size)]["phase"]]
            num_samples = len(ph)
            print(polygon_size, num_samples, "already computed")
        except:
            num_samples = 0
            data[str(array_size)][str(polygon_size)] = {}
            data[str(array_size)][str(polygon_size)]["phase"] = []
            data[str(array_size)][str(polygon_size)]["seed"] = []
            data[str(array_size)][str(polygon_size)]["sample"] = []

        if num_samples >= samples:
            PH_TUP.append((polygon_size, classify(set(data[str(array_size)][str(polygon_size)]["phase"]))))
            PH.append(classify(set(data[str(array_size)][str(polygon_size)]["phase"]))) 
            print(f"{num_samples} computed, skipping polygon_size {polygon_size}")
            continue

        if samples-num_samples < batch_size:
            print("yes")
            samples = num_samples+batch_size

        for batch_index in range(num_samples, samples, batch_size):
            print("batch index: ", batch_index)

            auto.set_init_voronoi_batch(polygon_size, batch_index)
            seeds = auto.seeds

            phases = get_approx_trans(auto, std, window_size)

            data[str(array_size)][str(polygon_size)]["phase"]+=phases
            data[str(array_size)][str(polygon_size)]["seed"]+=seeds
            indices = list(range(batch_index, batch_index+batch_size))
            data[str(array_size)][str(polygon_size)]["sample"]+=indices

            print(len(indices), len(phases))

            with open(path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


            print(polygon_size, classify(set(data[str(array_size)][str(polygon_size)]["phase"])))
        
        PH_TUP.append((polygon_size, classify(set(data[str(array_size)][str(polygon_size)]["phase"]))))
        PH.append(classify(set(data[str(array_size)][str(polygon_size)]["phase"]))) 

    
        """
        D = pickle.load(file)

        data = D["100"][str(polygon_size)]
        ogphases =  data["phase"]

        ogseeds = data["seed"]
        
        
        for i in range(batch_size):
            if phases[i] != ogphases[i]:
                print(i, phases[i], ogphases[i])
                auto1 = BatchLeniaMC((1,H,W), dt, params=params, num_channels=num_channels, device=device)
                auto1.to(device)
                auto1.set_init_voronoi(polygon_size, sample=i, seed=seeds[i])
                auto1.make_video(time_steps = 200, step=1, seed = seeds[i], config = auto1.state, video_path=f"video_{g_mju}_{g_sig}_{i}_{seeds[i]}.gif")
        """

    
                

        print("time: ", time.time() - start)
        print()

    return PH_TUP, PH



            

#============================== PARAMETERS ================================
device = 'cuda:2' # Device on which to run the automaton
W,H = 100,100 # Size of the automaton
array_size = W
dt = 0.1 # Time step size
num_channels= 1

samples = 64
beta = [1, 1, 1]
k_mju = [0.2, 0.5, 0.8]
k_sig = [0.08, 0.08, 0.08]

beta = [1]
k_mju = [0.5]
k_sig = [0.15]

kernel_folder = "_".join([str(s) for s in k_mju])+"_"+"_".join([str(s) for s in k_sig])

print(kernel_folder)

if not os.path.exists(f"unif_random_voronoi/{kernel_folder}"):
    os.mkdir(f"unif_random_voronoi/{kernel_folder}")
    os.mkdir(f"unif_random_voronoi/{kernel_folder}/data")


polygon_size_range = [10, 20, 30, 40, 50, 60, 70, 80]
#======================================================================


params = {'k_size': 27, 
          'mu': torch.tensor([[[0.1]]], device=device), 
          'sigma': torch.tensor([[[0.0]]], device=device), 
          'beta': torch.tensor([[[beta]]], device=device), 
          'mu_k': torch.tensor([[[k_mju]]], device=device), 
          'sigma_k': torch.tensor([[[k_sig]]], device=device), 
          'weights': torch.tensor([[[1]]], device=device)}


B = samples


#======================================================================
# Initializing automaton with batch size = 1 to get the exact same kernel every time for reproducibility

auto1 = BatchLeniaMC((1,H,W), dt, params=params, num_channels=num_channels, device=device)
auto1.to(device)
kernel1 = auto1.kernel[0]
auto1.plot_kernel()

auto = BatchLeniaMC((B,H,W), dt, params=params, num_channels=num_channels, device=device)
auto.to(device)
for i in range(B):
    auto.kernel[i]=kernel1
auto.kernel_eff = auto.kernel.reshape([auto.batch*auto.C*auto.C,1,auto.k_size,auto.k_size]) 

""" ph_tup, ph = get_approx_data(auto, polygon_size_range, array_size, samples, kernel_folder, params)
print(ph_tup, ph) """



#======================================================================

auto1 = BatchLeniaMC((1,H,W), dt, params=params, num_channels=num_channels, device=device)
auto1.to(device)
kernel1 = auto1.kernel[0]

for g_mju in np.arange(0.6, 0.7, 0.005):
    g_mju = np.round(g_mju, 4)
    for g_sig in np.arange(0.01,0.12, 0.001):
        g_sig = np.round(g_sig, 4)
        
        torch.cuda.empty_cache()

        params["mu"][0][0][0] = g_mju
        params["sigma"][0][0][0] = g_sig

        auto = BatchLeniaMC((B,H,W), dt, params=params, num_channels=num_channels, device=device)
        auto.to(device)
        for i in range(B):
            auto.kernel[i]=kernel1
        auto.kernel_eff = auto.kernel.reshape([auto.batch*auto.C*auto.C,1,auto.k_size,auto.k_size]) 

        ph_tup, ph = get_approx_data(auto, polygon_size_range, array_size, samples, kernel_folder, params) 

        # in case finer polygon sizes are wanted:
"""         if not "max" in ph:
            if "trans" in ph:
                r_min = min([k for (k, v) in ph_tup if v=="trans"])
                r_max = r_min+10
                new_polygon_size_range = np.arange(r_min, r_max, 1)
                print(new_polygon_size_range)
                ph_tup, ph = get_approx_data(auto, new_polygon_size_range, array_size, samples, kernel_folder, params)
                print(ph)
            elif ("order" in ph) and ("chaos" in ph):
                r_min = max([k for (k, v) in ph_tup if v == "order"])
                r_max = r_min + 10
                new_polygon_size_range = np.arange(r_min, r_max, 1)
                print(new_polygon_size_range)
                ph_tup, ph = get_approx_data(auto, new_polygon_size_range, array_size, samples, kernel_folder, params)
                print(ph)   """

 