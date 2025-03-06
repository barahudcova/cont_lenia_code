import numpy as np
import torch
import pickle

import cProfile, pstats, io
from pstats import SortKey

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


def check_monotonicity(mass_history):
    diff = mass_history[1:]-mass_history[:-1]
    return (torch.all(diff >= 0)) or (torch.all(diff <= 0))

def check_stable(cmh, std, window_size, current_time):
    if current_time<window_size:
        return False
    
    masses = cmh[current_time-window_size:current_time, :]
    diff = torch.max(torch.abs(torch.mean(masses, axis=0)-masses))
    """if not current_time%10:
        print(diff) """
    return torch.all(diff<std)


def hash_config(array, torch=True):
    if torch:
        pickled = pickle.dumps(array.cpu().numpy())
    else:
        pickled = pickle.dumps(array)
    return pickled

# @timeout(TIME_BOUND)
def get_approx_trans(auto, config, std, window_size):
    step = 50
    config = auto.state
    array_size = config.shape[-1]
    area = array_size**2
    T_MAX = int(np.round(1000*np.log2(array_size)))

    mc, tm = auto.get_mass_center(config)
    P = np.prod(mc.shape)

    history = {(torch.sum(mc).item(),torch.sum(tm).item()): 0}

    cmass_history = torch.zeros((T_MAX+2, P))
    short_mass_history = torch.zeros(T_MAX+2)
    long_mass_history = torch.zeros(T_MAX+2)

    cmass_history[0, :] = mc.view(P)
    short_mass_history[0] = torch.sum(tm)
    long_mass_history[0] = torch.sum(tm)

    config = auto.state
    
    t = 0
    while t <= T_MAX:
        t += 1
        auto.step()
        config = auto.state
        mc, tm = auto.get_mass_center(config)
        bytes_config = (torch.sum(mc).item(),torch.sum(tm).item())
        try:
            prev_time = history[bytes_config]
            return t - prev_time, prev_time, config, "order"
        except:
            center_stable = check_stable(cmass_history, std, window_size, t)
            monotone_short = check_monotonicity(short_mass_history[t-10:t])
            monotone_long = check_monotonicity(long_mass_history[:t])

            
            if center_stable and (not monotone_short) and (not monotone_long) and (short_mass_history[t-1] > area//10):
                return 0, t - window_size, config, "chaos"
            else:
                history[bytes_config] = t
                cmass_history[t, :] = mc.view(P)
                short_mass_history[t] = torch.sum(tm)
                if not t%step:
                    long_mass_history[t//step] = torch.sum(tm)
                    

    return 0, T_MAX, config, "max"



def get_approx_data(auto, polygon_size_range, array_size, samples, g_mju, g_sig, name, params, device='cuda'):
    g_mju = np.round(g_mju, 4)
    g_sig = np.round(g_sig, 4)

    print("rounded: ", g_mju, g_sig)

    path = f'unif_random_voronoi/{name}/data/{g_mju}_{g_sig}.pickle'

    std = 3
    window_size = 200

    phases_tup = []
    phases = []


    for polygon_size in polygon_size_range:
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
            trans = [t for t in data[str(array_size)][str(polygon_size)]["trans"]]
            num_samples = len(trans)
            print(polygon_size, num_samples, "already computed")
        except:
            num_samples = 0
            trans=[]
            data[str(array_size)][str(polygon_size)] = {}
            data[str(array_size)][str(polygon_size)]["trans"] = []
            data[str(array_size)][str(polygon_size)]["att"]  = []
            data[str(array_size)][str(polygon_size)]["phase"] = []
            data[str(array_size)][str(polygon_size)]["seed"] = []


        for sample in range(num_samples, samples):
            num_samples+=1

            auto.set_init_voronoi(polygon_size, sample%1024)
            config, seed = auto.state, auto.seed
            
    
            attractor, transient, config, phase = get_approx_trans(auto, config, std, window_size)


            trans.append(transient)
            data[str(array_size)][str(polygon_size)]["trans"].append(transient)
            data[str(array_size)][str(polygon_size)]["att"].append(attractor)
            data[str(array_size)][str(polygon_size)]["phase"].append(phase)
            data[str(array_size)][str(polygon_size)]["seed"].append(seed)


            if not num_samples % 4:

                with open(path, 'wb') as handle:
                    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        phases_tup.append((polygon_size, classify(set(data[str(array_size)][str(polygon_size)]["phase"]))))
        phases.append(classify(set(data[str(array_size)][str(polygon_size)]["phase"])))

    return phases_tup, phases



            

#============================== PARAMETERS ================================
device = 'cuda:3' # Device on which to run the automaton
W,H = 100,100 # Size of the automaton
array_size = W
dt = 0.1 # Time step size
num_channels= 1

samples = 64
ogname = "0.5_0.15"

polygon_size_range = [10, 20, 30, 40, 50, 60, 70, 80]
#======================================================================

params = {'k_size': 27, 
          'mu': torch.tensor([[[0.1]]], device=device), 
          'sigma': torch.tensor([[[0.04]]], device=device), 
          'beta': torch.tensor([[[[1]]]], device=device), 
          'mu_k': torch.tensor([[[[0.5]]]], device=device), 
          'sigma_k': torch.tensor([[[[0.15]]]], device=device), 
          'weights': torch.tensor([[[1]]], device=device)}



for g_mju in np.arange(0.3, 0.4, 0.005):
    g_mju = np.round(g_mju, 4)
    for g_sig in np.arange(0.00,0.02, 0.001):
        g_sig = np.round(g_sig, 4)

        params["mu"][0][0][0] = g_mju
        params["sigma"][0][0][0] = g_sig

        auto = BatchLeniaMC((1,H,W), dt, params=params, num_channels=num_channels, device=device)
        auto.to(device)

        get_approx_data(auto, polygon_size_range, array_size, samples, g_mju, g_sig, ogname, params, device=device)

        phases_tup, phases = get_approx_data(auto, polygon_size_range, array_size, samples, g_mju, g_sig, ogname, params, device=device)
        if not "max" in phases:
            if "trans" in phases:
                r_min = min([k for (k, v) in phases_tup if v=="trans"])
                r_max = r_min+10
                new_polygon_size_range = np.arange(r_min, r_max, 1)
                print(new_polygon_size_range)
                phases_tup, phases = get_approx_data(auto, new_polygon_size_range, array_size, samples, g_mju, g_sig, ogname, params, device=device)
                print(phases)
            elif ("order" in phases) and ("chaos" in phases):
                r_min = max([k for (k, v) in phases_tup if v == "order"])
                r_max = r_min + 10
                new_polygon_size_range = np.arange(r_min, r_max, 1)
                print(new_polygon_size_range)
                phases_tup, phases = get_approx_data(auto, new_polygon_size_range, array_size, samples, g_mju, g_sig, ogname, params, device=device)
                print(phases)

 


#======================================================================


