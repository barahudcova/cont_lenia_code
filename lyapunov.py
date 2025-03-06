import torch
from Automaton import BatchLeniaMC

import torch,torch.nn,torch.nn.functional as F
import numpy as np
from torchenhanced import DevModule
from utils.noise_gen import perlin,perlin_fractal
from utils.main_utils import gen_batch_params,gen_params
import matplotlib.pyplot as plt
import matplotlib.animation
import pickle
from utils.voronoi_utils import load_pattern
import os
from utils.hash_params_to_names import get_params_name
import time

import sys
np.set_printoptions(threshold=sys.maxsize)

 

def generate_laypunov(auto, g_mju, g_sig, polygon_size, pretime, multi_step_size, max_it, step):
    source_path = f"unif_random_voronoi/0.5_0.15/data/{g_mju}_{g_sig}.pickle"
    with open(source_path, 'rb') as handle:
        data = pickle.load(handle)


    seeds = data["100"][str(polygon_size)]["seed"]
    phases = data["100"][str(polygon_size)]["phase"]

    print(phases)


    for sample, (seed, phase) in enumerate(zip(seeds, phases)):
        print(phase, sample, seed)
        if phase == "chaos":
            A = auto.get_init_voronoi(polygon_size, sample=sample, seed=seed)

            largest_eigs = auto.get_lyapunov_seq(max_it=max_it, input=A, pretime=pretime, step=step)
            print(largest_eigs)

            path = f"lyapunov/{g_mju}_{g_sig}.pickle"


            try:
                with open(path, 'rb') as handle:
                    datafile = pickle.load(handle)
            except:
                datafile = []


            datafile.append({"seed": seed,
                    "polygon_size": polygon_size,
                    "sample": sample,
                    "pretime": pretime,
                    "multi_step_size": multi_step_size,
                    "largest_eigs": largest_eigs,
                    "phase": phase,
                    "max_it": max_it,
                    "step": step
                    })


            with open(path, 'wb') as handle:
                pickle.dump(datafile, handle, protocol=pickle.HIGHEST_PROTOCOL) 


def plot_lyapuno_largest_exp(auto, g_mju, g_sig, phases, pretimes, video=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"max": "#ff9585", "chaos": "#91d4c4"}

    # Set limits for x and y axes (similar to the image)

    # Add major gridlines
    ax.grid(True, which='major', color='gray', linestyle='-', linewidth=0.7)

    # Add minor gridlines
    ax.minorticks_on()
    ax.grid(True, which='minor', color='lightgray', linestyle='--', linewidth=0.5)


    # Set labels for axes (σ and μ)
    ax.set_xlabel(r'$n$', fontsize=14)
    ax.set_ylabel(r'$\lambda_n$', fontsize=14)
    size=3

    # Scatter plot with labels


    path = f"lyapunov/{g_mju}_{g_sig}.pickle"
    try:
        with open(path, 'rb') as handle:
            datafile = pickle.load(handle)
    except:
        print("no data")

    

    for item in datafile:
        try:
            print(item["phase"], item["pretime"])
            if item["phase"] in phases and item["pretime"] in pretimes:
                print(item["phase"], item["pretime"], item["step"], item["phase"], item["seed"], item["sample"], item["polygon_size"])
                
                mstep = item["multi_step_size"]*item["step"]
                tot_it = item["max_it"]*mstep
                ph = item["phase"]
                se = item["seed"]
                sa = item["sample"]
                Ts = np.arange(1, tot_it+1, mstep)
                rawvals = [eig[0].item() for eig in item["largest_eigs"]]
                vals = [np.log(eig[0].item())/(2*Ts[i]) for i, eig in enumerate(item["largest_eigs"])]
                ax.scatter(Ts, vals, s=5, label=item["sample"], color=colors[item["phase"]])
                if video:
                    A = auto.get_init_voronoi(polygon_size, sample=item["sample"], seed=item["seed"])
                    video_path = f"lyapunov/videos/{g_mju}_{g_sig}_{ph}_{sa}_{se}.gif"
                    auto.make_grad_video(time_steps = 100, config = A, video_path=video_path)

        except:
            continue
    plt.savefig(f"lyapunov/lplot{g_mju}_{g_sig}.png")

    


""" A = torch.rand((10000, 10000)).to(device)
B = torch.transpose(A, 0, 1)


C = torch.matmul(B, A)
lam = torch.lobpcg(C, k=1)[0]
print(lam)
 """

#print(torch.max(torch.abs(lam)))

#============================== PARAMETERS ================================

device = "cuda:1"
dt = 0.1
H, W = 100, 100

g_mju = 0.15  #0.11
g_sig = 0.015  #0.03



params = {'k_size': 27, 
          'mu': torch.tensor([[[g_mju]]], device=device), 
          'sigma': torch.tensor([[[g_sig]]], device=device), 
          'beta': torch.tensor([[[[1]]]], device=device), 
          'mu_k': torch.tensor([[[[0.5]]]], device=device), 
          'sigma_k': torch.tensor([[[[0.15]]]], device=device), 
          'weights': torch.tensor([[[1]]], device=device)}

polygon_size = 50
pretime = 1000

max_it = 50
multi_step_size = 10
step = 1

#===========================================================================

auto = BatchLeniaMC((1,H,W), dt, params=params, num_channels=1, device=device, multi_step_size=multi_step_size)
auto.to(device)

#generate_laypunov(auto, g_mju, g_sig, polygon_size, pretime, multi_step_size, max_it, step)

phases = ["chaos", "max"]
pretimes = [1000]

plot_lyapunov_largest_exp(auto, g_mju, g_sig, phases, pretimes, video=True)

#===========================================================================







""" for i, e in enumerate(eigs):
    print(np.log(np.abs(e.item()))/(i+1))
    vals.append(np.log(np.abs(e.item()))/(i+1))


path = f"{g_mju}_{g_sig}_{polygon_size}_{seed}_{sample}.pickle"



with open(path, 'wb') as handle:
    pickle.dump(vals, handle, protocol=pickle.HIGHEST_PROTOCOL)
 """
""" for _ in range(pretime):
    A = auto.step(A) """

""" start = time.time()
J = auto.jacob(A)
print(J.shape)
auto.get_eig(J)
print(time.time()-start) 

path = f"{g_mju}_{g_sig}.pickle"


with open(path, 'rb') as handle:
    datafile = pickle.load(handle)

for item in datafile:
    try:
        sample = item["sample"]
        phase = item["phase"]
        pretime = item["pretime"]
        if phase == "max" and pretime == 500:
            A = item["largest_eigs"]
    except:
        continue

vals = []
print(A)
for i, ten in enumerate(A):
    v = np.log(ten[0].item())/(i+1)
    vals.append(v)
    print(v)


"""


#auto.make_video(time_steps = 100, step=1, config = A, video_path="video.gif")