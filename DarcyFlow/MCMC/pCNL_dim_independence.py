import os
from multiprocessing import Pool, cpu_count
cpu_num = 10#cpu_count()
from mpi4py import MPI
import dolfinx as dlf
from dolfinx import fem
import numpy as np
from pathlib import Path
import scienceplots
import matplotlib.pyplot as plt
plt.style.use('science')
plt.style.use(['science','ieee'])

import sys
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from DarcyFlow.common import EquSolverDarcyFlow, ModelDarcyFlow
from core.probability import GaussianElliptic2
from core.noise import NoiseGaussianIID
from core.sample import pCN, VanillaMCMC, pCNL
from core.misc import project


data_folder = Path("data")
results_folder = Path("results")
data_folder.mkdir(exist_ok=True, parents=True)
results_folder.mkdir(exist_ok=True, parents=True)

noise_level = 0.05
data = {"coordinates": None, "data": None}
data["coordinates"] = np.load(data_folder/"measure_coordinates.npy", allow_pickle=True)
datafile = "noisy_data_" + str(noise_level) + ".npy"
data["data"] = np.load(data_folder/datafile, allow_pickle=True)
clean_data = np.load(data_folder/"clean_data.npy")

with dlf.io.XDMFFile(MPI.COMM_WORLD, data_folder/"true_function.xdmf", "r") as xdmf:
    msh = xdmf.read_mesh()
f = open(data_folder/'fun_type_info.txt', "r")
fun_type = f.read()
f = open(data_folder/'fun_degree_info.txt', "r")
fun_degree = int(f.read())
f.close()
Vh_true = fem.FunctionSpace(msh, (fun_type, fun_degree))
true_fun = fem.Function(Vh_true)
true_fun.x.array[:] = np.array(np.load(data_folder/"fun_data.npy"))

def sampling_with_different_dim(params):
    nx, beta = params
    length_total = np.int64(1e4)
    ## nx is the discretized dimension of the Darcy flow problem
    msh = dlf.mesh.create_unit_square(MPI.COMM_SELF, nx, nx, dlf.mesh.CellType.triangle)
    equ_solver = EquSolverDarcyFlow(msh)
    param0 = project(true_fun, equ_solver.Vh)
    ## generate truth and save
    params_prior = {
        "theta": lambda x: 0.1 + 0.0*x[0],
        "ax": lambda x: 0.5 + 0.0*x[0],
        "mean": lambda x: 0.0*x[0]
    }
    prior = GaussianElliptic2(equ_solver.Vh, params_prior)
    noise = NoiseGaussianIID(len(data["data"]))
    noise.set_parameters(std_dev=noise_level*max(abs(clean_data)))
    model = ModelDarcyFlow(prior, equ_solver, noise, data)

    sampler = pCNL(model, beta=beta, reduce_chain=np.int64(1e3), save_path=None, grad_smooth_degree=1e-2)
    print("Starting sampling with dim = " + str(nx) + " and beta = " + str(beta) + " ......")
    sampler.sampling(len_chain=length_total, u0=np.array(param0.x.array))
    acc_rate = sampler.acc_rate
    print("Sampling with dim = " + str(nx) + " and beta = " + str(beta) + " ended!")
    del prior, noise, equ_solver, msh, model, sampler, param0
    return acc_rate


# dims = np.array([10, 20, 50, 100])
# betas = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
dims = np.load(results_folder/"dims.npy")
betas = np.load(results_folder/"betas.npy")

test_params = []
for beta in betas:
    for dim in dims:
        test_params.append((dim, beta))

with Pool(processes=min(cpu_num, len(test_params))) as pool:
    acc_rates = pool.map(sampling_with_different_dim, test_params)

acc_rates = np.array(acc_rates).reshape(len(betas), len(dims))
np.save(results_folder/"betas", betas)
np.save(results_folder/"pCNL_dim_independence", acc_rates)

plt.figure()
for idx, dim in enumerate(dims):
    plt.plot(betas, acc_rates[:,idx], label="dim="+str(dim))
    plt.xscale("log")
    plt.legend()
plt.savefig(results_folder/"pCNL_dim_independence.png")
plt.close()


