import os
from multiprocessing import Pool, cpu_count
cpu_num = cpu_count()
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
from core.optimizer import NewtonCG
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

def eval_map(nx):
    msh = dlf.mesh.create_unit_square(MPI.COMM_WORLD, nx, nx)
    equ_solver = EquSolverDarcyFlow(msh, degree=1)
    ## generate truth and save
    params = {
        "theta": lambda x: 0.1 + 0.0*x[0],
        "ax": lambda x: 0.5 + 0.0*x[0],
        "mean": lambda x: 0.0*x[0]
    }
    prior = GaussianElliptic2(equ_solver.Vh, params)
    noise = NoiseGaussianIID(len(data["data"]))
    noise.set_parameters(std_dev=noise_level*max(abs(clean_data)))
    model = ModelDarcyFlow(prior, equ_solver, noise, data)

    ## set optimizer NewtonCG
    newton_cg = NewtonCG(model=model)
    max_iter = 200

    ## Without a good initial value, it seems hard for us to obtain a good solution
    init_fun = fem.Function(model.equ_solver.Vh)
    init_fun.x.array[:] = 0.0
    newton_cg.re_init(init_fun.x.array)

    loss_pre = model.loss()[0]
    for itr in range(max_iter):
        # newton_cg.descent_direction(cg_max=50, method='cg_my')
        newton_cg.descent_direction(cg_max=3, method='bicgstab')
        print(newton_cg.hessian_terminate_info)
        newton_cg.step(method='armijo', show_step=False)
        if newton_cg.converged == False:
            break
        loss = model.loss()[0]
        print("iter = %2d/%d, loss = %.4f" % (itr+1, max_iter, loss))
        if np.abs(loss - loss_pre) < 1e-3*loss:
            print("Iteration stoped at iter = %d" % itr)
            break
        loss_pre = loss

    estimated_param = fem.Function(model.equ_solver.Vh)
    estimated_param.x.array[:] = np.array(newton_cg.mk.copy())
    return estimated_param

# try to sampling from the MAP estimate, we may avoid the long mixing procedure
# estimated_MAP = eval_map(nx=450)

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
    # param0 = project(estimated_MAP, equ_solver.Vh)
    param0 = project(true_fun, equ_solver.Vh)
    # param0 = fem.Function(equ_solver.Vh)
    # param0.x.array[:] = 0.0
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

    sampler = pCN(model, beta=beta, reduce_chain=np.int64(1e3), save_path=None)
    print("Starting sampling with dim = " + str(nx) + " and beta = " + str(beta) + " ......")
    sampler.sampling(len_chain=length_total, u0=np.array(param0.x.array))
    acc_rate = sampler.acc_rate
    print("Sampling with dim = " + str(nx) + " and beta = " + str(beta) + " ended!")
    del prior, noise, equ_solver, msh, model, sampler, param0
    return acc_rate


# dims = np.array([10, 20, 50])
# betas = np.array([1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2])
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
np.save(results_folder/"pCN_dim_independence", acc_rates)

plt.figure()
for idx, dim in enumerate(dims):
    plt.plot(betas, acc_rates[:,idx], label="dim="+str(dim))
    plt.xscale("log")
    plt.legend()
plt.savefig(results_folder/"pCN_dim_independence.png")
plt.close()


