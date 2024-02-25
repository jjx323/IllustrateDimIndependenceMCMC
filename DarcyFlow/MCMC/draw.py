import numpy as np
from pathlib import Path
import scienceplots
import matplotlib.pyplot as plt
plt.style.use('science')
plt.style.use(['science','ieee'])

import sys, os
sys.path.append(os.pardir)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))


results_folder = Path("results")
dims = np.load(results_folder/"dims.npy")
betas = np.load(results_folder/"betas.npy")

plt.rcParams['font.family'] = 'Times New Romann'
plt.rcParams['font.size'] = 15
plt.figure(figsize=(11, 10))
plt.subplot(2,2,1)
for idx, dim in enumerate(dims):
    acc_rates_vanilla = np.load(results_folder / "VanillaMCMC_dim_priorC.npy")
    plt.plot(betas, acc_rates_vanilla[:,idx], label="dim="+str(dim))
    plt.xscale("log")
    plt.xlabel("Beta")
    plt.ylabel("Accept Rates")
    plt.legend(loc="lower left")
    plt.title("(a) Vanilla MCMC", pad=12.0)
plt.subplot(2,2,2)
for idx, dim in enumerate(dims):
    acc_rates_pCN = np.load(results_folder / "pCN_dim_independence.npy")
    plt.plot(betas, acc_rates_pCN[:,idx], label="dim="+str(dim))
    plt.xscale("log")
    plt.xlabel("Beta")
    plt.ylabel("Accept Rates")
    plt.legend(loc="lower left")
    plt.title("(b) pCN(Gaussian prior)", pad=12.0)
plt.subplot(2,2,3)
for idx, dim in enumerate(dims):
    acc_rates_pCN_priorI = np.load(results_folder / "pCN_dim_priorI.npy")
    plt.plot(betas, acc_rates_pCN_priorI[:,idx], label="dim="+str(dim))
    plt.xscale("log")
    plt.xlabel("Beta")
    plt.ylabel("Accept Rates")
    plt.legend(loc="lower left")
    plt.title("(c) pCN(white noise prior)", pad=12.0)
plt.subplot(2,2,4)
for idx, dim in enumerate(dims):
    acc_rates_pCNL = np.load(results_folder / "pCNL_dim_independence.npy")
    plt.plot(betas, acc_rates_pCNL[:,idx], label="dim="+str(dim))
    plt.xscale("log")
    plt.xlabel("Beta")
    plt.ylabel("Accept Rates")
    plt.legend(loc="lower left")
    plt.title("(d) pCNL(Gaussian prior)", pad=12.0)
plt.tight_layout(pad=1, w_pad=1, h_pad=2)
plt.savefig(results_folder/"dim_independence_compare.png")
plt.close()
