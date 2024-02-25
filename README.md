In this repository, we show the dimension independence of the infinite-dimensional MCMC algorithms, including pCN and pCNL, for the steady-state Darcy flow problem. 
The implementation recovers the figures in Subsection 3.4 of the paper "无限维贝叶斯反演理论、算法与应用"(Infinite-dimensional Bayesian inverse theories, algorithms, and applications) written in Chinese. 
The general references of these algorithms: 
1. S. L. Cotter, G. O. Roberts, A. M. Stuart and D. White, MCMC Methods for Functions: Modifying Old Algorithms to Make Them Faster, Statistical Science, 2013
2. M. Dashti, A. M. Stuart, The Bayesian Approch to Inverse Problems, Hankbook of Uncertainty Quantification, 2017
The discretized methods are illustrated in the following paper:
1. T. Bui-Thanh, O. Ghattas, J. Martin, G. Stadler, A computational framework for infinite-dimensional Bayesian inverse problems part I: The linearized case, with application to global seismic inversion, SIAM J. Sci. Comput., 2013

You need to run the programs in the following order: 
1. DarcyFlow/generate_data.py
2. DarcyFlow/MCMC/VanillaMCMC_dim_compare.py
3. DarcyFlow/MCMC/pCN_dim_independence.py
4. DarcyFlow/MCMC/pCN_dim_wrong_prior.py
5. DarcyFlow/MCMC/pCNL_dim_independence.py
6. DarcyFlow/MCMC/draw.py

The figure will be stored in the folder DarcyFlow/MCMC/results/

To run the program, you need to install FEniCSx(Version 0.7) https://fenicsproject.org/, numpy, scipy, and matplotlib. 
