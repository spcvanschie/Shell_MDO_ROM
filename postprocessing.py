import numpy as np
import openmdao.api as om

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *
from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.occ_utils import *

from opt_utils import *
from shell_sim import *

import time
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    E = Constant(71.7e9)  # Young's modulus, Pa
    nu = Constant(0.33)  # Poisson's ratio
    rho = Constant(2.81e3)  # Material density, kg/m^3
    n_load = Constant(2.5)  # Load factor
    h_th = Constant(20.0e-3)  # Thickness of surfaces, m

    p = 3  # spline order
    filename_igs = "eVTOL_wing_structure.igs"
    cwd = os.getcwd()

    # declare ShellSim to define some useful information for data processing
    shell_sim = ShellSim(p, E, nu, rho, n_load, 
                    filename_igs, comm=worldcomm)
    
    folder_name = os.path.join(cwd, 'snapshot_data_FOM_{}_DoFs'.format(shell_sim.iga_dof))

    # gather all snapshot information from written files
    print("Gathering snapshot data...")
    gather = True
    i = 1  # start at index 1 since the data at index 0 and 1 are usually identical (I don't know why)

    disp_list = []
    t_list = []

    while gather:
        # check whether file exists
        exists = os.path.exists(os.path.join(folder_name, "disp_{}".format(i)))
        if exists:
            disp_snapshot = np.genfromtxt(os.path.join(folder_name, "disp_{}".format(i)), delimiter=",")
            t_snapshot = np.genfromtxt(os.path.join(folder_name, "h_th_{}".format(i)), delimiter=",")
            disp_list += [disp_snapshot]
            t_list += [t_snapshot]
            i += 1
            
            # for t_element in t_list:
                # np.max(np.abs(t_list[0]- t_list[1]))
        else:
            gather = False

    # we've gathered all snapshot data, so we accumulate everything into matrices
    num_snaps = len(disp_list)
    X_mat = np.zeros((shell_sim.iga_dof, num_snaps))
    t_mat = np.zeros((shell_sim.num_surfs, num_snaps))

    for i in range(num_snaps):
        X_mat[:, i] = disp_list[i]
        t_mat[:, i] = t_list[i]

    # subtract mean from X_mat
    avg_vec = np.sum(X_mat, axis=1)/num_snaps
    avg_mat = np.tile(avg_vec[:, None], (1, num_snaps))

    X_mat = np.subtract(X_mat, avg_mat)


    # split snapshot matrix X_mat into separate matrices for each surface
    X_mats_list = []
    dofs_cumsum = [0]+list(np.cumsum(shell_sim.iga_dofs))

    for i in range(shell_sim.num_surfs):
        X_mats_list += [X_mat[dofs_cumsum[i]:dofs_cumsum[i+1], :]]

    # compute SVD of each surface
    sing_vals_list = []
    ER_list = []
    for i in range(shell_sim.num_surfs):
        u, s, vh = np.linalg.svd(X_mats_list[i])
        sing_vals_list += [s]
        # compute energy ratios
        ER_list += [1.-np.cumsum(s)/np.sum(s)]

    # plot singular values of the whole matrix
    fig, ax = plt.subplots(1, 1)
    u, s, _ = np.linalg.svd(X_mat)
    for i in range(shell_sim.num_surfs):
        ax.scatter(np.linspace(1,len(s)+1, len(s)), s)
    ax.set_yscale("log")
    # ax2.set_yscale("log")
    ax.set_ylabel("Singular value")
    ax.grid(True)#, which="both")
    ax.set_xlim(0, len(sing_vals_list[i]))

    plt.grid(True)
    plt.show()

    # fig, ax = plt.subplots(1, 1)
    # for i in range(shell_sim.num_surfs):
    #     ax.semilogy(np.linspace(1,len(sing_vals_list[i]), len(sing_vals_list[i])-1), ER_list[i][:-1])
    # ax.set_ylabel("1 - Energy Ratio")
    # ax.grid(True)
    # ax.set_xlim(0, len(sing_vals_list[i]))

    # plt.grid(True, which="both")
    # plt.show()


    # plot singular values of each surface
    fig, ax = plt.subplots(1, 1)
    for i in range(shell_sim.num_surfs):
        ax.scatter(np.linspace(1,len(sing_vals_list[i])+1, len(sing_vals_list[i])), sing_vals_list[i])
    ax.set_yscale("log")
    # ax2.set_yscale("log")
    ax.set_ylabel("Singular value")
    ax.grid(True, which="both")
    ax.set_xlim(0, len(sing_vals_list[i]))

    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    for i in range(shell_sim.num_surfs):
        ax.semilogy(np.linspace(1,len(sing_vals_list[i]), len(sing_vals_list[i])-1), ER_list[i][:-1])
    ax.set_ylabel("1 - Energy Ratio")
    ax.grid(True)
    ax.set_xlim(0, len(sing_vals_list[i]))

    plt.grid(True, which="both")
    plt.show()

    print("fin")