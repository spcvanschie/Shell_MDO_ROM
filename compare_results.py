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


def gather_data(pod_file, comp_file_list):
    print("Gathering snapshot data...")
    all_file_list = [pod_file] + comp_file_list
    disp_lists = [[] for i in range(len(all_file_list))]
    t_lists = [[] for i in range(len(all_file_list))]

    disp_mat_list = []
    t_mat_list = []

    for j, stor_name in enumerate(all_file_list):
        gather = True
        i = 1  # start at index 1 since the data at index 0 and 1 are usually identical (I don't know why)
        while gather:
            # check whether file exists
            exists = os.path.exists(os.path.join(stor_name, "disp_{}".format(i)))
            if exists:
                disp_snapshot = np.genfromtxt(os.path.join(stor_name, "disp_{}".format(i)), delimiter=",")
                t_snapshot = np.genfromtxt(os.path.join(stor_name, "h_th_{}".format(i)), delimiter=",")
                disp_lists[j] += [disp_snapshot]
                t_lists[j] += [t_snapshot]
                i += 1
            else:
                gather = False

        # we've gathered all snapshot data, so we accumulate everything into matrices

        disp_mat = np.zeros((disp_lists[j][0].shape[0], len(disp_lists[j])))
        t_mat = np.zeros((t_lists[j][0].shape[0], len(t_lists[j])))
        for k in range(len(disp_lists[j])):
            disp_mat[:, k] = disp_lists[j][k]
            t_mat[:, k] = t_lists[j][k]
        
        disp_mat_list += [disp_mat]
        t_mat_list += [t_mat]
    
    return disp_mat_list, t_mat_list

def comp_2norm_errors(disp_mat_list, t_mat_list, max_snap_idx=None):
    fom_disp = disp_mat_list[0]
    fom_t = t_mat_list[0]

    rel_err_lists = [[] for i in range(len(disp_mat_list)-1)]

    for j, rom_disp in enumerate(disp_mat_list[1:]):
        if max_snap_idx is None:
            max_snap_idx = min(rom_disp.shape[1], fom_disp.shape[1])

        for i in range(max_snap_idx):
            fom_col = fom_disp[:, i]
            rom_col = rom_disp[:, i]

            fom_t_snap = fom_t[:, i]
            rom_t_snap = t_mat_list[j+1][:, i]
            if np.sum(np.abs(fom_t_snap-rom_t_snap)) > 1e-15:
                print("FOM and ROM are at different locations in the parameter space, 2-norm difference: {}".format(np.linalg.norm(fom_t_snap-rom_t_snap)))
            
            rel_err_lists[j] += [np.linalg.norm(fom_col-rom_col)/np.linalg.norm(fom_col)]
    
    return rel_err_lists


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
    
    fom_folder_name = os.path.join(cwd, 'snapshot_data_FOM_{}_DoFs'.format(shell_sim.iga_dof))
    sparsePOD_folder_name = os.path.join(cwd, 'snapshot_data_sparsePOD_ROM_{}_DoFs'.format(shell_sim.iga_dof))
    fullPOD_folder_name = os.path.join(cwd, 'snapshot_data_fullPOD_ROM_{}_DoFs'.format(shell_sim.iga_dof))

    comp_datasets = [fullPOD_folder_name, sparsePOD_folder_name]

    disp_list, t_list = gather_data(fom_folder_name, comp_datasets)

    comp_2norm_errors(disp_list, t_list, max_snap_idx=22)