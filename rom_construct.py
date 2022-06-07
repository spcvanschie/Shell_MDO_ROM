import numpy as np
import scipy as sp
import openmdao.api as om

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *
from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.occ_utils import *

from opt_utils import *
from shell_sim import *
from rom_utils import *

import time
import os
import matplotlib.pyplot as plt

class POD_ROM():
    def __init__(self, dofs_per_surf, ER_threshold, subtract_mean=True):
        self.ER_threshold = ER_threshold
        self.dofs_per_patch = dofs_per_surf
        self.iga_dof = np.sum(dofs_per_surf, dtype="int64")
        self.num_surfs = len(dofs_per_surf)
        self.cwd = os.getcwd()
        self.folder_name = os.path.join(self.cwd, 'snapshot_data_FOM_{}_DoFs'.format(self.iga_dof))
        self.subtract_mean = subtract_mean
        self.avg_vec = None
        self.avg_vecs_list = None

        # gather snapshot data
        self.gather_data()
        # compute the POD basis for each surface
        self.compute_POD_bases()
        self.compute_global_POD_basis()
        self.construct_pod_nest_mat(self.V_mats_blocks)
    
    def gather_data(self):
        print("Gathering snapshot data...")
        gather = True
        i = 1  # start at index 1 since the data at index 0 and 1 are usually identical (I don't know why)

        disp_list = []
        t_list = []

        while gather:
            # check whether file exists
            exists = os.path.exists(os.path.join(self.folder_name, "disp_{}".format(i)))
            if exists:
                disp_snapshot = np.genfromtxt(os.path.join(self.folder_name, "disp_{}".format(i)), delimiter=",")
                t_snapshot = np.genfromtxt(os.path.join(self.folder_name, "h_th_{}".format(i)), delimiter=",")
                disp_list += [disp_snapshot]
                t_list += [t_snapshot]
                i += 1
                
                # for t_element in t_list:
                    # np.max(np.abs(t_list[0]- t_list[1]))
            else:
                gather = False

        # we've gathered all snapshot data, so we accumulate everything into matrices
        self.num_snaps = len(disp_list)
        X_mat = np.zeros((self.iga_dof, self.num_snaps))
        t_mat = np.zeros((self.num_surfs, self.num_snaps))

        for i in range(self.num_snaps):
            X_mat[:, i] = disp_list[i]
            t_mat[:, i] = t_list[i]

        if self.subtract_mean:
            # subtract mean from X_mat
            avg_vec = np.sum(X_mat, axis=1)/self.num_snaps
            avg_mat = np.tile(avg_vec[:, None], (1, self.num_snaps))

            X_mat = np.subtract(X_mat, avg_mat)
            self.avg_vec = avg_vec
            self.avg_vecs_list = []

        # split snapshot matrix X_mat into separate matrices for each surface
        X_mats_list = []
        dofs_cumsum = [0]+list(np.cumsum(self.dofs_per_patch))

        for i in range(self.num_surfs):
            X_mats_list += [X_mat[dofs_cumsum[i]:dofs_cumsum[i+1], :]]
            if self.subtract_mean:
                self.avg_vecs_list += [avg_vec[dofs_cumsum[i]:dofs_cumsum[i+1]]]
        
        # define complete data matrices and X per surface as properties
        self.X_mat = X_mat
        self.t_mat = t_mat
        self.X_mats_list = X_mats_list
    
    def compute_POD_bases(self):
        # compute SVD of each surface
        V_mats_blocks = [[None for i in range(self.num_surfs)] for j in range(self.num_surfs)]
        V_mats_list = []
        r_list = []
        sing_vals_list = []

        for i in range(self.num_surfs):
            u, s, _ = np.linalg.svd(self.X_mats_list[i])
            sing_vals_list += [s]
            
            # compute Energy Ratio defects
            ER_defect = 1.-np.cumsum(s)/np.sum(s)

            # compute index of first entry that satisfies the ER threshold
            r = next(x for x, val in enumerate(ER_defect)
                                  if val <= 1.-self.ER_threshold)
            
            # save POD basis for surface
            V_mats_blocks[i][i] = u[:, :r+1]
            V_mats_list += [u[:, :r+1]]

            # save number of basis vectors used on surface
            r_list += [r+1]
        
        self.V_mats_blocks = V_mats_blocks
        self.V_mat = sp.sparse.block_diag(V_mats_list)  # V_mat is a sparse diagonal matrix that contains the POD basis for each surface 
        self.r_list = r_list
        self.sing_vals_list = sing_vals_list
        print("POD basis vectors per surface: {}".format(r_list))
    
    def compute_global_POD_basis(self):
        u, s, _ = np.linalg.svd(self.X_mat)
            
        # compute Energy Ratio defects
        ER_defect = 1.-np.cumsum(s)/np.sum(s)
        r = next(x for x, val in enumerate(ER_defect)
                                  if val <= 1.-self.ER_threshold)
        
        self.V_mat_full = u[:, :r+1]
        self.r = r
        print("Global POD basis size: {}".format(r))
        self.sing_vals = s

        # break V_mat_full into blocks for further processing
        V_mats_blocks = [[None for i in range(self.num_surfs)] for j in range(self.num_surfs)]
        dofs_cumsum = [0] + list(np.cumsum(self.dofs_per_patch))
        for i in range(self.num_surfs):
            for j in range(self.num_surfs):
                V_mats_blocks[i][j] = self.V_mat_full[dofs_cumsum[i]:dofs_cumsum[i+1], :]
        self.V_mats_blocks = V_mats_blocks

    def construct_pod_nest_mat(self, V_mats):
        # construct a nested block matrix representation of the POD basis V_mat that can be used in PENGoLINS

        # A_list entries are either None or petsc4py.PETSc.Mat seqaij matrices
        V_mat_blocks = [[None for i in range(self.num_surfs)] for j in range(self.num_surfs)]
        V_blocks_petsc = [[None for i in range(self.num_surfs)] for j in range(self.num_surfs)]

        for i in range(self.num_surfs):
            for j in range(self.num_surfs):
                # convert V_mats[i][j] to PETSc matrix
                mat = V_mats[i][j]
                if mat is not None:
                    mat_sparse = sp.sparse.csr_matrix(mat)
                    mat_petsc = PETSc.Mat().createAIJWithArrays(size=(mat.shape[0],mat.shape[1]), csr=(mat_sparse.indptr, mat_sparse.indices,mat_sparse.data))

                    V_mat_blocks[i][j] = mat_sparse
                    V_blocks_petsc[i][j] = mat_petsc
            
        self.V_blocks_np = V_mat_blocks
        self.V_blocks_petsc = V_blocks_petsc

        if self.subtract_mean:
            b_list = []
                
            for i in range(self.num_surfs):
                b_subvec = PETSc.Vec().createWithArray(self.avg_vecs_list[i])
                b_list += [b_subvec]

            self.avg_iga_vec_petsc = create_nest_PETScVec(b_list)
        
        

class OI_ROM():
    def __init__(self, POD_model):
        # import POD model that contains snapshot data and the POD projection operator
        self.POD_model = POD_model
    
    def construct_H_r_mask(self, mapping_list):
        # Mapping_list contains the indices of the blocks that interact with one another in a coupling.
        # (i, j) in mapping_list corresponds to adding masking blocks 
        # at block indices (i, i, i), (j, j, j), (i, j, j) and (j, i ,i)
        r_cumsum = [0] + list(np.cumsum(self.POD_model.r_list))

        block_idx_mat = np.zeros((self.POD_model.num_surfs, self.POD_model.num_surfs, self.POD_model.num_surfs))
        mask_mat = np.zeros((np.sum(self.POD_model.r_list), np.sum(self.POD_model.r_list), np.sum(self.POD_model.r_list)))

        for i in range(len(mapping_list)):
            s_ind0, s_ind1 = mapping_list[i]
            block_idx_mat[s_ind0, s_ind0, s_ind0] = 1
            block_idx_mat[s_ind1, s_ind1, s_ind1] = 1
            block_idx_mat[s_ind0, s_ind1, s_ind1] = 1
            block_idx_mat[s_ind1, s_ind0, s_ind0] = 1

            mask_mat[r_cumsum[s_ind0]:r_cumsum[s_ind0+1], r_cumsum[s_ind0]:r_cumsum[s_ind0+1], r_cumsum[s_ind0]:r_cumsum[s_ind0+1]] = 1
            mask_mat[r_cumsum[s_ind1]:r_cumsum[s_ind1+1], r_cumsum[s_ind1]:r_cumsum[s_ind1+1], r_cumsum[s_ind1]:r_cumsum[s_ind1+1]] = 1
            mask_mat[r_cumsum[s_ind0]:r_cumsum[s_ind0+1], r_cumsum[s_ind1]:r_cumsum[s_ind1+1], r_cumsum[s_ind1]:r_cumsum[s_ind1+1]] = 1
            mask_mat[r_cumsum[s_ind1]:r_cumsum[s_ind1+1], r_cumsum[s_ind0]:r_cumsum[s_ind0+1], r_cumsum[s_ind0]:r_cumsum[s_ind0+1]] = 1

        # set the values (i,j,k) with k > j to zero, since the weights (i, j, k) will be identical to (i, k, j) (u_j*u_k = u_k*u_j)
        mask_mat = np.tril(mask_mat)
        block_idx_mat = np.tril(block_idx_mat)

        self.block_idx_mask = block_idx_mat
        self.mask_mat = mask_mat
    
    def construct_H_r_mask_decoupledshells(self):
        r_cumsum = [0] + list(np.cumsum(self.POD_model.r_list))

        block_idx_mat = np.zeros((self.POD_model.num_surfs, self.POD_model.num_surfs, self.POD_model.num_surfs))
        mask_mat = np.zeros((np.sum(self.POD_model.r_list), np.sum(self.POD_model.r_list), np.sum(self.POD_model.r_list)))

        for i in range(len(self.POD_model.r_list)):
            block_idx_mat[i, i, i] = 1

            mask_mat[r_cumsum[i]:r_cumsum[i+1], r_cumsum[i]:r_cumsum[i+1], r_cumsum[i]:r_cumsum[i+1]] = 1

        # set the values (i,j,k) with k > j to zero, since the weights (i, j, k) will be identical to (i, k, j) (u_j*u_k = u_k*u_j)
        mask_mat = np.tril(mask_mat)
        block_idx_mat = np.tril(block_idx_mat)

        self.block_idx_mask = block_idx_mat
        self.mask_mat = mask_mat
    
    # def train_H_r_decoupledshells(self):
    #     X_list = self.POD_model.X_mats_list
    #     V_list = self.POD_model.V_mats_list
        

    #     surf_dof_range = [0] + list(np.cumsum(self.POD_model.dofs_per_patch))
    
    #     # loop over surfaces to solve least-squares problem independently for each
    #     for i in range(len(X_list)):
    #         r_i = self.POD_model.r_list[i]
    #         num_snaps = X_list[i].shape[1]
    #         t_vec = self.POD_model.t_mat[i, :]
    #         V_mat = V_list[i]
    #         X_mat = X_list[i]

    #         H_d_i = np.tril(np.ones((r_i, r_i, r_i)))
    #         H_r_i = np.tril(np.ones((r_i, r_i, r_i)))

    #         for j in range(num_snaps):
    #             # project snapshot data to POD-space
    #             u_r_j = V_mat.T@X_mat[:, j]

    #             # compute parametric terms (simply scale them with the appropriate thickness for now)
    #             alpha_d_approx = t_vec[j]
    #             alpha_r_approx = t_vec[j]**3

    #             # incorporate displacement and thickness values into matrices
    #             H_d_i_mult = alpha_d_approx*np.einsum('ijk,k,j->ijk', H_d_i, u_r_j, u_r_j)
    #             H_r_i_mult = alpha_r_approx*np.einsum('ijk,k,j->ijk', H_r_i, u_r_j, u_r_j)


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
    
    folder_name = os.path.join(cwd, 'snapshot_data_{}_DoFs'.format(shell_sim.iga_dof))

    rom = POD_ROM(shell_sim.iga_dofs, 0.99)


    # plot singular values of each surface
    fig, ax = plt.subplots(1, 1)
    for i in range(shell_sim.num_surfs):
        ax.scatter(np.linspace(1,len(rom.sing_vals_list[i])+1, len(rom.sing_vals_list[i])), rom.sing_vals_list[i])
    ax.set_yscale("log")
    # ax2.set_yscale("log")
    ax.set_ylabel("Singular value")
    ax.grid(True, which="both")
    ax.set_xlim(0, len(rom.sing_vals_list[i]))

    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(1, 1)
    for i in range(shell_sim.num_surfs):
        ER_defect = 1.-np.cumsum(rom.sing_vals_list[i])/np.sum(rom.sing_vals_list[i])
        ax.semilogy(np.linspace(1,len(rom.sing_vals_list[i]), len(rom.sing_vals_list[i])-1), ER_defect[:-1])
    ax.set_ylabel("1 - Energy Ratio")
    ax.grid(True)
    ax.set_xlim(0, len(rom.sing_vals_list[i]))

    plt.grid(True, which="both")
    plt.show()

    print("fin")