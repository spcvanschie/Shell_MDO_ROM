from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *
from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.occ_utils import *
from opt_utils import *

import openmdao.api as om 

from geom_utils import *


class ShellSim:
    def __init__(self, p, E, nu, rho, n_load, 
                 filename_igs, geom_scale=2.54e-5, penalty_coefficient=1.0e3, comm=worldcomm):

        self.p = p
        self.E = E
        self.nu = nu
        self.rho = rho
        self.n_load = n_load
        self.penalty_coefficient = penalty_coefficient
        self.comm = comm

        self.print_info = True

        print("Importing geometry...")
        igs_shapes = read_igs_file(filename_igs, as_compound=False)
        evtol_surfaces = [topoface2surface(face, BSpline=True) 
                        for face in igs_shapes]

        # Outer skin indices: list(range(12, 18))
        # Spars indices: [78, 92, 79]
        # Ribs indices: list(range(80, 92))
        wing_indices = list(range(12, 18)) + [78, 92, 79]  + list(range(80, 92))
        wing_surfaces = [evtol_surfaces[i] for i in wing_indices]
        self.num_surfs = len(wing_surfaces)
        if mpirank == 0:
            print("Number of surfaces:", self.num_surfs)

        num_pts_eval = [16]*self.num_surfs
        u_insert_list = [8]*self.num_surfs
        v_insert_list = [8]*self.num_surfs
        ref_level_list = [1]*self.num_surfs
        for i in [4,5]:
            if ref_level_list[i] > 4:
                ref_level_list[i] = 2
            elif ref_level_list[i] <=4 and ref_level_list[i] >= 1:
                ref_level_list[i] = 1

        u_num_insert = []
        v_num_insert = []
        for i in range(len(u_insert_list)):
            u_num_insert += [ref_level_list[i]*u_insert_list[i]]
            v_num_insert += [ref_level_list[i]*v_insert_list[i]]

        # Geometry preprocessing and surface-surface intersections computation
        self.preprocessor = OCCPreprocessing(wing_surfaces, reparametrize=True, 
                                        refine=True)
        self.preprocessor.reparametrize_BSpline_surfaces(num_pts_eval, num_pts_eval,
                                                    geom_scale=geom_scale,
                                                    remove_dense_knots=True,
                                                    rtol=1e-4)
        self.preprocessor.refine_BSpline_surfaces(p, p, u_num_insert, v_num_insert, 
                                            correct_element_shape=True)
        print("Computing intersections...")
        self.preprocessor.compute_intersections(mortar_refine=2)

        if mpirank == 0:
            print("Total DoFs:", self.preprocessor.total_DoFs)
            print("Number of intersections:", self.preprocessor.num_intersections_all)
        
        if mpirank == 0:
            print("Creating splines...")
        
        self.h_th = self.num_surfs*[Constant(0.01)]
        # Create tIGAr extracted spline instances to define baseline geometry
        self.Geometry = GeometryWithLinearTransformations(self.preprocessor)
        self.splines = self.Geometry.baseline_surfs

        self.fe_dofs = [spline.M.size(0) for spline in self.splines]
        self.iga_dofs = [spline.M.size(1) for spline in self.splines]
        
        self.fe_dof = np.sum(self.fe_dofs)
        self.iga_dof = np.sum(self.iga_dofs)

        # Create non-matching problem
        self.problem = NonMatchingCoupling(self.splines, self.E, self.h_th, self.nu, comm=self.comm)
        self.nonmatching_setup_is_done = False
        self.nonmatching_setup()

        self.spline_funcs = self.problem.spline_funcs
        self.spline_test_funcs = self.problem.spline_test_funcs

        # Create nest PETSc vector for unknowns in IGA DoFs
        self.u_iga_list = []
        for spline_ind in range(self.num_surfs):
            self.u_iga_list += [zero_petsc_vec(
                             self.splines[spline_ind].M.size(1), 
                             comm=self.splines[spline_ind].comm)]
        self.u_iga_nest = create_nest_PETScVec(self.u_iga_list, 
                                               comm=self.comm)
        
        # define external load vector
        self.Body_weight = 1500  # kg
        self.update_external_loads()
        self.update_SVK_residuals()


    def nonmatching_setup(self, penalty_coefficient=1.0e3, family0='CG', degree0=1, family1='CG', degree1=1):
        self.problem.create_mortar_meshes(self.preprocessor.mortar_nels)
        self.problem.create_mortar_funcs(family0, degree0)
        self.problem.create_mortar_funcs_derivative(family1, degree1)
        self.problem.mortar_meshes_setup(self.preprocessor.mapping_list, 
                                         self.preprocessor.intersections_para_coords, 
                                         penalty_coefficient)
        self.nonmatching_setup_is_done = True

    def update_h_th(self, h_th):
        # convert numpy vector h_th to a list of Constant() objects for internal use
        h_th_list = [Constant(h) for h in h_th]
        self.h_th = h_th_list
        self.problem.h_th = h_th_list

    def update_displacements(self, u_array):
        # update the stored displacement values
        update_nest_vec(u_array, self.u_iga_nest, comm=self.comm)
        u_iga_sub = self.u_iga_nest.getNestSubVecs()
        for spline_ind in range(self.num_surfs):
            self.iga2fe_dofs(u_iga_sub[spline_ind], spline_ind)

        for i in range(len(self.problem.transfer_matrices_list)):
            for j in range(len(self.problem.transfer_matrices_list[i])):
                for k in range(len(self.problem.transfer_matrices_list[i][j])):
                    A_x_b(self.problem.transfer_matrices_list[i][j][k], 
                        self.problem.spline_funcs[
                            self.problem.mapping_list[i][j]].vector(), 
                        self.problem.mortar_vars[i][j][k].vector())

    def iga2fe_dofs(self, u_iga_petsc, spline_ind):
        # updates the displacement vector in-place
        u_fe_petsc = v2p(self.spline_funcs[spline_ind].vector())
        M_petsc = m2p(self.splines[spline_ind].M)
        M_petsc.mult(u_iga_petsc, u_fe_petsc)
        u_fe_petsc.ghostUpdate()
        u_fe_petsc.assemble()

    def update_external_loads(self):
        # Compute magnitude of weight load
        wing_volume = 0
        for i in range(self.num_surfs):
            wing_volume += assemble(self.h_th[i]*self.splines[i].dx)

        wing_weight = wing_volume*self.rho  # kg

        # if mpirank == 0:
            # print("Wing mass: {} kg".format(wing_weight))

        # Weight is a constant volumetric load in negative z-direction
        f1 = as_vector([Constant(0.0), Constant(0.0), self.n_load*Constant(9.81)*(self.Body_weight + wing_weight)/wing_volume])

        # Distributed downward load
        self.loads = [f1]*self.num_surfs
        source_terms = []
        
        for i in range(self.num_surfs):
            source_terms += [inner(self.loads[i], self.splines[i].rationalize(
                self.spline_test_funcs[i]))*self.h_th[i]*self.splines[i].dx]
        
        self.source_terms = source_terms

    def dRdu(self):
        if self.print_info:
            if MPI.rank(self.comm) == 0:
                print("--- Computing dRdu ...")
        residuals = self.SVK_residuals()
        self.problem.set_residuals(residuals)
        dRtdut_FE, Rt_FE = self.problem.assemble_nonmatching()
        dRtdut_IGA, _ = self.problem.extract_nonmatching_system(
                        Rt_FE, dRtdut_FE)

        if MPI.size(self.comm) == 1:
            dRtdut_IGA.convert('seqaij')
        else:
            dRtdut_IGA = create_aijmat_from_nestmat(dRtdut_IGA, 
                         self.problem.A_list, comm=self.comm)

        return dRtdut_IGA

    def dRdt(self):
        if self.print_info:
            if MPI.rank(self.comm) == 0:
                print("--- Computing dRdt ...")
        residuals = self.residuals
        self.dRtdh_ths_list = []
        for i in range(self.num_splines):
            self.dRtdh_ths_list += [[],]
            for j in range(self.num_splines):
                # TODO: Figure out whether the couplings (i != j) 
                # are sensitive w.r.t. thickness
                if i == j:
                    dRdh_th = derivative(residuals[i], self.h_th[i])
                    dRdh_th_mat_FE = m2p(assemble(dRdh_th))
                    dRdh_th_mat_IGA = m2p(self.splines[i].M).\
                                      transposeMatMult(dRdh_th_mat_FE)
                else:
                    dRdh_th_mat_IGA = None
                # if self.problem.A_list[i][j] is not None:
                #     dRdh_th = derivative(residuals[i], self.h_th[j])
                #     dRdh_th_mat_FE = m2p(assemble(dRdh_th))
                #     dRdh_th_mat_IGA = m2p(self.splines[i].M).\
                #                       transposeMatMult(dRdh_th_mat_FE)
                # else:
                #     dRdh_th_mat_IGA = None
                self.dRtdh_ths_list[i] += [dRdh_th_mat_IGA,]
        dRtdh_ths = create_nest_PETScMat(self.dRtdh_ths_list, comm=self.comm)

        if MPI.size(self.comm) == 1:
            dRtdh_ths.convert('seqaij')
        else:
            dRtdh_ths = create_aijmat_from_nestmat(dRtdh_ths, 
                        self.dRtdh_ths_list, comm=self.comm)

        return dRtdh_ths

    def update_SVK_residuals(self):
        if self.print_info:
            if MPI.rank(self.comm) == 0:
                print("--- Updating SVK residuals ...")
        residuals = []
        for i in range(self.num_surfs):
            residuals += [SVK_residual(self.splines[i], self.spline_funcs[i], 
                self.spline_test_funcs[i], self.E, self.nu, self.h_th[i], self.source_terms[i])]
        
        self.residuals = residuals
    
    @staticmethod
    def concatenate_list_into_vec(array_list):
        return np.concatenate(array_list)



class GeometryWithLinearTransformations():
    def __init__(self, preprocessor):
        """
        This class defines a baseline geometry from the given IGS file 
        and allows for its direct manipulation with linear transformations.
        The transformations are applied directly to the control points when 
        creating tIGAr `ExtractedSpline` objects.
        """
        self.preprocessor = preprocessor
        self.baseline_surfs = self.create_geometry()
        self.rot_origin = self.compute_originpoint_phys_space()
    
    def create_geometry(self, trans_mats=None, rot_origin=None):
        num_surfs = len(self.preprocessor.BSpline_surfs_refine)
        splines = []
        for i in range(num_surfs):
            if i in [0, 1]:
                # Apply clamped BC to surfaces near root
                spline = OCCBSpline2tIGArSpline(
                        self.preprocessor.BSpline_surfs_refine[i], 
                        setBCs=clampedBC, side=0, direction=0, index=i, 
                        trans_mats=trans_mats, rot_origin=rot_origin)
                splines += [spline,]
            else:
                spline = OCCBSpline2tIGArSpline(
                        self.preprocessor.BSpline_surfs_refine[i], 
                        index=i, trans_mats=trans_mats, rot_origin=rot_origin)
                splines += [spline,]
        return splines
    
    def compute_originpoint_phys_space(self):
        """
        This method computes the center point at the root of the wing,
        which will function as origin for the linear transformations that
        are applied to the geometry.
        The center point has 4-dimensions; the last dimension 
        (which corresponds to the NURBS weight mapping) is always equal to zero.
        """
        # define parametric coordinate at center of root chord
        xi_center = np.array([0.5, 0])
        surf_1 = self.baseline_surfs[0]
        surf_2 = self.baseline_surfs[1]
        
        coords_1 = np.zeros((4,))
        coords_2 = np.zeros((4,))
        w_1 = eval_func(surf_1.mesh, surf_1.cpFuncs[3], xi_center)
        w_2 = eval_func(surf_2.mesh, surf_2.cpFuncs[3], xi_center)

        # compute physical coordinates of center root point of lower and upper wing surface
        for i in range(3):
            coords_1[i] = surf_1.F[i](xi_center)/w_1
            coords_2[i] = surf_2.F[i](xi_center)/w_2
        
        # average coordinates of upper and lower surface center root points
        coords_avg = 0.5*np.add(coords_1, coords_2)

        return coords_avg
        
    def create_trans_geometry(self, trans_mats):
        """
        This function takes a list of linear transformation matrices `trans_mats`
        and pipes it to other functions to output a transformed geometry
        based on the OCC data from self.preprocessor
        """
        return self.create_geometry(trans_mats=trans_mats, rot_origin=self.rot_origin)


if __name__ == '__main__':
    E = Constant(71.7e9)  # Young's modulus, Pa
    nu = Constant(0.33)  # Poisson's ratio
    rho = Constant(2.81e3)  # Material density, kg/m^3
    n_load = Constant(2.5)  # Load factor
    h_th = Constant(20.0e-3)  # Thickness of surfaces, m

    p = 3  # spline order
    filename_igs = "eVTOL_wing_structure.igs"

    shell_sim = ShellSim(p, E, nu, rho, n_load, 
                    filename_igs, comm=worldcomm)