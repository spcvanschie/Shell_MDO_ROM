import numpy as np
import scipy as sp
import openmdao.api as om

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *
from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.occ_utils import *

from shell_sim import *
from geom_utils import *
from opt_utils import *

class ConstraintsComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('shell_sim')
        self.options.declare('max_vonmises_stress')

    def setup(self):
        self.print_info = True
        self.shell_sim = self.options['shell_sim']
        self.max_stress = self.options['max_vonmises_stress']

        right_srf_ind = 3  # currently hardcoded index of one of the wing surface patches at the tip
        xi = array([1, 1])
        self.tip_disp_basis_funcs = ConstraintsComp.sample_basis_funcs(self.shell_sim.splines[right_srf_ind], xi)

        self.dGdu = self.comp_dGdu()

        self.add_input('displacements', shape=self.shell_sim.iga_dof)
        self.add_input('h_th', shape=(self.shell_sim.num_surfs,), 
                       val=np.ones(self.shell_sim.num_surfs)*0.01)

        self.add_output('tip_disp', shape=1)
        self.add_output('max_von_Mises_stress', shape=1)
        self.add_output('h_th_range', shape=(self.shell_sim.num_surfs,))

        self.declare_partials('tip_disp', 'h_th')
        self.declare_partials('tip_disp', 'displacements', val=self.dGdu)
        # set nonzero positive gradients to stabilize the optimizer; compute the actual gradients later
        # self.declare_partials('max_von_Mises_stress', 'h_th', val=-np.ones(self.shell_sim.num_surfs))
        # self.declare_partials('max_von_Mises_stress', 'displacements', val=np.ones(self.shell_sim.iga_dof))

        # self.dresdu = self.shell_sim.dRdu()
        # self.dresdt = self.shell_sim.dRdt()

    # TODO: Define separate setup_partials method to declare partial derivatives?


    def compute(self, inputs, outputs):
        self.shell_sim.update_displacements(inputs['displacements'])
        self.shell_sim.update_h_th(inputs['h_th'])

        outputs['tip_disp'] = self.compute_tip_disp(inputs)
        # output the Von Mises stress normalized with the max Von Mises stress to bring it to O(1) for numerical accuracy purposes
        max_vm_stress, _, _ = self.compute_von_mises_stress(inputs)
        outputs['max_von_Mises_stress']= max_vm_stress/self.max_stress
        outputs['h_th_range'] = inputs['h_th']

        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Evaluating constraints ...")
                print("Tip displacement: {}".format(outputs['tip_disp']))
                # print("Tip displacement from derivatives: {}".format(self.dGdu@inputs['displacements']))
                # print("Max Von Mises stress: {}".format(outputs['max_von_Mises_stress']))
    
    def compute_partials(self, inputs, partials):
        self.shell_sim.update_displacements(inputs['displacements'])
        self.shell_sim.update_h_th(inputs['h_th'])
        self.shell_sim.update_external_loads()
        # self.shell_sim.update_SVK_residuals()

        self.dresdu = self.shell_sim.dRdu()
        self.dresdt = self.shell_sim.dRdt()

        # we need to invert self.dresdu, to do this we first convert it to a numpy array
        dresdu_np = sp.sparse.csr_matrix((self.dresdu.getValuesCSR()[2], self.dresdu.getValuesCSR()[1], self.dresdu.getValuesCSR()[0])).toarray()

        dresdt_dense = self.dresdt.convert("dense")
        dresdt_np = dresdt_dense.getDenseArray()
        dresdu_np_inv = np.linalg.inv(dresdu_np)


        partials['tip_disp', 'h_th'] = -self.dGdu@dresdu_np_inv@dresdt_np
        print("max partial deriv of tip_disp wrt h_th: {}".format(np.max(np.abs(partials['tip_disp', 'h_th']))))
        # partials['tip_disp', 'displacements'] = self.dGdu
        

    def compute_tip_disp(self, inputs):
        # compute vertical displacement at the tip of trailing edge
        self.shell_sim.update_displacements(inputs['displacements'])
        self.shell_sim.update_h_th(inputs['h_th'])

        right_srf_ind = 3  # currently hardcoded index of one of the wing surface patches at the tip
        xi = array([1, 1])
        # construct spline function on given spline from vector input
        spline = self.shell_sim.splines[right_srf_ind]
        spline_func = Function(spline.V)
        u_iga_sub = self.shell_sim.u_iga_nest.getNestSubVecs()
        disp_FE = IGA2FE(spline, u_iga_sub[right_srf_ind])
        spline_func.vector().set_local(disp_FE)

        z_disp_hom = eval_func(self.shell_sim.splines[right_srf_ind].mesh, 
                            spline_func, xi)
        w = eval_func(self.shell_sim.splines[right_srf_ind].mesh, 
                    self.shell_sim.splines[right_srf_ind].cpFuncs[3], xi)

        tip_disp = z_disp_hom/w
        return tip_disp[2]

    def compute_von_mises_stress(self, inputs):
        # Compute von Mises stress
        # if mpirank == 0:
        #     print("Computing von Mises stresses...")
        self.shell_sim.update_displacements(inputs['displacements'])
        self.shell_sim.update_h_th(inputs['h_th'])
        
        # von_Mises_tops = []
        max_stress = 0.
        u_iga_sub = self.shell_sim.u_iga_nest.getNestSubVecs()

        shell_max_idx = 0
        fe_bf_max_idx = 0
        for i in range(self.shell_sim.num_surfs):
            displacement = Function(self.shell_sim.splines[i].V)
            disp_FE = IGA2FE(self.shell_sim.splines[i], u_iga_sub[i])

            displacement.vector().set_local(disp_FE)

            spline_stress = ShellStressSVK(self.shell_sim.splines[i], 
                                        displacement,
                                        self.shell_sim.E, self.shell_sim.nu, self.shell_sim.h_th[i], linearize=True,) 
                                        # G_det_min=5e-2)
            # von Mises stresses on top surfaces
            von_Mises_top = spline_stress.vonMisesStress(Constant(self.shell_sim.h_th[i]/2))  # ShNAPr requires a UFL object input
            von_Mises_top_proj = self.shell_sim.splines[i].projectScalarOntoLinears(
                                von_Mises_top, lumpMass=True)
            # von_Mises_tops += [von_Mises_top_proj]
            abs_stress = np.abs(von_Mises_top_proj.vector().get_local())
            max_stress_i = np.max(abs_stress)
            if max_stress_i > max_stress:
                max_stress = max_stress_i
                shell_max_idx = i
                fe_bf_max_idx = np.argmax(abs_stress)

        return max_stress, shell_max_idx, fe_bf_max_idx

    def sample_basis_funcs(spline, x):
        """
        Function that computes the basis function values of `spline_func` at point `x` in the parameter space
        """
        spline_func = Function(spline.V)
        vec_in = np.zeros((len(spline_func.vector().get_local())))
        vec_out = np.zeros((len(spline_func.vector().get_local())))
        for i in range(vec_in.shape[0]):
            vec_in[i] = 1.
            spline_func.vector().set_local(vec_in)
            vec_out[i] = spline_func(x)[2]
            vec_in[i] = 0.
        return vec_out

    def comp_dGdu(self):
        right_srf_ind = 3  # currently hardcoded index of one of the wing surface patches at the tip
        xi = array([1, 1])
        self.tip_disp_basis_funcs = ConstraintsComp.sample_basis_funcs(self.shell_sim.splines[right_srf_ind], xi)

        # self.dGdu contains d u_tip / du for the patch with index `right_srf_ind`. 
        # The derivative is equal to zero for all other patches.
        dGdu_patch = AT_x(self.shell_sim.splines[right_srf_ind].M, 
                                                      PETSc.Vec().createWithArray(self.tip_disp_basis_funcs))
        
        dGdu = np.zeros((self.shell_sim.iga_dof))
        dGdu[np.sum(self.shell_sim.iga_dofs[:right_srf_ind]):np.sum(self.shell_sim.iga_dofs[:right_srf_ind+1])] = dGdu_patch.array

        return dGdu




    # TODO: Add partial derivative computation here. 
    # We solve a linear system to find displacements, 
    # so from that system we should be able to extract the sensitivity 
    # of tip displacement w.r.t. the individual shell thicknesses

    # def compute_partials(self, inputs, partials)


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

    # h_th_list = 4*[h_th]+(shell_sim.num_surfs-4)*[h_th]

    prob = om.Problem()
    comp = ConstraintsComp(shell_sim=shell_sim, max_vonmises_stress=4.6e8)

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # prob.model.compute_partials(prob.model.inputs, prob.model.partials)
    print('check_partials:')
    prob.check_partials(compact_print=True)