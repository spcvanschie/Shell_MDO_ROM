import numpy as np
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
        self.options.declare('debug', default=False)

    def setup(self):
        self.shell_sim = self.options['shell_sim']

        self.add_input('displacements', shape=self.spline_sim.iga_dof)
        self.add_input('h_th', shape=(self.shell_sim.num_surfs,), 
                       val=np.ones(self.shell_sim.num_surfs)*0.01)
        self.add_output('tip_disp')
        self.add_output('max_von_Mises_stress')

    def compute(self, inputs, outputs):
        self.shell_sim.update_displacements(inputs['displacements'])
        self.shell_sim.update_h_th(inputs['h_th'])

        outputs['tip_disp'] = self.compute_tip_disp(inputs)
        outputs['max_von_Mises_stress'] = self.compute_von_mises_stress(inputs)

        # if mpirank == 0:
        #     print("Trailing edge tip vertical displacement: {:10.8f}.\n".format(QoI))
    
    def compute_tip_disp(self, inputs):
        # compute vertical displacement at the tip of trailing edge
        right_srf_ind = 3  # currently hardcoded index of one of the wing surface patches at the tip
        xi = array([1, 1])

        # construct spline function on given spline from vector input
        spline = self.shell_sim.splines[right_srf_ind]
        spline_func = Function(spline.V)
        u_iga_sub = self.u_iga_nest.getNestSubVecs()
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

        von_Mises_tops = []
        max_stress = 0.
        u_iga_sub = self.u_iga_nest.getNestSubVecs()
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
            von_Mises_tops += [von_Mises_top_proj]
            max_stress_i = np.max(von_Mises_top_proj.vector().get_local())
            max_stress = max(max_stress, max_stress_i)
        return max_stress

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
    comp = ConstraintsComp(shell_sim=shell_sim, debug=True)

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    # print('check_partials:')
    # prob.check_partials(compact_print=True)