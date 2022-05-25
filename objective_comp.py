import numpy as np
import openmdao.api as om

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *
from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.occ_utils import *

from shell_sim import *
from geom_utils import *

class ObjectiveComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('shell_sim')
    
    def setup(self):
        self.print_info = True

        self.shell_sim = self.options['shell_sim']
        self.add_input('h_th', shape=(self.shell_sim.num_surfs,), 
                       val=np.ones(self.shell_sim.num_surfs)*0.01)
        self.add_output('weight', shape=1)
        self.declare_partials('weight', 'h_th')
       
    def compute(self, inputs, outputs):
        self.shell_sim.update_h_th(inputs['h_th'])

        shell_volumes = self.compute_shell_volumes(inputs['h_th'])
        outputs['weight']= np.sum(shell_volumes)*self.shell_sim.rho

        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Evaluating design weight ...")
                print("Weight: {}".format(outputs['weight']))

    def compute_shell_volumes(self, h_th):
        shell_volumes = [0.]*self.shell_sim.num_surfs
        for i in range(self.shell_sim.num_surfs):
            shell_volumes[i] = assemble(h_th[i]*self.shell_sim.splines[i].dx)

        return np.array(shell_volumes)

    def compute_partials(self, inputs, partials):
        self.shell_sim.update_h_th(inputs['h_th'])

        partials['weight', 'h_th'] = self.shell_sim.rho*np.divide(self.compute_shell_volumes(self.shell_sim.h_th), inputs['h_th'])


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
    comp = ObjectiveComp(shell_sim=shell_sim)

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)