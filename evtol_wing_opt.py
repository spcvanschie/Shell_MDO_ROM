"""
The required eVTOL geometry can be downloaded from:

    https://drive.google.com/file/d/1xpY8ACQlodmwkUZsiEQvTZPcUu-uezgi/view?usp=sharing

and extracted using the command "tar -xvzf eVTOL_wing_structure.tgz".
"""
from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *
from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.occ_utils import *
from LinearTransformations import *

import matplotlib.pyplot as plt
import openmdao.api as om 

from geom_utils import *
from state_comp import *
from objective_comp import *
from constraints_comp import *

SAVE_PATH = "./"

def plot_controlpoints(surfs, surfs_mod):

    fig = plt.figure()

    # subplot with original control mesh
    ax1 = fig.add_subplot(121,projection='3d')
    for i in range(len(surfs)):
        x_vec = surfs[i].cpFuncs[0].vector().get_local()
        y_vec = surfs[i].cpFuncs[1].vector().get_local()
        z_vec = surfs[i].cpFuncs[2].vector().get_local()
        ax1.scatter(x_vec, y_vec, z_vec, cmap='viridis', linewidth=0.5)
        # for j in range(phys_points.shape[1]):
        #     ax1.plot3D(phys_points[:, j, 0], phys_points[:, j, 1], phys_points[:, j, 2])
        # for j in range(phys_points.shape[0]):
        #     ax1.plot3D(phys_points[j, :, 0], phys_points[j, :, 1], phys_points[j, :, 2])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    # ax1.set_xlim([4.75-2.5, 4.75+2.5])
    # ax1.set_ylim([0.5, 5.5])
    # ax1.set_zlim([3.3-2.5, 3.3+2.5])

    # subplot with modified control mesh
    ax2 = fig.add_subplot(122,projection='3d')
    for i in range(len(surfs_mod)):
        x_vec = surfs_mod[i].cpFuncs[0].vector().get_local()
        y_vec = surfs_mod[i].cpFuncs[1].vector().get_local()
        z_vec = surfs_mod[i].cpFuncs[2].vector().get_local()
        ax2.scatter(x_vec, y_vec, z_vec, cmap='viridis', linewidth=0.5)
        # ax2.plot3D(phys_points[:, :, 0], phys_points[:, :, 1], phys_points[:, :, 2])
        # ax2.plot3D(phys_points[:, :, 0].T, phys_points[:, :, 1].T, phys_points[:, :, 2].T)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    # ax2.set_xlim([4.75-2.5, 4.75+2.5])
    # ax2.set_ylim([0.5, 5.5])
    # ax2.set_zlim([3.3-2.5, 3.3+2.5])

    plt.tight_layout()
    plt.show()


class ShellGroup(om.Group):
    def initialize(self):
        self.options.declare('shell_sim')
        # self.options.declare('tip_disp')
        self.options.declare('max_vonmises_stress', default=4.6e8)

    def setup(self):
        self.shell_sim = self.options['shell_sim']
        self.max_vonmises_stress = self.options['max_vonmises_stress']

        # define optimization inputs
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('h_th', shape=(self.shell_sim.num_surfs,),
                              val=np.ones(self.shell_sim.num_surfs)*0.01)
        self.add_subsystem('inputs_comp', inputs_comp)

        # define solution state computation class (wraps both FOM and ROM)
        state_comp = StateComp(shell_sim=self.shell_sim)
        self.add_subsystem('state_comp', state_comp)

        # define objective function computation class
        objective_comp = ObjectiveComp(shell_sim=self.shell_sim)
        self.add_subsystem('objective_comp', objective_comp)

        # define constraint computation class
        constraints_comp = ConstraintsComp(shell_sim=self.shell_sim, max_vonmises_stress=self.max_vonmises_stress)
        self.add_subsystem('constraints_comp', constraints_comp)

        self.connect('inputs_comp.h_th', 'state_comp.h_th')
        self.connect('inputs_comp.h_th', 'constraints_comp.h_th')
        self.connect('inputs_comp.h_th', 'objective_comp.h_th')
        self.connect('state_comp.displacements', 
                     'constraints_comp.displacements')

        # define design variables (shell thicknesses)
        self.add_design_var('inputs_comp.h_th', lower=2e-3, upper=5e-2)
        
        # define objective function (weight of wing structure)
        self.add_objective('objective_comp.weight')

        # constraints: max tip displacement and max von Mises stress
        self.add_constraint('constraints_comp.tip_disp', upper=0.1)

        # NOTE: Temporarily deactivated the maximum Von Mises stress bound
        # self.add_constraint('constraints_comp.max_von_Mises_stress', upper=1.)



if __name__ == "__main__":
    save_disp = True
    save_stress = True

    # Define parameters
    # Material properties of Al-7075 T6 (common aerospace alloy)
    E = Constant(71.7e9)  # Young's modulus, Pa
    nu = Constant(0.33)  # Poisson's ratio
    rho = Constant(2.81e3)  # Material density, kg/m^3
    n_load = Constant(2.5)  # Load factor
    Von_Mises_max = Constant(4.6e8)

    p = 3  # spline order
    filename_igs = "eVTOL_wing_structure.igs"

    shell_sim = ShellSim(p, E, nu, rho, n_load, 
                    filename_igs, comm=worldcomm)

    # h_th_list = 4*[h_th]+(shell_sim.num_surfs-4)*[h_th]

    model = ShellGroup(shell_sim=shell_sim, max_vonmises_stress=Von_Mises_max)
    prob = om.Problem(model=model)

    # SNOPT optimizer
    # prob.driver = om.pyOptSparseDriver()
    # prob.driver.options['optimizer'] = 'SNOPT'
    # prob.driver.opt_settings['Major feasibility tolerance'] = 1e-9
    # prob.driver.opt_settings['Major optimality tolerance'] = 1e-9
    # prob.driver.options['disp'] = True
    # prob.driver.options['maxiter'] = 10
    # prob.driver.options['debug_print'] = ['objs']
    # prob.driver.options['print_results'] = True


    # SLSQP optimizer
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-9
    prob.driver.options['disp'] = True
    prob.driver.options['maxiter'] = 10


    # COBYLA optimizer (gradient-free)
    # prob.driver = om.ScipyOptimizeDriver()
    # prob.driver.options['optimizer'] = 'COBYLA'
    # prob.driver.options['tol'] = 1e-9
    # prob.driver.options['disp'] = True
    # prob.driver.opt_settings['rhobeg'] = 0.008
    # prob.driver.opt_settings['catol'] = 0.
    # # prob.driver.options['debug_print'] = ['objs']
    # prob.driver.options['maxiter'] = 25


    prob.setup()
    prob.run_driver()
    if mpirank == 0:
        print("Optimized thickness")
        print(prob['inputs_comp.h_th'],'\n')
        print(prob['objective_comp.weight'],'\n')
        
    # o_comp = ObjectiveComp(shell_sim=shell_sim)
    # o_comp.setup()
    # o_comp.compute()

    # mat_A = problem.A
    # Ai, Aj, Av = mat_A.getValuesCSR()
    # mat_A_dense = mat_A.convert("dense")
    # result = mat_A_dense.getDenseArray()
    # # mat_sparsity = np.zeros_like(mat_A)
    # # mat_sparsity[A != 0.] = 1.
    # fig, ax = plt.subplots()
    # ax.spy(result)
    # plt.show()


    # Compute von Mises stress
    if mpirank == 0:
        print("Computing von Mises stresses...")

    von_Mises_tops = []
    for i in range(shell_sim.num_surfs):
        spline_stress = ShellStressSVK(shell_sim.problem.splines[i], 
                                    shell_sim.problem.spline_funcs[i],
                                    shell_sim.E, shell_sim.nu, shell_sim.h_th[i], linearize=True,) 
                                    # G_det_min=5e-2)
        # von Mises stresses on top surfaces
        von_Mises_top = spline_stress.vonMisesStress(shell_sim.h_th[i]/2)
        von_Mises_top_proj = shell_sim.problem.splines[i].projectScalarOntoLinears(
                            von_Mises_top, lumpMass=True)
        von_Mises_tops += [von_Mises_top_proj]

    if mpirank == 0:
        print("Saving results...")

    if save_disp:
        for i in range(shell_sim.num_surfs):
            save_results(shell_sim.splines[i], shell_sim.problem.spline_funcs[i], i, 
                        save_path=SAVE_PATH, folder="results/", 
                        save_cpfuncs=True, comm=worldcomm)
    if save_stress:
        for i in range(shell_sim.num_surfs):
            von_Mises_tops[i].rename("von_Mises_top_"+str(i), 
                                    "von_Mises_top_"+str(i))
            File(SAVE_PATH+"results/von_Mises_top_"+str(i)+".pvd") \
                << von_Mises_tops[i]