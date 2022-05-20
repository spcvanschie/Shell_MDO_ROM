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
        self.options.declare('tip_disp', default=.1)
        self.options.declare('max_vonmises_stress', default=80e7)

    def setup(self):
        self.shell_sim = self.options['shell_sim']

        # define optimization inputs
        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('h_th', shape=(self.shell_sim.num_surfs,),
                              val=np.ones(self.shell_sim.num_surfs)*0.005)
        self.add_subsystem('inputs_comp', inputs_comp)

        # define solution state computation class (wraps both FOM and ROM)
        state_comp = StateComp(spline_sim=self.spline_sim)
        self.add_subsystem('state_comp', state_comp)

        # define objective function computation class
        # NOTE: Initial implementation is functional (no errors)
        objective_comp = ObjectiveComp(spline_sim=self.spline_sim)
        self.add_subsystem('objective_comp', objective_comp)

        # define constraint computation class
        # NOTE: Initial implementation is functional (no errors)
        constraints_comp = ConstraintsComp(spline_sim=self.spline_sim)
        self.add_subsystem('constraints_comp', constraints_comp)



        self.connect('inputs_comp.h_th', 'states_comp.h_th')
        self.connect('inputs_comp.h_th', 'constraints_comp.h_th')
        self.connect('inputs_comp.h_th', 'objective_comp.h_th')
        self.connect('states_comp.displacements', 
                     'constraints_comp.displacements')

        # define design variables (shell thicknesses)
        self.add_design_var('inputs_comp.h_th', lower=1e-3, upper=5e-2)
        
        # define objective function (weight of wing structure)
        self.add_objective('objective_comp.weight')

        # constraints: max tip displacement and max von Mises stress
        self.add_constraint('constraints_comp.tip_disp', upper=0.1)
        self.add_constraint('constraints_comp.max_von_Mises_stress', lower=-70e9, upper=70e9)


save_disp = True
save_stress = True

# Define parameters
# Material properties of Al-7075 T6 (common aerospace alloy)
E = Constant(71.7e9)  # Young's modulus, Pa
nu = Constant(0.33)  # Poisson's ratio
rho = Constant(2.81e3)  # Material density, kg/m^3
n_load = Constant(2.5)  # Load factor
h_th = Constant(20.0e-3)  # Thickness of surfaces, m

p = 3  # spline order
filename_igs = "eVTOL_wing_structure.igs"

shell_sim = ShellSim(p, E, nu, rho, n_load, 
                 filename_igs, comm=worldcomm)

h_th_list = 4*[h_th]+(num_surfs-4)*[h_th]

# Starting point of optimization loop

# Create non-matching problem
problem = NonMatchingCoupling(splines, E, h_th_list, nu, comm=worldcomm)
problem.create_mortar_meshes(preprocessor.mortar_nels)
problem.create_mortar_funcs('CG',1)
problem.create_mortar_funcs_derivative('CG',1)

if mpirank == 0:
    print("Setting up mortar meshes...")

problem.mortar_meshes_setup(preprocessor.mapping_list, 
                            preprocessor.intersections_para_coords, 
                            penalty_coefficient)

# Compute magnitude of weight load
Body_weight = 1500
wing_volume = 0
for i in range(num_surfs):
    wing_volume += assemble(h_th_list[i]*splines[i].dx)

wing_weight = wing_volume*rho

if mpirank == 0:
    print("Wing mass: {} kg".format(wing_weight))

# Weight is a constant volumetric load in negative z-direction
f1 = as_vector([Constant(0.0), Constant(0.0), n_load*Constant(9.81)*(Body_weight + wing_weight)/wing_volume])

# Distributed downward load
loads = [f1]*num_surfs
source_terms = []
residuals = []
for i in range(num_surfs):
    source_terms += [inner(loads[i], problem.splines[i].rationalize(
        problem.spline_test_funcs[i]))*h_th_list[i]*problem.splines[i].dx]
    residuals += [SVK_residual(problem.splines[i], problem.spline_funcs[i], 
        problem.spline_test_funcs[i], E, nu, h_th_list[i], source_terms[i])]
problem.set_residuals(residuals)

if mpirank == 0:
    print("Solving linear non-matching problem...")

problem.solve_linear_nonmatching_problem()

# mat_A = problem.A
# Ai, Aj, Av = mat_A.getValuesCSR()
# mat_A_dense = mat_A.convert("dense")
# result = mat_A_dense.getDenseArray()
# # mat_sparsity = np.zeros_like(mat_A)
# # mat_sparsity[A != 0.] = 1.
# fig, ax = plt.subplots()
# ax.spy(result)
# plt.show()

# print out vertical displacement at the tip of trailing edge
right_srf_ind = 3
xi = array([1, 1])
z_disp_hom = eval_func(problem.splines[right_srf_ind].mesh, 
                       problem.spline_funcs[right_srf_ind][2], xi)
w = eval_func(problem.splines[right_srf_ind].mesh, 
              problem.splines[right_srf_ind].cpFuncs[3], xi)
QoI = z_disp_hom/w

if mpirank == 0:
    print("Trailing edge tip vertical displacement: {:10.8f}.\n".format(QoI))

# Compute von Mises stress
if mpirank == 0:
    print("Computing von Mises stresses...")

von_Mises_tops = []
for i in range(problem.num_splines):
    spline_stress = ShellStressSVK(problem.splines[i], 
                                   problem.spline_funcs[i],
                                   E, nu, h_th_list[i], linearize=True,) 
                                   # G_det_min=5e-2)
    # von Mises stresses on top surfaces
    von_Mises_top = spline_stress.vonMisesStress(h_th_list[i]/2)
    von_Mises_top_proj = problem.splines[i].projectScalarOntoLinears(
                         von_Mises_top, lumpMass=True)
    von_Mises_tops += [von_Mises_top_proj]

if mpirank == 0:
    print("Saving results...")

if save_disp:
    for i in range(problem.num_splines):
        save_results(splines[i], problem.spline_funcs[i], i, 
                     save_path=SAVE_PATH, folder="results/", 
                     save_cpfuncs=True, comm=worldcomm)
if save_stress:
    for i in range(problem.num_splines):
        von_Mises_tops[i].rename("von_Mises_top_"+str(i), 
                                 "von_Mises_top_"+str(i))
        File(SAVE_PATH+"results/von_Mises_top_"+str(i)+".pvd") \
            << von_Mises_tops[i]