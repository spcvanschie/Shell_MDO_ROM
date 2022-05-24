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

SAVE_PATH = "./"

class NURBSControlMeshWithTransformations4OCC(AbstractControlMesh):
    """
    This class represents a control mesh with NURBS geometry.
    It is an expanded version of PENGoLINS' `NURBSControlMesh4OCC` class;
    the added functionality allows for the use of linear transformation matrices to move control points
    """
    def __init__(self,occ_bs_surf,useRect=USE_RECT_ELEM_DEFAULT,
                 overRefine=0, trans_mats=None, rot_origin=None):
        """
        Generates a NURBS control mesh from PythonOCC B-spline surface 
        input data .
        The optional parameter ``overRefine``
        indicates how many levels of refinement to apply beyond what is
        needed to represent the spline functions; choosing a value greater
        than the default of zero may be useful for
        integrating functions with fine-scale features.
        overRefine > 0 only works for useRect=False.
        """
        if trans_mats is None:
            trans_mats = [np.eye(4)]
        if rot_origin is None:
            rot_origin = np.zeros((4,))

        bs_data = BSplineSurfaceData(occ_bs_surf)

        # create a BSpline scalar space given the knot vector(s)
        self.scalarSpline = BSpline(bs_data.degree,bs_data.knots,
                                    useRect,overRefine)
        
        # get the control net; already in homogeneous form
        nvar = len(bs_data.degree)
        if(nvar==2):
            M = bs_data.control.shape[0]
            N = bs_data.control.shape[1]
            dim = bs_data.control.shape[2]
            self.bnet = zeros((M*N,dim))
            for j in range(0,N):
                for i in range(0,M):
                    dist_origin = np.subtract(bs_data.control[i,j,:], rot_origin)
                    for trans_mat in trans_mats:
                        # apply transformation matrices in order
                        dist_origin = dist_origin@trans_mat

                    self.bnet[ij2dof(i,j,M),:]\
                        = np.add(dist_origin, rot_origin)
        else:
            raise ValueError("Linear transformations not implemented for {}-dimensional control meshes".format(nvar))
            
    def getScalarSpline(self):
        return self.scalarSpline

    def getHomogeneousCoordinate(self,node,direction):
        return self.bnet[node,direction]

    def getNsd(self):
        return self.bnet.shape[1]-1

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


def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=2)
        spline_generator.addZeroDofs(field, side_dofs)

def OCCBSpline2tIGArSpline(surface, num_field=3, quad_deg_const=4, 
                        setBCs=None, side=0, direction=0, index=0, 
                        trans_mats=None, rot_origin=None):
    """
    Generate ExtractedBSpline from OCC B-spline surface.
    """
    quad_deg = surface.UDegree()*quad_deg_const
    DIR = SAVE_PATH+"spline_data/extraction_"+str(index)+"_init"
    # spline = ExtractedSpline(DIR, quad_deg)
    spline_mesh = NURBSControlMeshWithTransformations4OCC(surface, useRect=False, 
                                                          trans_mats=trans_mats, rot_origin=rot_origin)
    spline_generator = EqualOrderSpline(worldcomm, num_field, spline_mesh)
    if setBCs is not None:
        setBCs(spline_generator, side, direction)
    spline_generator.writeExtraction(DIR)
    spline = ExtractedSpline(spline_generator, quad_deg)
    return spline

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

save_disp = True
save_stress = True
# Define parameters
# Scale down the geometry using ``geom_scale``to make the length 
# of the wing in the span-wise direction around 5 m.
geom_scale = 2.54e-5  # Convert current length unit to m

# Material properties of Al-7075 T6 (common aerospace alloy)
E = Constant(71.7e9)  # Young's modulus, Pa
nu = Constant(0.33)  # Poisson's ratio
rho = Constant(2.81e3)  # Material density, kg/m^3

n_load = Constant(2.5)  # Load factor

h_th_min = 2e-3 #lower bound on shell thickness
h_th_max = 5e-2 #upper bound on shell thickness
h_th = Constant(np.random.uniform(h_th_min,h_th_max))  # Thickness of surfaces, m

print("Selected shell thickness: {} m".format(h_th))

p = 3  # spline order
penalty_coefficient = 1.0e3

print("Importing geometry...")
filename_igs = "eVTOL_wing_structure.igs"
igs_shapes = read_igs_file(filename_igs, as_compound=False)
evtol_surfaces = [topoface2surface(face, BSpline=True) 
                for face in igs_shapes]

# Outer skin indices: list(range(12, 18))
# Spars indices: [78, 92, 79]
# Ribs indices: list(range(80, 92))
wing_indices = list(range(12, 18)) + [78, 92, 79]  + list(range(80, 92))
wing_surfaces = [evtol_surfaces[i] for i in wing_indices]
num_surfs = len(wing_surfaces)
if mpirank == 0:
    print("Number of surfaces:", num_surfs)

num_pts_eval = [16]*num_surfs
u_insert_list = [8]*num_surfs
v_insert_list = [8]*num_surfs
ref_level_list = [1]*num_surfs
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
preprocessor = OCCPreprocessing(wing_surfaces, reparametrize=True, 
                                refine=True)
preprocessor.reparametrize_BSpline_surfaces(num_pts_eval, num_pts_eval,
                                            geom_scale=geom_scale,
                                            remove_dense_knots=True,
                                            rtol=1e-4)
preprocessor.refine_BSpline_surfaces(p, p, u_num_insert, v_num_insert, 
                                    correct_element_shape=True)
print("Computing intersections...")
preprocessor.compute_intersections(mortar_refine=2)

if mpirank == 0:
    print("Total DoFs:", preprocessor.total_DoFs)
    print("Number of intersections:", preprocessor.num_intersections_all)

# # Display B-spline surfaces and intersections using 
# # PythonOCC build-in 3D viewer.
# display, start_display, add_menu, add_function_to_menu = init_display()
# preprocessor.display_surfaces(display, save_fig=True)
# preprocessor.display_intersections(display, save_fig=True)

if mpirank == 0:
    print("Creating splines...")
# Create tIGAr extracted spline instances to define baseline geometry
Geometry = GeometryWithLinearTransformations(preprocessor)
splines = Geometry.baseline_surfs

#generate list of thickness between max and min values
h_th_list = []
for i in range(num_surfs):
    h_th_list.append(Constant(np.random.uniform(h_th_min,h_th_max)))

#h_th_list = num_surfs*[h_th]

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

iter = 6
#save snapshots
filename = './Shell_MDO_ROM/snapshots/disp_' + str(iter) + '.npy'
np.save(filename,problem.u.getArray())
#filename = './Shell_MDO_ROM/snapshots/h_th_list_' + str(iter) + '.npy'
#np.save(filename,h_th_list)
# filename = './Shell_MDO_ROM/snapshots/A_' + str(1) + '.npy'
# np.save(filename,problem.A.getArray())
# filename = './Shell_MDO_ROM/snapshots/b_' + str(1) + '.npy'
# np.save(filename,problem.b.getArray())


####POST PROCESSING BELOW


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

'''
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

'''