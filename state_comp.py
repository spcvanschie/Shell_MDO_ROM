import numpy as np
import openmdao.api as om

from PENGoLINS.occ_preprocessing import *
from PENGoLINS.nonmatching_coupling import *
from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.occ_utils import *

from opt_utils import *
from shell_sim import *

class StateComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('shell_sim')
    

    def setup(self):
        self.shell_sim = self.options['shell_sim']
        self.print_info = True

        self.add_input('h_th', shape=(self.shell_sim.num_surfs,), 
                               val=np.ones(self.shell_sim.num_surfs)*0.01)
        self.add_output('displacements', shape=self.shell_sim.iga_dof)

        # self.declare_partials('displacements', 'h_th')
        self.declare_partials('displacements', 'h_th')
        self.declare_partials('displacements', 'displacements')


    def apply_nonlinear(self, inputs, outputs, residuals):
        """
        Computes the SVK nonlinear residual.
        """
        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Running apply_nonlinear ...")
        
        # update shell_sim properties based on inputs
        self.shell_sim.update_h_th(inputs['h_th'])
        self.shell_sim.update_displacements(outputs['displacements'])
        self.shell_sim.update_external_loads()
        self.shell_sim.update_SVK_residuals()
        
        # update nonmatching problem parameters based on updated inputs
        self.shell_sim.problem.set_residuals(self.shell_sim.residuals)
        dRtdut_FE, Rt_FE = self.shell_sim.problem.assemble_nonmatching()
        dRtdut_IGA, Rt_IGA = self.shell_sim.problem.extract_nonmatching_system(Rt_FE, dRtdut_FE)

        if mpirank == 0:
            print("Setting up mortar meshes...")

        self.shell_sim.problem.mortar_meshes_setup(self.shell_sim.preprocessor.mapping_list, 
                                    self.shell_sim.preprocessor.intersections_para_coords, 
                                    self.shell_sim.penalty_coefficient)
        
        residuals['displacements'] = get_petsc_vec_array(Rt_IGA, 
                                     self.shell_sim.comm)
        
        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Finished apply_nonlinear ...")


    def solve_nonlinear(self, inputs, outputs):
        """
        Solve displacements for SVK residual.
        """

        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Running solve_nonlinear ...")
                print("h_th: {}".format(inputs['h_th']))

        self.shell_sim.update_h_th(inputs['h_th'])
        self.shell_sim.update_displacements(outputs['displacements'])
        self.shell_sim.update_external_loads()
        self.shell_sim.update_SVK_residuals()
        

        self.shell_sim.problem.set_residuals(self.shell_sim.residuals)

        # this derivative computation is just here for debugging purposes
        self.shell_sim.dRdt()

        _, u_iga = self.shell_sim.problem.\
                   solve_nonlinear_nonmatching_problem(
                   max_it=30, zero_mortar_funcs=True, iga_dofs=True, 
                   rtol=1e-3)

        outputs['displacements'] = get_petsc_vec_array(u_iga, 
                                   self.shell_sim.comm)

        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Finished solve_nonlinear ...")


    def linearize(self, inputs, outputs, jacobian):
        """
        Compute derivatives of jacobian (residual) w.r.t. inputs and outputs
        """
        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Running linearize ...")

        self.shell_sim.update_h_th(inputs['h_th'])
        self.shell_sim.update_displacements(outputs['displacements'])
        self.shell_sim.update_external_loads()
        self.shell_sim.update_SVK_residuals()
        
        # Compute derivatives in PETSc matrix format
        self.dresdu = self.shell_sim.dRdu()
        self.dresdt = self.shell_sim.dRdt()

        jacobian['displacements', 'h_th'] = sp.sparse.csr_matrix((self.dresdt.getValuesCSR()[2], self.dresdt.getValuesCSR()[1], self.dresdt.getValuesCSR()[0]))
        jacobian['displacements', 'displacements'] = sp.sparse.csr_matrix((self.dresdu.getValuesCSR()[2], self.dresdu.getValuesCSR()[1], self.dresdu.getValuesCSR()[0]))

        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Finished linearize ...")

    def apply_linear(self, inputs, outputs, d_inputs, 
                     d_outputs, d_residuals, mode):
        """
        Compute linear increments.
        """
        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("---- Running apply_linear ...")

        self.shell_sim.update_h_th(inputs['h_th'])
        self.shell_sim.update_displacements(outputs['displacements'])
        self.shell_sim.update_external_loads()
        self.shell_sim.update_SVK_residuals()

        if mode == 'fwd':
            if 'displacements' in d_residuals:
                dres_petsc = self.shell_sim.u_iga_nest.copy()
                    
                if 'displacements' in d_outputs:
                    # if mpisize > 1:
                    #     du_petsc = self.spline_sim.u_iga_nest.copy()
                    #     du_petsc.zeroEntries()
                    #     update_nest_vec(d_outputs['displacements'], du_petsc)
                    # else:
                    #     du_petsc = array2petsc_vec(d_outputs['displacements'])

                    du_petsc = self.shell_sim.u_iga_nest.copy()
                    update_nest_vec(d_outputs['displacements'], du_petsc)
                    A_x_b(self.dresdu, du_petsc, dres_petsc)
                    # dres = A_x(self.dresdu, du_petsc)
                    d_residuals['displacements'] += get_petsc_vec_array(
                        dres_petsc, self.shell_sim.comm)
                if 'h_th' in d_inputs:
                    # if mpisize > 1:
                    #     dt_petsc = self.spline_sim.h_th_nest.copy()
                    #     dt_petsc.zeroEntries()
                    #     update_nest_vec(d_inputs['t'], dt_petsc)
                    # else:
                    #     dt_petsc = array2petsc_vec(d_inputs['t'])

                    # TODO: Update, h_th is not a PETSc vector
                    dt = self.shell_sim.h_th
                    dresdt_np = sp.sparse.csr_matrix((self.dresdt.getValuesCSR()[2], self.dresdt.getValuesCSR()[1], self.dresdt.getValuesCSR()[0])).toarray()
                    dres = dresdt_np@dt
                    # dres = A_x(self.dresdt, dt_petsc)
                    d_residuals['displacements'] += dres

        elif mode == 'rev':
            if 'displacements' in d_residuals:
                # if mpisize > 1:
                #     dres_petsc = self.spline_sim.u_iga_nest.copy()
                #     dres_petsc.zeroEntries()
                #     update_nest_vec(d_residuals['displacements'], dres_petsc)
                # else:
                #     dres_petsc = array2petsc_vec(d_residuals['displacements'])

                dres_petsc = self.shell_sim.u_iga_nest.copy()
                update_nest_vec(d_residuals['displacements'], dres_petsc)

                if 'displacements' in d_outputs:
                    du_petsc = self.shell_sim.u_iga_nest.copy()
                    # du_petsc.zeroEntries()
                    AT_x_b(self.dresdu, dres_petsc, du_petsc)
                    d_outputs['displacements'] += get_petsc_vec_array(
                        du_petsc, self.shell_sim.comm)
                if 'h_th' in d_inputs:
                    # dt_petsc = self.shell_sim.h_th_nest.copy()
                    # # dt.zeroEntries()
                    # AT_x_b(self.dresdt, dres_petsc, dt_petsc)
                    # # dt = AT_x(self.dresdt, dres_petsc)
                    # d_inputs['h_th'] += get_petsc_vec_array(
                    #                  dt_petsc, self.shell_sim.comm)
                    
                    dt = self.shell_sim.h_th
                    dresdt_np = sp.sparse.csr_matrix((self.dresdt.getValuesCSR()[2], self.dresdt.getValuesCSR()[1], self.dresdt.getValuesCSR()[0])).toarray()
                    dres = dresdt_np.T@(dres_petsc.array)
                    d_inputs['h_th'] += dres

        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("---- Finished apply_linear ...")

    def solve_linear(self, d_outputs, d_residuals, mode):
        """
        Solve linear increments.
        """
        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Running solve_linear ...")
        dresdu_temp = self.dresdu.copy()

        if mode == 'fwd':

            # if mpisize > 1:
            #     dres_petsc = self.spline_sim.u_iga_nest.copy()
            #     dres_petsc.zeroEntries()
            #     update_nest_vec(d_residuals['displacements'], dres_petsc)
            # else:
            #     dres_petsc = array2petsc_vec(d_residuals['displacements'])

            dres_petsc = self.shell_sim.u_iga_nest.copy()
            # dres_petsc.zeroEntries()
            update_nest_vec(d_residuals['displacements'], dres_petsc)

            d_outputs['displacements'] = self.shell_sim.solve_Ax_b(
                                         dresdu_temp, dres_petsc, True)

        elif mode == 'rev':

            # if mpisize > 1:
            #     du_petsc = self.spline_sim.u_iga_nest.copy()
            #     du_petsc.zeroEntries()
            #     update_nest_vec(d_outputs['displacements'], du_petsc)
            # else:
            #     du_petsc = array2petsc_vec(d_outputs['displacements'])
            
            du_petsc = self.shell_sim.u_iga_nest.copy()
            # du_petsc.zeroEntries()
            update_nest_vec(d_outputs['displacements'], du_petsc)

            # zero_dofs = self.spline_sim.nonmatching.global_zero_dofs()
            # dresdu_temp.zeroRowsColumns(zero_dofs, diag=1)
            d_residuals['displacements'] = self.shell_sim.solve_ATx_b(
                                           dresdu_temp, du_petsc, True)
            
        if self.print_info:
            if MPI.rank(self.shell_sim.comm) == 0:
                print("--- Finished solve_linear ...")


if __name__ == "__main__":
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
    comp = StateComp(shell_sim=shell_sim)

    prob.model = comp
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print('check_partials:')
    prob.check_partials(compact_print=True)