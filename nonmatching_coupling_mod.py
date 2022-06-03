from PENGoLINS.nonmatching_coupling import *

import time

class NonMatchingCouplingMod(NonMatchingCoupling):
    def assemble_nonmatching(self):
        """
        Assemble the non-matching system.
        """

        time_start = time.time()

        # Step 1: assemble residuals of ExtractedSplines 
        # and derivatives.
        if self.residuals is None:
            if MPI.rank(self.comm) == 0:
                raise RuntimeError("Shell residuals are not specified.") 
        if self.deriv_residuals is None:
            if MPI.rank(self.comm) == 0:
                raise RuntimeError("Derivatives of shell residuals are "
                                   "not specified.")

        # Compute contributions from shell residuals and derivatives
        R_FE = []
        dR_du_FE = []
        for i in range(self.num_splines):
            R_assemble = assemble(self.residuals[i])
            dR_du_assemble = assemble(self.deriv_residuals[i])
            if self.point_sources is not None:
                for j, ps_ind in enumerate(self.point_source_inds):
                    if ps_ind == i:
                        self.point_sources[j].apply(R_assemble)
            R_FE += [v2p(R_assemble),]
            dR_du_FE += [m2p(dR_du_assemble),]

        time_post_shell_assemble = time.time()

        # Step 2: assemble non-matching contributions
        # Create empty lists for non-matching contributions
        Rm_FE = [None for i1 in range(self.num_splines)]
        dRm_dum_FE = [[None for i1 in range(self.num_splines)] 
                      for i2 in range(self.num_splines)]

        # add_nonmatching = True
        # if add_nonmatching:
        #     print("Adding non-matching contributions ...")

        # Compute non-matching contributions ``Rm_FE`` and 
        # ``dRm_dum_FE``.
        for i in range(self.num_interfaces):
            dx_m = dx(domain=self.mortar_meshes[i], 
                      metadata=self.int_measure_metadata)
            PE = penalty_energy(self.splines[self.mapping_list[i][0]], 
                self.splines[self.mapping_list[i][1]], self.mortar_meshes[i],
                self.Vms_control[i], self.dVms_control[i], 
                self.transfer_matrices_control_list[i][0], 
                self.transfer_matrices_control_list[i][1], 
                self.alpha_d_list[i], self.alpha_r_list[i], 
                self.mortar_vars[i][0], self.mortar_vars[i][1], 
                self.t1_A_list[i], self.t2_A_list[i], 
                dx_m=dx_m)

            # An initial check for penalty energy, if ``PE``is nan,
            # raise RuntimeError.
            PE_value = assemble(PE)
            if PE_value is nan:
                if MPI.rank(self.comm) == 0:
                    raise RuntimeError("Penalty energy value is nan between "
                          "splines {:2d} and {:2d}.".format(
                          self.mapping_list[i][0], self.mapping_list[i][1]))

            Rm_list = penalty_differentiation(PE, 
                self.mortar_vars[i][0], self.mortar_vars[i][1])
            Rm = transfer_penalty_differentiation(Rm_list, 
                self.transfer_matrices_list[i][0], 
                self.transfer_matrices_list[i][1])
            dRm_dum_list = penalty_linearization(Rm_list, 
                self.mortar_vars[i][0], self.mortar_vars[i][1])            
            dRm_dum = transfer_penalty_linearization(dRm_dum_list, 
                self.transfer_matrices_list[i][0], 
                self.transfer_matrices_list[i][1])

            for j in range(len(dRm_dum)):
                if Rm_FE[self.mapping_list[i][j]] is not None:
                    Rm_FE[self.mapping_list[i][j]] += Rm[j]
                else:
                    Rm_FE[self.mapping_list[i][j]] = Rm[j]

                for k in range(len(dRm_dum[j])):
                    if dRm_dum_FE[self.mapping_list[i][j]]\
                       [self.mapping_list[i][k]] is not None:
                        dRm_dum_FE[self.mapping_list[i][j]]\
                            [self.mapping_list[i][k]] += dRm_dum[j][k]
                    else:
                        dRm_dum_FE[self.mapping_list[i][j]]\
                            [self.mapping_list[i][k]] = dRm_dum[j][k]

        time_post_nonmatching = time.time()

        # Step 3: add spline residuals and non-matching 
        # contribution together
        for i in range(self.num_splines):
            if Rm_FE[i] is not None:
                Rm_FE[i] += R_FE[i]
                dRm_dum_FE[i][i] += dR_du_FE[i]
            else:
                Rm_FE[i] = R_FE[i]
                dRm_dum_FE[i][i] = dR_du_FE[i]

        time_post_assemble = time.time()

        # Step 4: add contact contributions if contact is given
        if self.contact is not None:
            Kcs, Fcs = self.contact.assembleContact(self.spline_funcs, 
                                                    output_PETSc=True)
            for i in range(self.num_splines):
                if Rm_FE[i] is None:
                    Rm_FE[i] = Fcs[i]
                elif Rm_FE[i] is not None and Fcs[i] is not None:
                    Rm_FE[i] += Fcs[i]
                for j in range(self.num_splines):
                    if dRm_dum_FE[i][j] is None:
                        dRm_dum_FE[i][j] = Kcs[i][j]
                    elif dRm_dum_FE[i][j] is not None \
                        and Kcs[i][j] is not None:
                        dRm_dum_FE[i][j] += Kcs[i][j]

        time_post_contact = time.time()

        print("-------------")
        print("Nonmatching assembly times:")
        print("Shell assembly: {}".format(time_post_shell_assemble-time_start))
        print("Nonmatching assemble: {}".format(time_post_nonmatching-time_post_shell_assemble))
        print("Combine assemblies: {}".format(time_post_assemble-time_post_nonmatching))
        print("Contact assemble: {}".format(time_post_contact-time_post_assemble))
        print("Total time: {}".format(time_post_contact-time_start))
        print("-------------")

        return dRm_dum_FE, Rm_FE

    def solve_nonlinear_nonmatching_problem(self, solver="direct", 
                                            ref_error=None, rtol=1e-3, max_it=20,
                                            zero_mortar_funcs=True, 
                                            ksp_type=PETSc.KSP.Type.CG, 
                                            pc_type=PETSc.PC.Type.FIELDSPLIT, 
                                            fieldsplit_type="additive",
                                            fieldsplit_ksp_type=PETSc.KSP.Type.PREONLY,
                                            fieldsplit_pc_type=PETSc.PC.Type.LU, 
                                            ksp_rtol=1e-15, ksp_max_it=100000,
                                            ksp_view=False, ksp_monitor_residual=False, 
                                            iga_dofs=False):

        time_start = time.time()

        # Zero out values in mortar mesh functions if True
        if zero_mortar_funcs:
            for i in range(len(self.transfer_matrices_list)):
                for j in range(len(self.transfer_matrices_list[i])):
                    for k in range(len(self.transfer_matrices_list[i][j])):
                            self.mortar_vars[i][j][k].interpolate(
                                                      Constant((0.,0.,0.)))

        # If iga_dofs is True, only starts from zero displacements,
        # this argument is designed for solving nonlinear 
        # displacements in IGA DoFs in optimization problem.
        if iga_dofs:
            u_iga_list = []
            for i in range(self.num_splines):
                u_FE_temp = Function(self.splines[i].V)
                u_iga_list += [v2p(multTranspose(self.splines[i].M,
                                   u_FE_temp.vector())),]
                self.spline_funcs[i].interpolate(Constant((0.,0.,0.)))
            u_iga = create_nest_PETScVec(u_iga_list, comm=self.comm)
        
        time_start_loop = time.time()
        presolve_loop_time = time_start_loop-time_start
        assembly_time = 0.
        extraction_time = 0.
        solution_time = 0.
        
        print("-------------")
        print("Nonlinear solving times:")
        print("Pre-solve loop time: {}".format(presolve_loop_time))
        
        

        for newton_iter in range(max_it+1):

            time_start_loop = time.time()

            dRt_dut_FE, Rt_FE = self.assemble_nonmatching()

            time_post_assemble = time.time()

            self.extract_nonmatching_system(Rt_FE, dRt_dut_FE)

            time_post_extract = time.time()

            if solver == "direct":
                if MPI.size(self.comm) == 1:
                    self.A.convert("seqaij")
                else:
                    self.A = create_aijmat_from_nestmat(self.A, self.A_list, 
                                                        comm=self.comm)

            if solver == "ksp" and pc_type != PETSc.PC.Type.FIELDSPLIT:
                self.A = create_aijmat_from_nestmat(self.A, self.A_list, 
                                                    comm=self.comm)

            current_norm = self.b.norm()

            if newton_iter==0 and ref_error is None:
                ref_error = current_norm

            rel_norm = current_norm/ref_error
            if newton_iter >= 0:
                if MPI.rank(self.comm) == 0:
                    print("Solver iteration: {}, relative norm: {:.12}."
                          .format(newton_iter, rel_norm))
                    print("Assembly time: {}".format(time_post_assemble-time_start_loop))
                    print("Extraction time: {}".format(time_post_extract-time_post_assemble))
                    assembly_time += time_post_assemble-time_start_loop
                    extraction_time += time_post_extract-time_post_assemble
                sys.stdout.flush()

            if rel_norm < rtol:
                if MPI.rank(self.comm) == 0:
                    print("Newton's iteration finished in {} "
                          "iterations (relative tolerance: {})."
                          .format(newton_iter, rtol))
                break

            if newton_iter == max_it:
                if MPI.rank(self.comm) == 0:
                    raise StopIteration("Nonlinear solver failed to "
                          "converge in {} iterations.".format(max_it))

            du_list = []
            du_IGA_list = []
            for i in range(self.num_splines):
                du_list += [Function(self.splines[i].V),]
                du_IGA_list += [zero_petsc_vec(self.splines[i].M.size(1), 
                                               comm=self.splines[i].comm)]
            du = create_nest_PETScVec(du_IGA_list, comm=self.comm)

            solve_nonmatching_mat(self.A, du, -self.b, solver=solver,
                                  ksp_type=ksp_type, pc_type=pc_type, 
                                  fieldsplit_type=fieldsplit_type,
                                  fieldsplit_ksp_type=fieldsplit_ksp_type,
                                  fieldsplit_pc_type=fieldsplit_pc_type, 
                                  rtol=ksp_rtol, max_it=ksp_max_it, 
                                  ksp_view=ksp_view, 
                                  monitor_residual=ksp_monitor_residual)


            if iga_dofs:
                u_iga += du

            for i in range(self.num_splines):
                self.splines[i].M.mat().mult(du_IGA_list[i], 
                                             du_list[i].vector().vec())
                self.spline_funcs[i].assign(self.spline_funcs[i]+du_list[i])
                v2p(du_list[i].vector()).ghostUpdate()

            for i in range(len(self.transfer_matrices_list)):
                for j in range(len(self.transfer_matrices_list[i])):
                    for k in range(len(self.transfer_matrices_list[i][j])):
                        A_x_b(self.transfer_matrices_list[i][j][k], 
                            self.spline_funcs[
                                self.mapping_list[i][j]].vector(), 
                            self.mortar_vars[i][j][k].vector())
            
            post_solve_time = time.time()

            if MPI.rank(self.comm) == 0:
                print("Assignment, solution & update time: {}".format(post_solve_time-time_post_extract))
                solution_time += post_solve_time-time_post_extract

        print("-------------")
        print("Total times:")
        print("Pre-solve loop time: {}".format(presolve_loop_time))
        print("Assembly time: {}".format(assembly_time))
        print("Extraction time: {}".format(extraction_time))
        print("Assignment, solution & update time: {}".format(solution_time))

        if iga_dofs:
            return self.spline_funcs, u_iga
        else:
            return self.spline_funcs