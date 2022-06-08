from PENGoLINS.nonmatching_coupling import *

import time

class NonMatchingCouplingMod(NonMatchingCoupling):
    def mortar_meshes_setup(self, mapping_list, mortar_parametric_coords, 
                            penalty_coefficient=1000):
        """
        Set up coupling of non-matching system for mortar meshes.

        Parameters
        ----------
        mapping_list : list of ints
        mortar_parametric_coords : list of ndarrays
        penalty_coefficient : float, optional, default is 1000
        """
        self._create_mortar_vars()
        self.mapping_list = mapping_list
        self.penalty_coefficient = penalty_coefficient
        self.t1_A_list = []
        self.t2_A_list = []
        self.transfer_matrices_list = []
        self.transfer_matrices_control_list = []
        self.transfer_matrices_linear_list = []
        self.hm_avg_list = []

        # print("checkpoint 1")

        for i in range(self.num_interfaces):
            transfer_matrices = [[], []]
            transfer_matrices_control = [[], []]
            transfer_matrices_linear = [[], []]
            if self.h_th_is_function:
                transfer_matrices_thickness = [[], []]
            for j in range(len(self.mapping_list[i])):
                # print("checkpoint pre-2, i: {}, j: {}".format(i, j))
                move_mortar_mesh(self.mortar_meshes[i], 
                                 mortar_parametric_coords[i][j])
                if j == 0:
                    t11, t21 = tangent_components(self.mortar_meshes[i])
                    self.t1_A_list += [t11]
                    self.t2_A_list += [t21]

                # print("checkpoint 2, i: {}, j: {}".format(i, j))
                # Create transfer matrices
                transfer_matrices[j] = create_transfer_matrix_list(
                    self.splines[self.mapping_list[i][j]].V, 
                    self.Vms[i], self.dVms[i])
                # print("checkpoint 2a, i: {}, j: {}".format(i, j))
                transfer_matrices_control[j] = create_transfer_matrix_list(
                    self.splines[self.mapping_list[i][j]].V_control, 
                    self.Vms_control[i], self.dVms_control[i])
                # print("checkpoint 2b, i: {}, j: {}".format(i, j))
                transfer_matrices_linear[j] = create_transfer_matrix(
                    self.splines[self.mapping_list[i][j]].V_linear,
                    self.Vms_control[i])
                # print("checkpoint 2c, i: {}, j: {}".format(i, j))

            # Store transfers in lists for future use
            self.transfer_matrices_list += [transfer_matrices,]
            self.transfer_matrices_control_list += [transfer_matrices_control]
            self.transfer_matrices_linear_list += [transfer_matrices_linear,]

            # print("checkpoint 2.5, i: {}".format(i))

            s_ind0, s_ind1 = self.mapping_list[i]
            # Compute element length
            h0 = spline_mesh_size(self.splines[s_ind0])
            h0_func = self.splines[s_ind0]\
                .projectScalarOntoLinears(h0, lumpMass=True)
            h0m = Function(self.Vms_control[i])
            A_x_b(transfer_matrices_linear[0], h0_func.vector(), h0m.vector())
            h1 = spline_mesh_size(self.splines[s_ind1])
            h1_func = self.splines[s_ind1]\
                .projectScalarOntoLinears(h1, lumpMass=True)
            h1m = Function(self.Vms_control[i])
            A_x_b(transfer_matrices_linear[1], h1_func.vector(), h1m.vector())
            hm_avg = 0.5*(h0m+h1m)
            self.hm_avg_list += [hm_avg,]

        # print("checkpoint 3")

        if self.h_th_is_function:
            self._create_transfer_matrices_thickness()
        
        # print("checkpoint 4")
        self.penalty_parameters()

    def penalty_parameters(self, E=None, h_th=None, nu=None, 
                           method='minimum'):
        """
        Create lists for pealty paramters for displacement and rotation.

        Parameters
        ----------
        E : ufl Constant or list, Young's modulus
        h_th : ufl Constant or list, thickness of the splines
        nu : ufl Constant or list, Poisson's ratio
        method: str, {'minimum', 'maximum', 'average'}
        """
        # First initialize material and geometric paramters, then
        # check if h_th is DOLFIN function and if the transfer
        # matrices are created for the thickness.
        self._init_properties(E, h_th, nu)
        if (self.h_th_is_function and not 
            hasattr(self, 'transfer_matrices_thickness_list')):
            self._create_transfer_matrices_thickness()

        self.alpha_d_list = []
        self.alpha_r_list = []

        for i in range(self.num_interfaces):
            s_ind0, s_ind1 = self.mapping_list[i]

            # # Original implementation
            # # Use "Minimum" method for spline patches with different
            # # material properties. 
            # # For other methods, see Herrema et al. Section 4.2
            # # For uniform isotropic material:
            # max_Aij0 = float(self.E[s_ind0]*self.h_th[s_ind0]\
            #            /(1-self.nu[s_ind0]**2))
            # max_Aij1 = float(self.E[s_ind1]*self.h_th[s_ind1]\
            #            /(1-self.nu[s_ind1]**2))
            # alpha_d = Constant(self.penalty_coefficient)\
            #           /self.hm_avg_list[i]*min(max_Aij0, max_Aij1)
            # max_Dij0 = float(self.E[s_ind0]*self.h_th[s_ind0]**3\
            #            /(12*(1-self.nu[s_ind0]**2)))
            # max_Dij1 = float(self.E[s_ind1]*self.h_th[s_ind1]**3\
            #            /(12*(1-self.nu[s_ind1]**2)))
            # alpha_r = Constant(self.penalty_coefficient)\
            #           /self.hm_avg_list[i]*min(max_Dij0, max_Dij1)
            # self.alpha_d_list += [alpha_d,]
            # self.alpha_r_list += [alpha_r,]

            if self.h_th_is_function:
                h_th0 = Function(self.Vms_control[i])
                h_th1 = Function(self.Vms_control[i])
                A_x_b(self.transfer_matrices_thickness_list[i][0],
                      self.h_th[s_ind0].vector(), h_th0.vector())
                A_x_b(self.transfer_matrices_thickness_list[i][1],
                      self.h_th[s_ind1].vector(), h_th1.vector())
            else:
                h_th0 = self.h_th[s_ind0]
                h_th1 = self.h_th[s_ind1]

            max_Aij0 = self.E[s_ind0]*h_th0\
                       /(1-self.nu[s_ind0]**2)
            max_Aij1 = self.E[s_ind1]*h_th1\
                       /(1-self.nu[s_ind1]**2)
            max_Dij0 = self.E[s_ind0]*h_th0**3\
                       /(12*(1-self.nu[s_ind0]**2))
            max_Dij1 = self.E[s_ind1]*h_th1**3\
                       /(12*(1-self.nu[s_ind1]**2))

            if method == 'minimum':
                alpha_d = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*min_value(max_Aij0, max_Aij1)
                alpha_r = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*min_value(max_Dij0, max_Dij1)
            elif method == 'maximum':
                alpha_d = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*max_value(max_Aij0, max_Aij1)
                alpha_r = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*max_value(max_Dij0, max_Dij1)
            elif method == 'average':
                alpha_d = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*(max_Aij0+max_Aij1)*0.5
                alpha_r = Constant(self.penalty_coefficient)\
                          /self.hm_avg_list[i]*(max_Dij0+max_Dij1)*0.5
            else:
                raise TypeError("Method:", method, "is not supported.")
            self.alpha_d_list += [alpha_d,]
            self.alpha_r_list += [alpha_r,]

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
        R_FE, dR_du_FE = self.assemble_KL_shells()
        time_post_shell_assemble = time.time()

        # Step 2: assemble non-matching contributions
        # Create empty lists for non-matching contributions
        Rm_FE, dRm_dum_FE = self.assemble_intersections()

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

    def assemble_KL_shells(self, deriv_list=True):
        R_FE = []
        if deriv_list:
            dR_du_FE = []
        else:
            dR_du_FE = [[None for i1 in range(self.num_splines)] 
                      for i2 in range(self.num_splines)]
        for i in range(self.num_splines):
            R_assemble = assemble(self.residuals[i])
            dR_du_assemble = assemble(self.deriv_residuals[i])
            if self.point_sources is not None:
                for j, ps_ind in enumerate(self.point_source_inds):
                    if ps_ind == i:
                        self.point_sources[j].apply(R_assemble)
            R_FE += [v2p(R_assemble),]
            if deriv_list:
                dR_du_FE += [m2p(dR_du_assemble),]
            else:
                dR_du_FE[i][i] = m2p(dR_du_assemble)
        return R_FE, dR_du_FE
    
    def assemble_intersections(self):
        Rm_FE = [None for i1 in range(self.num_splines)]
        dRm_dum_FE = [[None for i1 in range(self.num_splines)] 
                      for i2 in range(self.num_splines)]

        # TODO: Add function to update 

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
        return Rm_FE, dRm_dum_FE

    def extract_nonmatching_system(self, Rt_FE, dRt_dut_FE, save_as_self=True):
        """
        Extract matrix and vector to IGA space.

        Parameters
        ----------
        Rt_FE : list of assembled vectors
        dRt_dut_FE : list of assembled matrices

        Returns
        -------
        b : petsc4py.PETSc.Vec
            LHS of non-matching system
        A : petsc4py.PETSc.Mat
            RHS of non-matching system
        """
        Rt_IGA = []
        dRt_dut_IGA = []
        for i in range(self.num_splines):
            Rt_IGA += [v2p(FE2IGA(self.splines[i], Rt_FE[i], True)),]
            # Rt_IGA += [AT_x(self.splines[i].M, Rt_FE[i]),]
            dRt_dut_IGA += [[],]
            for j in range(self.num_splines):
                if dRt_dut_FE[i][j] is not None:
                    dRm_dum_IGA_temp = AT_R_B(m2p(self.splines[i].M), 
                                  dRt_dut_FE[i][j], m2p(self.splines[j].M))

                    if i==j:
                        dRm_dum_IGA_temp = apply_bcs_mat(self.splines[i], 
                                           dRm_dum_IGA_temp, diag=1)
                    else:
                        dRm_dum_IGA_temp = apply_bcs_mat(self.splines[i], 
                                           dRm_dum_IGA_temp, self.splines[j], 
                                           diag=0)
                else:
                    dRm_dum_IGA_temp = None

                dRt_dut_IGA[i] += [dRm_dum_IGA_temp,]

        A_list = dRt_dut_IGA
        b_list = Rt_IGA

        b = create_nest_PETScVec(Rt_IGA, comm=self.comm)
        A = create_nest_PETScMat(dRt_dut_IGA, comm=self.comm)

        if save_as_self:
            self.A_list = A_list
            self.b_list = b_list
            self.A = A
            self.b = b

        return A, b, A_list, b_list

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
                                            iga_dofs=False, POD_obj=None):

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


            # map quantities to POD basis if POD_obj is defined
            if POD_obj is not None:
                if POD_obj.basistype != "Global":
                    du_IGA_r_list = []
                    for i in range(self.num_splines):
                        du_IGA_r_list += [zero_petsc_vec(POD_obj.r_list[i], 
                                                    comm=self.splines[i].comm)]
                    
                    du = create_nest_PETScVec(du_IGA_r_list, comm=self.comm)

                    # generate reduced A_list and use it to generate a new A matrix
                    A_r_list = []

                    for i in range(self.num_splines):

                        A_r_list += [[],]

                        for j in range(self.num_splines):
                            A_submat = self.A_list[i][j]
                            if A_submat is not None:
                                V_submat_l = POD_obj.V_blocks_petsc[i][i]
                                V_submat_r = POD_obj.V_blocks_petsc[j][j]
                                A_r_list[i] += [V_submat_l.transposeMatMult(A_submat).matMult(V_submat_r)]
                            else:
                                A_r_list[i] += [None]

                    self.A = create_nest_PETScMat(A_r_list, comm=self.comm)

                    # convert A matrix to correct form
                    if solver == "direct":
                        if MPI.size(self.comm) == 1:
                            self.A.convert("seqaij")
                        else:
                            self.A = create_aijmat_from_nestmat(self.A, A_r_list, 
                                                                comm=self.comm)

                    if solver == "ksp" and pc_type != PETSc.PC.Type.FIELDSPLIT:
                        self.A = create_aijmat_from_nestmat(self.A, A_r_list, 
                                                            comm=self.comm)

                    # generate reduced b_list and use it to generate a new b vector
                    b_r_list = []
                    
                    for i in range(self.num_splines):
                        b_subvec = self.b_list[i]
                        V_submat = POD_obj.V_blocks_petsc[i][i]

                        b_r = V_submat.createVecRight()
                        V_submat.multTranspose(b_subvec, b_r)
                        b_r_list += [b_r]

                    self.b = create_nest_PETScVec(b_r_list, comm=self.comm)

                else:
                    # create full V_mat
                    V_mat = create_nest_PETScMat([POD_obj.V_blocks_petsc])
                    V_mat = create_aijmat_from_nestmat(V_mat, [POD_obj.V_blocks_petsc])
                    self.A = V_mat.transposeMatMult(self.A).matMult(V_mat)

                    b_r = V_mat.createVecRight()
                    V_mat.multTranspose(self.b, b_r)
                    self.b = create_nest_PETScVec([b_r], comm=self.comm)

                    du_IGA_r_list = [zero_petsc_vec(POD_obj.r)]
                    
                    du = create_nest_PETScVec(du_IGA_r_list, comm=self.comm)

            else:
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

            # if a POD-ROM is used, convert du back to its full size for the rest of the computations
            if POD_obj is not None:
                if POD_obj.basistype != "Global":
                    du_list = []
                    du_IGA_list = []
                    
                    for i, subvec in enumerate(du_IGA_r_list):
                        du_list += [Function(self.splines[i].V),]

                        V_submat = POD_obj.V_blocks_petsc[i][i]

                        iga_petsc_vec = zero_petsc_vec(self.splines[i].M.size(1), 
                                                    comm=self.splines[i].comm)

                        V_submat.mult(subvec, iga_petsc_vec)
                        du_IGA_list += [iga_petsc_vec]

                    du = create_nest_PETScVec(du_IGA_list, comm=self.comm)
                else:
                    du_list = []
                    du_IGA_list = []

                    du_iga_array = POD_obj.V_mat_full@du.array
                    own_range = [0] + list(np.cumsum(POD_obj.dofs_per_patch))
                    for i in range(len(self.splines)):
                        du_list += [Function(self.splines[i].V),]

                        # iga_petsc_vec = zero_petsc_vec(self.splines[i].M.size(1), 
                        #                             comm=self.splines[i].comm)                        
                        du_fom_i = du_iga_array[own_range[i]:own_range[i+1]]

                        petsc_vec_test = PETSc.Vec().createWithArray(du_fom_i)
                        # petsc_vec_test.setType('mpi')
                        du_IGA_list += [petsc_vec_test]

                    du = create_nest_PETScVec(du_IGA_list, comm=self.comm)

                if POD_obj.subtract_mean and newton_iter == 0:
                    du += POD_obj.avg_iga_vec_petsc

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