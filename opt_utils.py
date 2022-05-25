from dolfin import *
from tIGAr import *
from tIGAr.BSplines import *
from tIGAr.NURBS import *
import scipy as sp

from PENGoLINS.nonmatching_utils import *

"""
Taken from PENGoLINS optimization example from fe-iga-mdao-sandbox. 
Some routines might have been modified or added.
"""

def array2petsc_vec(ndarray, comm=worldcomm):
    """
    Convert numpy array to petsc vector.
    """
    size = ndarray.size
    petsc_vec = zero_petsc_vec(size, 'mpi', comm=comm)
    petsc_vec.setValues(np.arange(size, dtype='int32'), ndarray)
    petsc_vec.assemble()
    return petsc_vec

def get_petsc_vec_array(petsc_vec, comm=worldcomm):
    """
    Get global values from petsc vector
    """
    if MPI.size(comm) > 1:
        if petsc_vec.type == 'nest':
            sub_vecs = petsc_vec.getNestSubVecs()
            num_sub_vecs = len(sub_vecs)
            sub_vecs_array_list = []
            for i in range(num_sub_vecs):
                sub_vecs_array_list += [np.concatenate(
                    comm.allgather(sub_vecs[i].array))]
            array = np.concatenate(sub_vecs_array_list)
        else:
            array = np.concatenate(comm.allgather(petsc_vec.array))
    else:
        array = petsc_vec.array
    return array

def update_func(u, u_array):
    """
    Update values in a dolfin function.
    """
    u_petsc = v2p(u.vector())
    u_petsc.setValues(np.arange(u_array.size, dtype='int32'), u_array)
    u_petsc.assemble()
    u_petsc.ghostUpdate()

def update_nest_vec(vec_array, nest_vec, comm=worldcomm):
    """
    Assign values from numpy array to a nest petsc vector.
    """
    if nest_vec.type != 'nest':
        if MPI.rank(comm) == 0:
            raise TypeError("Type of PETSc vector is not nest.")

    sub_vecs = nest_vec.getNestSubVecs()
    num_sub_vecs = len(sub_vecs)

    sub_vecs_range = []
    sub_vecs_size = []
    for i in range(num_sub_vecs):
        sub_vecs_range += [sub_vecs[i].getOwnershipRange(),]
        sub_vecs_size += [sub_vecs[i].size,]

    sub_array_list = []
    array_ind_off = 0
    for i in range(num_sub_vecs):
        sub_array_list += [vec_array[array_ind_off+sub_vecs_range[i][0]: 
                                     array_ind_off+sub_vecs_range[i][1]],]
        array_ind_off += sub_vecs_size[i]
    sub_array = np.concatenate(sub_array_list)
    nest_vec.setArray(sub_array)
    nest_vec.assemble()

def petsc_vec_to_aijmat(vec, comm=worldcomm):
    vec_arr = vec.array
    vec_sparse = sp.sparse.csr_matrix(np.expand_dims(vec_arr, 1))
    # A_new = PETSc.Mat(comm)
    # A_new.createAIJ((vec_arr.shape[0],1), comm=comm)
    # A_new.setPreallocationNNZ([vec_arr.shape[0], 1])
    # # A_new.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
    # # A_new.setUp()
    # # A_new.assemble()
    # A_new.setValues(np.linspace(0,vec_arr.shape[0]-1,vec_arr.shape[0], dtype=int), col_ind_subs[col_range_iter], 
    #                                     vec_arr)
    # A_new.setUp()
    # A_new.assemble()

    # A = PETSc.Mat().create()
    # A.setSizes([vec_arr.shape[0], 1])
    # A.setType("aij")
    # A.setUp()

    # First arg is list of row indices, second list of column indices
    # A.setValues(vec_sparse.indptr, vec_sparse.indices, vec_sparse.data)
    # A.assemble()

    A = PETSc.Mat().createAIJ(size=vec_sparse.shape, csr=(vec_sparse.indptr, vec_sparse.indices, vec_sparse.data))
    return A



def PETSc_ksp_solve(A, x, b, ksp_type='cg', pc_type ='jacobi', 
                    max_it=10000, rtol=1e-15):
    """
    PETSc KSP solver to solve "Ax=b".
    """
    ksp = PETSc.KSP().create()
    ksp.setType(ksp_type)
    pc = ksp.getPC()
    PETScOptions.set('pc_type', pc_type)
    pc.setType(pc_type)
    ksp.setTolerances(rtol=rtol)
    ksp.max_it = max_it
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.solve(b, x)
    x.ghostUpdate()
    x.assemble()
    return x

def Newton_solve(A, x, b, max_it=10):
    ref_norm = b.norm()
    b_solve = b.copy()
    b_solve.zeroEntries()
    dx = x.copy()
    dx.zeroEntries()

    for i in range(max_it):

        A.mult(x, b_solve)
        b_diff = b_solve - b

        rel_norm = b_diff.norm()/ref_norm

        print("Iteration: {}, relative norm: {}".format(i, rel_norm))

        solve_nonmatching_mat(A, dx, -b_diff, solver='direct')

        x += dx