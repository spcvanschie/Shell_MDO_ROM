from tIGAr.common import *
from tIGAr.BSplines import *
from PENGoLINS.occ_utils import *


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


def OCCBSpline2tIGArSpline(surface, SAVE_PATH= "./", num_field=3, quad_deg_const=4, 
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

def clampedBC(spline_generator, side=0, direction=0):
    """
    Apply clamped boundary condition to spline.
    """
    for field in [0,1,2]:
        scalar_spline = spline_generator.getScalarSpline(field)
        side_dofs = scalar_spline.getSideDofs(direction, side, nLayers=2)
        spline_generator.addZeroDofs(field, side_dofs)