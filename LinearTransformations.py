import numpy as np

class Lineartransformations:
    @staticmethod
    def coordinate_stretching(par, ind):
        """
        Generic function for coordinate stretching
        """
        if ind > 2:
            return ValueError("Index out of range, use 0, 1 or 2 for x, y or z")
        mat = np.eye(4)
        mat[3,3] = 0
        mat[ind, ind] = par
        return mat
    
    @staticmethod
    def coordinate_rotation(par, ind):
        """
        Generic function for coordinate rotations. 
        Note that `par` must be given in radians
        """
        if par > 2*np.pi:
            return ValueError("Rotation angle `par` must be given in radians")

        mat = np.zeros(4)
        if ind == 0:
            mat[0,0] = 1; 
            mat[1,1] = np.cos(par); mat[1,2] = -np.sin(par)
            mat[2,1] = np.sin(par); mat[2,2] = np.cos(par)
        elif ind == 1:
            mat[1,1] = 1; 
            mat[0,0] = np.cos(par); mat[0,2] = np.sin(par)
            mat[2,0] = -np.sin(par); mat[2,2] = np.cos(par)
        elif ind == 2:
            mat[2,2] = 1; 
            mat[0,0] = np.cos(par); mat[0,1] = -np.sin(par)
            mat[1,0] = np.sin(par); mat[1,1] = np.cos(par)
        else:
            return ValueError("Index out of range, use 0, 1 or 2 for x, y or z")
        return mat
