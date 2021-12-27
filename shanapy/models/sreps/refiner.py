"""
The refinement aims to optimize the interior geometry represented by the s-rep.
This optimization ought to consider i) the goodness of fitting to the boundary geometry and 
ii) the smoothness of interior radial distance level surfaces.
As of Dec. 27, 2021, the refinement in this python package can optimize the spokes' lengths.
"""
import numpy as np
import vtk
import nlopt
class Refiner:
    """This class optimize an initial s-rep to better fit to the boundary"""
    def __init__(self):
        ## TODO: Set parameters of refinement
        self.eps = np.finfo(float).eps
        pass
    
    def refine(self, srep, input_mesh):
        """
        The main entry of the refinement
        Input: an initial s-rep  srep
        Input: the boundary mesh input_mesh
        """
        print('Refining ...')
        srep_poly = vtk.vtkPolyData()
        srep_poly.DeepCopy(srep)
        num_pts = srep_poly.GetNumberOfPoints()
        num_spokes = num_pts // 2
        radii_array = np.zeros(num_spokes)
        dir_array = np.zeros((num_spokes, 3))
        base_array = np.zeros((num_spokes,3))

        ### read the parameters from s-rep
        for i in range(num_spokes):
            id_base_pt = i * 2
            id_bdry_pt = id_base_pt + 1
            base_pt = np.array(srep_poly.GetPoint(id_base_pt))
            bdry_pt = np.array(srep_poly.GetPoint(id_bdry_pt))

            radius = np.linalg.norm(bdry_pt - base_pt)
            direction = (bdry_pt - base_pt) / radius

            radii_array[i] = radius
            dir_array[i, :] = direction
            base_array[i, :] = base_pt
        def obj_func(radii, grad=None):
            """
            Square of signed distance from tips
            of spokes to the input_mesh
            """
            implicit_distance = vtk.vtkImplicitPolyDataDistance()
            implicit_distance.SetInput(input_mesh)
            total_loss = 0
            for i, radius in enumerate(radii):
                direction = dir_array[i, :]
                base_pt   = base_array[i, :]
                bdry_pt   = base_pt + radius * direction

                dist = implicit_distance.FunctionValue(bdry_pt)
                total_loss += dist ** 2
            return total_loss
        # from scipy import optimize as opt
        # minimum = opt.fmin(obj_func, radii_array)

        # minimizer = minimum[0]

        ### optimize the variables (i.e., radii)
        
        opt = nlopt.opt(nlopt.LN_NEWUOA, len(radii_array))
        opt.set_min_objective(obj_func)
        opt.set_maxeval(2000)
        minimizer = opt.optimize(radii_array)

        ## update radii of s-rep and return the updated
        num_diff_spokes = 0
        arr_length = vtk.vtkDoubleArray()
        arr_length.SetNumberOfComponents(1)
        arr_length.SetName("spokeLength")

        arr_dirs = vtk.vtkDoubleArray()
        arr_dirs.SetNumberOfComponents(3)
        arr_dirs.SetName("spokeDirection")
        for i in range(num_spokes):
            id_base_pt = i * 2
            id_bdry_pt = id_base_pt + 1
            base_pt = base_array[i, :]
            radius = minimizer[i]
            direction = dir_array[i, :]

            new_bdry_pt = base_pt + radius * direction
            arr_length.InsertNextValue(radius)
            arr_dirs.InsertNextTuple(direction)
            srep_poly.GetPoints().SetPoint(id_bdry_pt, new_bdry_pt)

            if np.abs(np.linalg.norm(new_bdry_pt - base_pt) - radii_array[i]) > self.eps:
                num_diff_spokes += 1
        srep_poly.GetPointData().AddArray(arr_length)
        srep_poly.GetPointData().AddArray(arr_dirs)
        srep_poly.Modified()
        return srep_poly