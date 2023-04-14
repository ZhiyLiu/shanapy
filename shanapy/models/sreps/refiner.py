"""
The refinement aims to optimize the interior geometry represented by the s-rep.
This optimization ought to consider i) the goodness of fitting to the boundary geometry and
ii) the smoothness of interior radial distance level surfaces.
As of Dec. 27, 2021, the refinement in this python package can optimize the spokes' lengths.
"""
from audioop import avg
import numpy as np
import vtk
import nlopt
from .geometry import Geometry

class Refiner:
    """This class optimize an initial s-rep to better fit to the boundary"""
    def __init__(self, surface_mesh):
        ## TODO: Set parameters of refinement
        self.eps = np.finfo(float).eps
        self.input_mesh = surface_mesh
        pass
    def relocate(self, bdry_pt, input_mesh):
        """
        Relocate base points of a (fold) spoke with the two end points: base_pt and bdry_pt,
        such that the length of the spoke is reciprocal of boundary curvature
        """
        # find the closest mesh point to the tip of the spoke
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(input_mesh)
        cell_locator.BuildLocator()

        cellId = vtk.reference(0)
        c = [0.0, 0.0, 0.0]
        subId = vtk.reference(0)
        d = vtk.reference(0.0)
        cell_locator.FindClosestPoint(bdry_pt, c, cellId, subId, d)
        pt_ids = vtk.vtkIdList()
        input_mesh.GetCellPoints(cellId, pt_ids)

        curvature = vtk.vtkCurvatures()
        curvature.SetInputData(input_mesh)
        curvature.SetCurvatureTypeToMean()
        curvature.Update()
        mean_curvatures = curvature.GetOutput()
        mean_curvature = []
        for i in range(pt_ids.GetNumberOfIds()):
            pt_id = pt_ids.GetId(i)
            mean_curvature.append(mean_curvatures.GetPointData().GetArray(0).GetValue(pt_id))
        return 1/ np.mean(mean_curvature)


    def refine(self, srep, num_crest_points=24):
        """
        The main entry of the refinement
        Input: an initial s-rep  srep
        Return spokes poly (a set of spokes that can be visualized) and fold points
        """
        print('Refining ...')
        srep_poly = vtk.vtkPolyData()
        srep_poly.DeepCopy(srep)
        num_pts = srep_poly.GetNumberOfPoints()
        num_spokes = num_pts // 2
        radii_array = np.zeros(num_spokes)
        dir_array = np.zeros((num_spokes, 2))
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
            dir_array[i, :] = Geometry.cart2sph(direction[None, :])[:, :2].squeeze()
            base_array[i, :] = base_pt

        def obj_func(opt_vars, grad=None):
            """
            Square of signed distance from tips
            of spokes to the input_mesh
            """
            implicit_distance = vtk.vtkImplicitPolyDataDistance()
            implicit_distance.SetInput(self.input_mesh)
            total_loss = 0
            tmp_radii_array = opt_vars[:num_spokes]
            tmp_dir_array = np.reshape(opt_vars[num_spokes:], (-1, 2))
            for i in range(num_spokes):
                radius    = tmp_radii_array[i]
                direction = Geometry.sph2cart(tmp_dir_array[i, :][None, :]).squeeze()
                base_pt   = base_array[i, :]
                bdry_pt   = base_pt + radius * direction

                dist = implicit_distance.FunctionValue(bdry_pt)
                total_loss += dist ** 2
            return total_loss
        ### optimize the variables (i.e., radii, directions)
        opt_vars = np.concatenate((radii_array, dir_array.flatten()))
        opt = nlopt.opt(nlopt.LN_BOBYQA, len(opt_vars))
        opt.set_min_objective(obj_func)
        opt.set_maxeval(2000)
        minimizer = opt.optimize(opt_vars)
        min_loss = opt.last_optimum_value()
        opt_dirs = Geometry.sph2cart(np.reshape(minimizer[num_spokes:], (-1, 2)))

        ## update radii of s-rep and return the updated
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
            direction = opt_dirs[i, :]

            new_bdry_pt = base_pt + radius * direction
            arr_length.InsertNextValue(radius)
            arr_dirs.InsertNextTuple(direction)
            srep_poly.GetPoints().SetPoint(id_bdry_pt, new_bdry_pt)

            ### relocate base points for fold spokes such that their lengths are reciprocal of boundary mean curvature
            if i >= num_spokes - num_crest_points:
                new_radius = min(radius - 1, self.relocate(new_bdry_pt, self.input_mesh))

                new_base_pt = new_bdry_pt - new_radius * direction
                srep_poly.GetPoints().SetPoint(id_base_pt, new_base_pt)

        srep_poly.GetPointData().AddArray(arr_length)
        srep_poly.GetPointData().AddArray(arr_dirs)
        srep_poly.Modified()
        return srep_poly
