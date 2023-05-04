import vtk
import numpy as np
class SrepFeatures():
    def __init__():
        pass
    @staticmethod
    def default_srep_features(srep_file_name):
        """
        Spoke features: skeletal points, radii and directions
        Output tuples: skeletal_pts (3 x n), dirs (3 x n), radii (1 x n)
        """
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(srep_file_name)
        reader.Update()
        srep_poly = reader.GetOutput()

        num_spokes = srep_poly.GetNumberOfPoints() // 2
        radii = np.zeros((1, num_spokes))
        dirs = np.zeros((3, num_spokes))
        skeletal_pts = np.zeros((3, num_spokes))
        for i in range(num_spokes):
            base_pt_id = i * 2
            skeletal_pt = np.array(srep_poly.GetPoint(base_pt_id))
            skeletal_pts[:, i] = skeletal_pt

            bdry_pt = np.array(srep_poly.GetPoint(base_pt_id + 1))
            radius = np.linalg.norm(bdry_pt - skeletal_pt)
            radii[:, i] = radius
            direction = (bdry_pt - skeletal_pt) / radius
            dirs[:, i] = direction

        return skeletal_pts, dirs, radii

    

