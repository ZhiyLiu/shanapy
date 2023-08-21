import vtk
import numpy as np
from .principal_nested_spheres import PNS
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

    @staticmethod
    def euclideanized_srep_features(srep_file_name):
        """
        Euclideanize spokes features and skeletal points given a srep file
        Output tuples: skeletal_pts (3 x n), dirs (3 x n), radii (1 x n), where n denotes
                        the number of skeletal points
        """
        skeletal_pts, dirs, radii = SrepFeatures.default_srep_features(srep_file_name)

        # Euclideanize angle
        pns_model = PNS(dirs, itype=9)
        pns_model.fit()
        euc_dirs, PNS_coords_dirs = pns_model.output

        # Euclideanize skeletal points
        pdms_xs = skeletal_pts[:, ::3]
        pdms_ys = skeletal_pts[:, 1::3]
        pdms_zs = skeletal_pts[:, 2::3]

        centered_xs = pdms_xs - np.mean(pdms_xs, axis=1)[:, None]
        centered_ys = pdms_ys - np.mean(pdms_ys, axis=1)[:, None]
        centered_zs = pdms_zs - np.mean(pdms_zs, axis=1)[:, None]
        var = np.sqrt(centered_xs ** 2 + centered_ys ** 2 + centered_zs ** 2)

        scaled_xs = centered_xs / var
        scaled_ys = centered_ys / var
        scaled_zs = centered_zs / var

        scaled_pdms = np.concatenate((scaled_xs, scaled_ys, scaled_zs), axis=1)
        pns_model_pdm = PNS(scaled_pdms, itype=9)
        pns_model_pdm.fit()
        euc_pdm, PNS_coords_pdmss = pns_model_pdm.output

        # Commensurate radii
        log_radii = np.log(radii)
        geo_mean = np.exp(np.mean(log_radii, axis=-1))
        radii_comm = (log_radii - geo_mean)

        return euc_pdm, euc_dirs, radii_comm
