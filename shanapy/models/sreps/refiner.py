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
import pyvista as pv

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
        return 1/ np.abs(np.mean(mean_curvature))

    def compute_srad_penalty(self, srep, spoke_offset=0):
        # dx/du, dx/dv, dr/du, dr/dv, ds/du, ds/dv
        num_crest_pt = 24
        skeletal_pt_mat  = np.zeros((3, 72))
        spoke_vector_mat = np.zeros((3, 72))
        spoke_dir_mat    = np.zeros((3, 72))
        spoke_rad_mat    = np.zeros((1, 72))
        fold_pt_mat = np.zeros((3, 24)) # num_crest_pt
        fold_vec_mat = np.zeros((3, 24))
        fold_rad_mat = np.zeros((1, 24))

        num_pts = srep.GetNumberOfPoints()
        i_fold_spoke = 0
        for fold_pt_id in range(num_pts - 1*2, num_pts - (num_crest_pt+1) * 2, -2):
            fold_base_pt = np.array(srep.GetPoint(fold_pt_id))
            fold_pt_mat[:, i_fold_spoke] = fold_base_pt
            fold_spoke = np.array(srep.GetPoint(fold_pt_id + 1)) - fold_base_pt
            fold_spoke_rad = np.linalg.norm(fold_spoke)

            fold_vec_mat[:, i_fold_spoke] = fold_spoke
            fold_rad_mat[:, i_fold_spoke] = fold_spoke_rad
            i_fold_spoke += 1

        # for computing finite differences: u <==> theta; v <==> tau_1
        backward_u_id = np.zeros(72, np.int16) - 1
        forward_u_id  = np.zeros(72, np.int16) - 1
        backward_v_id = np.zeros(72, np.int16) - 1
        forward_v_id  = np.zeros(72, np.int16) - 1
        for i in range(72):
            base_pt_id = i * 2 + spoke_offset
            skeletal_pt = np.array(srep.GetPoint(base_pt_id))
            skeletal_pt_mat[:, i] = skeletal_pt

            bdry_pt = np.array(srep.GetPoint(base_pt_id + 1))
            spoke = bdry_pt - skeletal_pt
            spoke_radius = np.linalg.norm(spoke)
            spoke_dir_mat[:, i] = spoke / spoke_radius
            spoke_rad_mat[:, i] = spoke_radius
            spoke_vector_mat[:, i] = spoke

            if (i + 1) % 3 == 0:
                # the id associated with fold spokes
                forward_v_id[i] = 72 + ((i+1)//3 - 1)
            else:
                forward_v_id[i] = i + 1
            
            backward_v_id[i] = i - 1 if i not in {0, 36} else max(3, i-3)
            if (i % 3) == 0 and i not in {0, 36}:
                backward_v_id[i] = (num_crest_pt - (i // 3)) * 3 + 1

            forward_u_id[i] = i + 3
            backward_u_id[i] = i - 3 if i not in {0, 1, 2, 36, 37, 38} else -1
        
        # merge smooth skeletal points and fold points
        all_skeletal_points = np.concatenate((skeletal_pt_mat, fold_pt_mat), axis=1)
        all_spoke_radii = np.concatenate((spoke_rad_mat, fold_rad_mat), axis=1)
        all_spokes = np.concatenate((spoke_vector_mat, fold_vec_mat), axis=1)
        dp_du_mat = np.zeros((3, 72)) 
        dp_dv_mat = np.zeros((3, 72))
        ds_du_mat = np.zeros((3, 72)) 
        ds_dv_mat = np.zeros((3, 72))
        dr_du_mat = np.zeros((1, 72))
        dr_dv_mat = np.zeros((1, 72))

        total_r_srad_penalty = 0.0
        for i in range(72):
            # central difference of dx/dv
            dp_dv_mat[:, i] = 0.5 * (all_skeletal_points[:, forward_v_id[i]] - all_skeletal_points[:, backward_v_id[i]])
            dr_dv_mat[:, i] = 0.5 * (all_spoke_radii[:, forward_v_id[i]] - all_spoke_radii[:, backward_v_id[i]])
            ds_dv_mat[:, i] = 0.5 * (all_spokes[:, forward_v_id[i]] - all_spokes[:, backward_v_id[i]])

            # central difference or forward difference of dx/du because of singularity at the ends of spine
            dp_du_mat[:, i] = 0.5 * (all_skeletal_points[:, forward_u_id[i]] - all_skeletal_points[:, backward_u_id[i]]) \
                if backward_u_id[i] != -1 else all_skeletal_points[:, forward_u_id[i]] - all_skeletal_points[:, i]
            dr_du_mat[:, i] = 0.5 * (all_spoke_radii[:, forward_u_id[i]] - all_spoke_radii[:, backward_u_id[i]]) \
                if backward_u_id[i] != -1 else all_spoke_radii[:, forward_u_id[i]] - all_spoke_radii[:, i]
            ds_du_mat[:, i] = 0.5 * (all_spokes[:, forward_u_id[i]] - all_spokes[:, backward_u_id[i]]) \
                if backward_u_id[i] != -1 else all_spokes[:, forward_u_id[i]] - all_spokes[:, i]
            
            ut_u_minus_I = np.outer(spoke_dir_mat[:, i], spoke_dir_mat[:, i]) - np.eye(3)

            Q = np.concatenate((np.dot(dp_du_mat[:, i][None, :], ut_u_minus_I), np.dot(dp_dv_mat[:, i][None, :], ut_u_minus_I)))
            ds_du = np.concatenate((ds_du_mat[:, i][None, :], ds_dv_mat[:, i][None, :])) # 2 x 3
            dr_du = np.concatenate((dr_du_mat[:, i][None, :], dr_dv_mat[:, i][None, :]))   # 2 x 1
            q_free_term = ds_du - np.dot(dr_du, spoke_dir_mat[:, i][None, :])
            q_term = np.dot(Q.T, np.linalg.inv(np.dot(Q, Q.T)))
            r_srad = np.dot(q_free_term, q_term)
            det_r_srad = np.linalg.det(r_srad)
            if det_r_srad < 1: continue
            total_r_srad_penalty += abs(det_r_srad) - 1.0 if det_r_srad >= 0 else 100.0
        return np.float64(total_r_srad_penalty)
    def update_srep(self, opt_vars, skeletal_points, num_crest_points=24, relocate_fold=False):
        num_spokes = skeletal_points.shape[0]
        opt_dirs = np.reshape(opt_vars[num_spokes:], (-1, 2))
        opt_radii = np.exp(opt_vars[:num_spokes])
        # Identify spokes on spine
        for i in range(1, num_crest_points//2):
            avg_radii_up = (opt_radii[i*3] + opt_radii[(num_crest_points - i) * 3]) / 2
            opt_radii[i*3] = opt_radii[(num_crest_points - i) * 3] = avg_radii_up

            avg_radii_down = (opt_radii[(i+24)*3] + opt_radii[(num_crest_points - i + 24) * 3]) / 2
            opt_radii[(i+24)*3] = opt_radii[(num_crest_points - i + 24) * 3] = avg_radii_down

            avg_dir_up = (opt_dirs[i*3, :] + opt_dirs[(num_crest_points - i) * 3, :]) / 2
            opt_dirs[i*3, :] = opt_dirs[(num_crest_points - i) * 3, :] = avg_dir_up

            avg_dir_down = (opt_dirs[(i+24)*3, :] + opt_dirs[(num_crest_points - i + 24) * 3, :]) / 2
            opt_dirs[(i+24)*3, :] = opt_dirs[(num_crest_points - i + 24) * 3, :] = avg_dir_down

        opt_dirs = Geometry.sph2cart(opt_dirs)

        refined_srep_poly = vtk.vtkPolyData()
        refined_srep_spokes = vtk.vtkCellArray()
        refined_points = vtk.vtkPoints()
        
        for i in range(num_spokes):
            id_base_pt = i * 2
            id_bdry_pt = id_base_pt + 1
            base_pt = skeletal_points[i, :]
            radius = opt_radii[i]
            direction = opt_dirs[i, :]
            new_bdry_pt = base_pt + radius * direction
        
            ### relocate base points for fold spokes such that their lengths are reciprocal of boundary mean curvature
            if relocate_fold and i >= num_spokes - num_crest_points:
                new_radius = min(radius - 1, self.relocate(new_bdry_pt, self.input_mesh))
                base_pt = new_bdry_pt - new_radius * direction
           
            id_base = refined_points.InsertNextPoint(base_pt)
            id_bdry = refined_points.InsertNextPoint(new_bdry_pt)
            line_spoke = vtk.vtkLine()
            line_spoke.GetPointIds().SetId(0, id_base)
            line_spoke.GetPointIds().SetId(1, id_bdry)
            refined_srep_spokes.InsertNextCell(line_spoke)

            refined_srep_poly.SetPoints(refined_points)
            refined_srep_poly.SetLines(refined_srep_spokes)
        return refined_srep_poly
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
            temp_vars = np.array(opt_vars, copy=True)
            temp_srep = self.update_srep(temp_vars, base_array)
            up_srad_penalty = self.compute_srad_penalty(temp_srep)
            down_srad_penalty = self.compute_srad_penalty(temp_srep, 72)
            
            implicit_distance = vtk.vtkImplicitPolyDataDistance()
            implicit_distance.SetInput(self.input_mesh)
            total_loss = 0
            tmp_radii_array = opt_vars[:num_spokes]
            tmp_dir_array = np.reshape(opt_vars[num_spokes:], (-1, 2))

            bdry_penalties = []
            dir_penalties = []
            total_losses = []
            for i in range(num_spokes):
                radius    = np.exp(tmp_radii_array[i])
                direction = Geometry.sph2cart(tmp_dir_array[i, :][None, :]).squeeze()
                base_pt   = base_array[i, :]
                bdry_pt   = base_pt + radius * direction
                grad_bdry = np.zeros(3, np.float32)
                dist_bdry = implicit_distance.EvaluateFunction(bdry_pt)
                implicit_distance.EvaluateGradient(bdry_pt, grad_bdry)

                spoke_dir = (bdry_pt - base_pt) / (np.linalg.norm(bdry_pt - base_pt) + self.eps)
                bdry_penalty = np.abs(dist_bdry)
                dir_penalty = (1 - np.dot(spoke_dir, grad_bdry)) * np.abs(dist_bdry)
                
                total_loss += bdry_penalty * 10 + dir_penalty # * 0.05
                bdry_penalties.append(bdry_penalty)
                dir_penalties.append(dir_penalty)
                total_losses.append(bdry_penalty + dir_penalty)

            return total_loss + 5 * (up_srad_penalty + down_srad_penalty)
        ### optimize the variables (i.e., radii, directions)
        opt_vars = np.concatenate((np.log(radii_array), dir_array.flatten()))
        opt = nlopt.opt(nlopt.LN_BOBYQA, len(opt_vars))
        opt.set_min_objective(obj_func)
        opt.set_maxeval(2000)
        minimizer = opt.optimize(opt_vars)
        refined_srep_poly = self.update_srep(minimizer, base_array, relocate_fold=True)
        # new_loss = obj_func(minimizer)
        
        return refined_srep_poly
