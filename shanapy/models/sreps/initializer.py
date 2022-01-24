import vtk
import numpy as np
class Initializer:
    def __init__(self, num_crest_points=24):
        """TODO: allow users to config the following parameters"""
        # the number of iterations of mean curvature flow
        self.iter_num = 500 
        # the flow step size
        self.dt = 0.001
        # determin the resolution of the discrete s-rep
        self.num_crest_points = num_crest_points
    def _get_thin_plate_spline_deform(self, input_target_mesh, input_source_mesh):
        """Compute the deformation via thin plate spline
        from the input target mesh to the input source mesh.
        Note: the boundary points are downsampled to reduce computational cost
        """
        target_mesh = input_target_mesh
        source_mesh = input_source_mesh
    #    compute_distance_between_poly(target_mesh, source_mesh)
        source_pts = vtk.vtkPoints()
        target_pts = vtk.vtkPoints()
        ## downsample the boundary points (10)
        for i in range(0, target_mesh.GetNumberOfPoints(), 20):
            source_pts.InsertNextPoint(source_mesh.GetPoint(i))
            target_pts.InsertNextPoint(target_mesh.GetPoint(i))
        tps = vtk.vtkThinPlateSplineTransform()
        tps.SetSourceLandmarks(source_pts)
        tps.SetTargetLandmarks(target_pts)
        tps.SetBasisToR()
        tps.Modified()

        return tps
    def _forward_flow(self, mesh):
        """
        Flow an input mesh to an ellipsoid.
        Current implementation is based on mean curvature flow, see Liu et al. MEDIA paper in 2021.

        Input: A triangle mesh with no holes (e.g., resulting from SPHARM-PDM).
        Output: A near-ellipsoid surface mesh
        Output: A list of intermediate meshes from the input mesh to the near-ellipsoid mesh.
        """
        prev_mesh = mesh
        thin_plate_spline_list = []
        deformed_surfaces = []
        for i in range(self.iter_num + 1):
            # deformed_surface_writer = vtk.vtkPolyDataWriter()
            # deformed_surface_writer.SetFileName('../../data/forward/' + str(i) + '.vtk')
            # deformed_surface_writer.SetInputData(mesh)
            # deformed_surface_writer.Update()
            deformed_surfaces.append(mesh)

            taubin_smooth = vtk.vtkWindowedSincPolyDataFilter()
            taubin_smooth.SetInputData(mesh)
            taubin_smooth.SetNumberOfIterations(20)
            taubin_smooth.BoundarySmoothingOff()
            taubin_smooth.FeatureEdgeSmoothingOff()
            taubin_smooth.SetPassBand(0.01)
            taubin_smooth.NonManifoldSmoothingOn()
            taubin_smooth.NormalizeCoordinatesOn()

            taubin_smooth.Update()
            mesh = taubin_smooth.GetOutput()

            normal_generator = vtk.vtkPolyDataNormals()
            normal_generator.SetInputData(mesh)
            normal_generator.SplittingOff()
            normal_generator.ComputePointNormalsOn()
            normal_generator.ComputeCellNormalsOff()
            normal_generator.Update()
            mesh = normal_generator.GetOutput()

            curvatures = vtk.vtkCurvatures()
            curvatures.SetCurvatureTypeToMean()
            curvatures.SetInputData(mesh)
            curvatures.Update()

            mean_curvatures = curvatures.GetOutput().GetPointData().GetArray("Mean_Curvature")
            normals = normal_generator.GetOutput().GetPointData().GetNormals()

            mesh_pts = mesh.GetPoints()
            deform_fields = []
            pv_mean_curvatures = []
            for j in range(mesh.GetNumberOfPoints()):
                current_point = mesh.GetPoint(j)
                current_normal = np.array(normals.GetTuple3(j))
                current_mean_curvature = mean_curvatures.GetValue(j)

                pt = np.array(mesh_pts.GetPoint(j))
                pt -= self.dt * current_mean_curvature * current_normal
                # deform_fields.append(pyvista.Arrow(pt, -dt * current_mean_curvature * current_normal))
                # pv_mean_curvatures.append(-current_mean_curvature)
                mesh_pts.SetPoint(j, pt)
            
            mesh_pts.Modified()
            mesh.SetPoints(mesh_pts)
            mesh.Modified()

            tps_deform = self._get_thin_plate_spline_deform(prev_mesh, mesh)

            prev_mesh = mesh
            thin_plate_spline_list.append(tps_deform)
        
        return mesh, thin_plate_spline_list
    def _compute_discrete_srep(self, input_center, rx, ry, rz, vh):
        """Compute the s-rep (also Blum axis) for a standard ellipsoid parametrized by rx, ry, rz. """
        num_crest_points = self.num_crest_points
        eps = np.finfo(float).eps
        mrx_o = (rx*rx-rz*rz)/rx
        mry_o = (ry*ry-rz*rz)/ry

        ELLIPSE_SCALE = 0.9
        mrb = mry_o * ELLIPSE_SCALE
        mra = mrx_o * ELLIPSE_SCALE

        delta_theta = 2 * np.pi / num_crest_points
        num_steps = 3
        skeletal_pts_x = np.zeros((num_crest_points, num_steps))
        skeletal_pts_y = np.zeros((num_crest_points, num_steps))
        skeletal_pts_z = np.zeros((num_crest_points, num_steps))
        bdry_up_x = np.zeros((num_crest_points, num_steps))
        bdry_up_y = np.zeros((num_crest_points, num_steps))
        bdry_up_z = np.zeros((num_crest_points, num_steps))

        bdry_down_x = np.zeros((num_crest_points, num_steps))
        bdry_down_y = np.zeros((num_crest_points, num_steps))
        bdry_down_z = np.zeros((num_crest_points, num_steps))

        crest_bdry_pts = np.zeros((num_crest_points, 3))
        crest_skeletal_pts = np.zeros((num_crest_points, 3))
        for i in range(num_crest_points):
            theta = np.pi - delta_theta * i
            x = mra * np.cos(theta)
            y = mrb * np.sin(theta)

            mx_ = (mra * mra - mrb * mrb) * np.cos(theta) / mra
            my_ = .0
            dx_ = x - mx_
            dy_ = y - my_

            step_size = 1.0 / float(num_steps-1)

            for j in range(num_steps):
                sp_x = mx_ + step_size * j * dx_
                sp_y = my_ + step_size * j * dy_

                ### Make the inner loops (onion skins) sharper
                # if i == 0 and j == 0:
                #     sp_x = mx_ + step_size * j * dx_ - 1
                # elif i == 0 and j == 1:
                #     sp_x = mx_ + step_size * j * dx_-0.75
                # elif i == num_crest_points / 2 and j == 1:
                #     sp_x = mx_ + step_size * j * dx_+0.75
                # elif i == num_crest_points / 2 and j == 0:
                #     sp_x = mx_ + step_size * j * dx_+1
                skeletal_pts_x[i, j] = sp_x
                skeletal_pts_y[i, j] = sp_y
                sin_spoke_angle = sp_y * mrx_o
                cos_spoke_angle = sp_x * mry_o

                # normalize to [-1, 1]
                l = np.sqrt(sin_spoke_angle ** 2 + cos_spoke_angle ** 2)
                if l >  eps:
                    sin_spoke_angle /= l
                    cos_spoke_angle /= l
                cos_phi = l / (mrx_o * mry_o)
                sin_phi = np.sqrt(1 - cos_phi ** 2)
                bdry_x = rx * cos_phi * cos_spoke_angle
                bdry_y = ry * cos_phi * sin_spoke_angle
                bdry_z = rz * sin_phi
                bdry_up_x[i, j] = bdry_x
                bdry_up_y[i, j] = bdry_y
                bdry_up_z[i, j] = bdry_z

                bdry_down_x[i, j] = bdry_x
                bdry_down_y[i, j] = bdry_y
                bdry_down_z[i, j] = -bdry_z

                ## if at the boundary of the ellipse, add crest spokes
                if j == num_steps - 1:
                    cx = rx * cos_spoke_angle - sp_x
                    cy = ry * sin_spoke_angle - sp_y
                    cz = 0
                    vec_c = np.asarray([cx, cy, cz])
                    norm_c = np.linalg.norm(vec_c)
                    dir_c = np.asarray([bdry_x - sp_x, bdry_y - sp_y, 0.0])
                    dir_c = dir_c / np.linalg.norm(dir_c)

                    crest_spoke = norm_c * dir_c
                    crest_bdry_x = crest_spoke[0] + sp_x
                    crest_bdry_y = crest_spoke[1] + sp_y
                    crest_bdry_z = 0.0

                    crest_bdry_pts[i] = np.asarray([crest_bdry_x, crest_bdry_y, crest_bdry_z])
                    crest_skeletal_pts[i] = np.asarray([sp_x, sp_y, 0.0])
        ### Rotate skeletal/implied boundary points as boundary points of the ellipsoid
        rot_obj = np.flipud(vh.T)
        ## make this rotation matrix same with c++ computation with Eigen3
        # rot_obj[0, :] *= -1
        # rot_obj[-1, :] *= -1

        concate_skeletal_pts = np.concatenate((skeletal_pts_x.flatten()[:, np.newaxis], \
                                            skeletal_pts_y.flatten()[:, np.newaxis], \
                                            skeletal_pts_z.flatten()[:, np.newaxis]), \
                                                    axis=1)
        concate_bdry_up_pts = np.concatenate((bdry_up_x.flatten()[:, np.newaxis], \
                                        bdry_up_y.flatten()[:, np.newaxis], \
                                        bdry_up_z.flatten()[:, np.newaxis]), axis=1)
        concate_bdry_down_pts = np.concatenate((bdry_down_x.flatten()[:, np.newaxis], \
                                                bdry_down_y.flatten()[:, np.newaxis], \
                                                bdry_down_z.flatten()[:, np.newaxis]), axis=1)

        second_moment_srep = np.matmul(concate_skeletal_pts.T, concate_skeletal_pts)
        s_srep, v_srep = np.linalg.eig(second_moment_srep)

        rot_srep = np.flipud(v_srep.T)

        rotation = np.flipud(np.flipud(np.matmul(rot_obj, rot_srep)).T).T

        transformed_concate_skeletal_pts = np.matmul(concate_skeletal_pts, rotation) + input_center
        transformed_concate_bdry_up_pts = np.matmul(concate_bdry_up_pts, rotation) + input_center
        transformed_concate_bdry_down_pts = np.matmul(concate_bdry_down_pts, rotation) + input_center
        transformed_crest_bdry_pts = np.matmul(crest_bdry_pts, rotation) + input_center
        transformed_crest_skeletal_pts = np.matmul(crest_skeletal_pts, rotation) + input_center

        ### Convert spokes to visualizable elements
        up_spokes_poly = vtk.vtkPolyData()
        up_spokes_pts = vtk.vtkPoints()
        up_spokes_cells = vtk.vtkCellArray()
        down_spokes_poly = vtk.vtkPolyData()
        down_spokes_pts = vtk.vtkPoints()
        down_spokes_cells = vtk.vtkCellArray()
        crest_spokes_poly = vtk.vtkPolyData()
        crest_spokes_pts = vtk.vtkPoints()
        crest_spokes_cells = vtk.vtkCellArray()

        for i in range(concate_skeletal_pts.shape[0]):
            id_s = up_spokes_pts.InsertNextPoint(transformed_concate_skeletal_pts[i, :])
            id_b = up_spokes_pts.InsertNextPoint(transformed_concate_bdry_up_pts[i, :])

            id_sdwn = down_spokes_pts.InsertNextPoint(transformed_concate_skeletal_pts[i, :])
            id_down = down_spokes_pts.InsertNextPoint(transformed_concate_bdry_down_pts[i, :])

            up_spoke = vtk.vtkLine()
            up_spoke.GetPointIds().SetId(0, id_s)
            up_spoke.GetPointIds().SetId(1, id_b)
            up_spokes_cells.InsertNextCell(up_spoke)

            down_spoke = vtk.vtkLine()
            down_spoke.GetPointIds().SetId(0, id_sdwn)
            down_spoke.GetPointIds().SetId(1, id_down)
            down_spokes_cells.InsertNextCell(down_spoke)


        up_spokes_poly.SetPoints(up_spokes_pts)
        up_spokes_poly.SetLines(up_spokes_cells)
        down_spokes_poly.SetPoints(down_spokes_pts)
        down_spokes_poly.SetLines(down_spokes_cells)

        for i in range(num_crest_points):
            id_crest_s = crest_spokes_pts.InsertNextPoint(transformed_crest_skeletal_pts[i, :])
            id_crest_b = crest_spokes_pts.InsertNextPoint(transformed_crest_bdry_pts[i, :])
            crest_spoke = vtk.vtkLine()
            crest_spoke.GetPointIds().SetId(0, id_crest_s)
            crest_spoke.GetPointIds().SetId(1, id_crest_b)
            crest_spokes_cells.InsertNextCell(crest_spoke)
        crest_spokes_poly.SetPoints(crest_spokes_pts)
        crest_spokes_poly.SetLines(crest_spokes_cells)

    
        append_filter = vtk.vtkAppendPolyData()
        append_filter.AddInputData(up_spokes_poly)
        append_filter.AddInputData(down_spokes_poly)
        append_filter.AddInputData(crest_spokes_poly)
        append_filter.Update()
        return append_filter.GetOutput()
    def _compute_ellipsoidal_srep(self, near_ellipsoid):
        """Compute the s-rep (also Blum axis) for an ellipsoid. """
        # 1. obtain the standard ellipsoid that is the best approximation of the near-ellipsoid
        ## Note: This is done based on eigen-analysis on the boundary points. 
        ## It is necessary that the boundary points are uniformly distributed.
        ## Otherwise, try to remesh the boundary points.
        np_vertices = []
        mesh = near_ellipsoid
        for i in range(mesh.GetNumberOfPoints()):
            np_vertices.append(np.array(mesh.GetPoint(i)))
        np_vertices = np.array(np_vertices)
        center = np.mean(np_vertices, axis=0)

        centered_vertices = np_vertices - center[None, :]
        w, v = np.linalg.eig(np.matmul(centered_vertices.T, centered_vertices))
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:, idx]

        ## correct the volume
        r0, r1, r2 = np.sqrt(w[0]), np.sqrt(w[1]), np.sqrt(w[2])
        ellipsoid_volume = 4 / 3.0 * np.pi * r0 * r1 * r2
        mass_filter = vtk.vtkMassProperties()
        mass_filter.SetInputData(mesh)
        mass_filter.Update()

        # radii of the best fitting ellipsoid
        volume_factor = pow(mass_filter.GetVolume() / ellipsoid_volume, 1.0 / 3.0)
        r0 *= volume_factor
        r1 *= volume_factor
        r2 *= volume_factor
        ## 2. Compute the s-rep for the best fitting ellipsoid, see wenqi's slides for notations.
        srep_poly = self._compute_discrete_srep(center, r0, r1, r2, v)
        
        # ellipsoid_param = vtk.vtkParametricEllipsoid()
        # ellipsoid_param.SetXRadius(r0)
        # ellipsoid_param.SetYRadius(r1)
        # ellipsoid_param.SetZRadius(r2)
        # param_funct = vtk.vtkParametricFunctionSource()
        # param_funct.SetUResolution(30)
        # param_funct.SetVResolution(30)
        # param_funct.SetParametricFunction(ellipsoid_param)
        # param_funct.Update()
        # best_fitting_ellipsoid = param_funct.GetOutput()
        
        return srep_poly
    def _backward_flow(self, srep_poly, tps_list):
        """Flow s-rep back to fit the input mesh. """
        deformed_srep = vtk.vtkPolyData()
        deformed_srep.DeepCopy(srep_poly)

        new_pts = vtk.vtkPoints()
        
        ### Convert vtk Points to numpy array for the backward flow
        ### to reduce computational cost.
        transform_pts = []
        for j in range(deformed_srep.GetNumberOfPoints()):
            transform_pts.append(np.array(deformed_srep.GetPoint(j)))
        transform_pts = np.array(transform_pts)        
        while tps_list:
            tps = tps_list.pop()
            
            for k in range(transform_pts.shape[0]):
                transform_pts[k] = tps.TransformPoint(transform_pts[k])
            
        for j in range(transform_pts.shape[0]):
            new_pts.InsertNextPoint(transform_pts[j])
        deformed_srep.SetPoints(new_pts)
        deformed_srep.Modified() 
        return deformed_srep
    def fit(self, input_mesh):
        """Main entry of initializing an s-rep given a closed surface mesh (input_mesh)."""
        print("Initializing ...")
        # 1. Via mean curvature flow, deform the input mesh to a near-ellipsoid
        near_ellipsoid, tps_list = self._forward_flow(input_mesh)

        # 2. Compute the s-rep that best fit the near-ellipsoid
        srep_poly = self._compute_ellipsoidal_srep(near_ellipsoid)

        # 3. Via inverse mean curvature flow, deform the s-rep of the near-ellipsoid to fit the obj
        obj_srep = self._backward_flow(srep_poly, tps_list)

        return obj_srep
        