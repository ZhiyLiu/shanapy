"""
The refinement aims to optimize the interior geometry represented by the s-rep.
This optimization ought to consider i) the goodness of fitting to the boundary geometry and 
ii) the smoothness of interior radial distance level surfaces.
As of Dec. 27, 2021, the refinement in this python package can optimize the spokes' lengths.
"""
from audioop import avg
from turtle import color
import numpy as np
import vtk
import nlopt
# from shanapy.models.sreps import Spoke, Interpolater, Onion
from shanapy.models.sreps.spoke import Spoke
from shanapy.models.sreps.interpolater import Interpolater
from shanapy.models.sreps.onion_skins import Onion
import pyvista as pv

class Refiner:
    """This class optimize an initial s-rep to better fit to the boundary"""
    def __init__(self, num_crest_points=24, num_radial_samples=3):
        ## TODO: Set parameters of refinement
        self.eps = np.finfo(float).eps
        self.interpolate_level = 1
        self.num_crest_points = num_crest_points
        self.num_radial_samples = num_radial_samples
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
    def sph2cart(self, rtp):
        """
        Transform spherical to Cartesian coordinates.
        [X,Y,Z] = sph2cart(rthetaphi) transforms corresponding elements of
        data stored in spherical coordinates (azimuth TH, elevation PHI,
        radius R) to Cartesian coordinates X,Y,Z.  The arrays TH, PHI, and
        R must be the same size (or any of them can be scalar).  TH and
        PHI must be in radians.
    
        TH is the counterclockwise angle in the xy plane measured from the
        positive x axis.  PHI is the elevation angle from the xy plane.

        Input rthetaphi:  phi, theta
        Return matrix: n x 3
        """
        if len(rtp.shape) == 2:
            az, elev = rtp[:, 0], rtp[:, 1]
            r = np.ones_like(az)

            z = np.multiply(r, np.sin(elev))[:, np.newaxis]
            rcoselev = np.multiply(r, np.cos(elev))
            x = np.multiply(rcoselev, np.cos(az))[:, np.newaxis]
            y = np.multiply(rcoselev, np.sin(az))[:, np.newaxis]
            return np.hstack((x, y, z))
        else:
            ## input n x k x 2
            n = rtp.shape[0]
            ret = []
            for ni in range(n):
                feat_slice = rtp[ni, :, :]
                feat_cart = self.sph2cart(feat_slice)
                ret.append(feat_cart)
            return np.array(ret)
    def cart2sph(self, xyz):
        """
        Transform Cartesian to spherical coordinates.
        [TH,PHI,R] = cart2sph(X,Y,Z) transforms corresponding elements of
        data stored in Cartesian coordinates X,Y,Z to spherical
        coordinates (azimuth TH, elevation PHI, and radius R).  The arrays
        X,Y, and Z must be the same size (or any of them can be scalar).
        TH and PHI are returned in radians.
    
        TH is the counterclockwise angle in the xy plane measured from the
        positive x axis.  PHI is the elevation angle from the xy plane.

        Input xyz: n x 3 or n x k x 3
        Return n x 3 or n x k x 3
        """
        ## vectorization to speedup
        if len(xyz.shape) == 2:
            xy = xyz[:,0]**2 + xyz[:,1]**2
            r = np.sqrt(xy + xyz[:,2]**2)  #r
            elev = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from Z-axis down
            az = np.arctan2(xyz[:,1], xyz[:,0]) # az (i.e., theta)
            return np.hstack((az[:, np.newaxis], elev[:, np.newaxis], r[:, np.newaxis]))
        else:
            ## input n x k x 3
            n = xyz.shape[0]
            ret = []
            for ni in range(n):
                feat_slice = xyz[ni, :, :]
                feat_sph = self.cart2sph(feat_slice)
                ret.append(feat_sph)
            return np.array(ret)
    def _update_srep(self, params):
        """
        Assume the parameters contain spokes' directions by Feb., 2022
        Input params: ndarray of dimension n x 3, where n is the number of skeletal points
         (n = 72 + 72 + 24: there are 72 skeletal points on each smooth side and 24 fold points)
        """
        spoke_end_pts = vtk.vtkPoints()
        ret_spokes = []
        # p = pv.Plotter()
        # p.add_mesh(self.srep, color='white', line_width=4)
        dir_diff = 0
        for i in range(self.srep.GetNumberOfPoints()//2):
            base_pt = np.array(self.srep.GetPoint(i*2))
            bdry_pt = np.array(self.srep.GetPoint(i*2+1))
            spoke = bdry_pt - base_pt
            spoke_len = np.linalg.norm(spoke)
            spoke_dir = spoke / spoke_len
            new_spoke_dir = params[i, :]
            new_spoke_end = base_pt + spoke_len * new_spoke_dir
            spoke_end_pts.InsertNextPoint(base_pt)
            spoke_end_pts.InsertNextPoint(new_spoke_end)
            new_spoke = Spoke(base_pt=base_pt, bdry_pt=new_spoke_end)
            ret_spokes.append(new_spoke)
            dir_diff += np.dot(spoke_dir, new_spoke_dir)
        self.srep.SetPoints(spoke_end_pts)
        print(dir_diff)
        # p.add_mesh(self.srep, color='red', line_width=4)
        # p.show()
        return ret_spokes                                   
        
    def _obtain_level_surface(self, spokes):
        """
        Generate interior level surfaces 
        """
        ## Interpolate up spokes
        interpolate_level = self.interpolate_level
        interp = Interpolater(interpolate_level=interpolate_level)
        interp_spokes, up_spokes = interp.interpolate(self.srep, self.num_crest_points, self.num_radial_samples)

        ## Interpolate down spokes
        interp.interpolate_up = False
        interp_down_spokes, down_spokes = interp.interpolate(self.srep, self.num_crest_points, self.num_radial_samples)

        ## interpolate fold spokes
        crest_spokes = interp.interpolate_crest(self.srep, up_spokes, down_spokes, self.num_crest_points)

        num_steps = np.power(2, interpolate_level)
        onion_skins = Onion(num_steps, num_fold_pts=len(crest_spokes)//2)
        top_spokes, bot_spokes = [], []

        symm_ids = (num_steps + 1) * 2
        
        for total_ri in up_spokes.keys():
            if len(up_spokes[total_ri]) != (num_steps - 1) * (self.num_radial_samples - 1)  + self.num_radial_samples:
                top_spokes += [up_spokes[total_ri - symm_ids][0]]
                bot_spokes += [down_spokes[total_ri-symm_ids][0]]
                symm_ids += 2
            top_spokes += up_spokes[total_ri]
            bot_spokes += down_spokes[total_ri]
            
        interior_surfs = onion_skins.get_skins(top_spokes + bot_spokes + crest_spokes)
        
        # colors = ['#063852', '#636466', '#1e9adf', '#ffc100', '#e6ebed']
        # labels = ['Tau = 0.2', 'Tau = 0.4', 'Tau = 0.6', 'Tau = 0.8', 'Tau = 1']
        # obj_polydata = pv.PolyData(interior_surfs[-1])
        # cube_center = [obj_polydata.center[0] + 6, obj_polydata.center[1] + 9, obj_polydata.center[2]]
        # cube = pv.Cube(center=(cube_center), x_length=10, y_length=20, z_length=12)
        # # cube.rotate_x(2)
        
        # for i, level_surf in enumerate(interior_surfs):
        #     p = pv.Plotter()
        #     clipped = pv.PolyData(level_surf).clip_box(cube)
        #     p.add_mesh(level_surf, color='white', label=labels[i], show_edges=True)
        #     p.add_mesh(self.srep, color='#ffc100', label='Spokes', line_width=4)
        #     p.add_legend()
        #     p.add_axes(box=True)
        #     p.show()
        return interior_surfs
    def _orthogonal_loss(self, surfs):
        """Compute orthogonal penalties of spokes w.r.t. level surfaces"""
        for i in range(self.srep.GetNumberOfPoints()//2):
            curr_base_pt = self.srep.GetPoint(i*2)
            curr_bdry_pt = self.srep.GetPoint((i*2)+1)
            idx_next_theta = (i + self.num_radial_samples) % (self.num_crest_points * self.num_radial_samples)
    def _compute_loss(self, dirs, grad=None):
        """
        Loss function 
        """
        dirs_sph = dirs.reshape((-1, 2))
        dirs_car = self.sph2cart(dirs_sph)
        ## Update s-rep according to the current optimizing parameters
        updated_spokes = self._update_srep(dirs_car)

        ## Obtain level surfaces from the updated spokes
        interior_surfs = self._obtain_level_surface(updated_spokes)

        ## Evaluate the loss
        loss = self._orthogonal_loss(interior_surfs)
        print(loss)
        return loss
    def refine(self, srep, input_mesh):
        """
        The main entry of the refinement
        Input: an initial s-rep  srep
        Input: the boundary mesh input_mesh
        Return spokes poly (a set of spokes that can be visualized) and fold points
        """
        print('Refining ...')
        
        srep_poly = vtk.vtkPolyData()
        srep_poly.DeepCopy(srep)
        self.srep = srep_poly
        num_pts = srep_poly.GetNumberOfPoints()
        num_spokes = num_pts // 2
        num_smooth_spokes = self.num_crest_points * self.num_radial_samples
        # spokes = []
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
            base_array[i, :] = base_pt
            if i >= num_smooth_spokes * 2:
                ## fold spokes
                dir_array[i, :] = direction
                continue
            idx_next_theta = (i%num_smooth_spokes + self.num_radial_samples) 
            idx_pre_theta =  num_smooth_spokes + (i%num_smooth_spokes - self.num_radial_samples)
            if i in [0, num_smooth_spokes//2, num_smooth_spokes, num_smooth_spokes//2 + num_smooth_spokes]:
                ## spine's ends
                idx_next_theta = idx_pre_theta + 1
                idx_pre_theta, idx_next_theta = idx_next_theta, idx_pre_theta
            elif (i%num_smooth_spokes) > num_smooth_spokes - self.num_radial_samples:
                ## on the last radial line
                idx_next_theta = self.num_radial_samples - (num_smooth_spokes - i)
            if i >= num_smooth_spokes:
                idx_pre_theta, idx_next_theta = idx_next_theta, idx_pre_theta
            idx_next_tau = (i + 1) if ((i+1)%self.num_radial_samples)!=0 else i
            idx_pre_tau = (i - 1) if (i % self.num_radial_samples)!=0 else i
            
            vec_theta = np.array(srep_poly.GetPoint(idx_next_theta *2)) - np.array(srep_poly.GetPoint(idx_pre_theta*2))
            
            unit_vec_theta = vec_theta / np.linalg.norm(vec_theta)
            vec_tau = np.array(srep_poly.GetPoint(idx_next_tau*2)) - np.array(srep_poly.GetPoint(idx_pre_tau*2))
            unit_vec_tau = vec_tau / np.linalg.norm(vec_tau)
            new_dir = np.cross(unit_vec_theta, unit_vec_tau)
            
            # p = pv.Plotter()
            # # p.add_mesh(pv.Sphere(center=np.array(srep_poly.GetPoint(idx_next_theta *2))), color='red', label='Next')
            # # p.add_mesh(pv.Sphere(center=np.array(srep_poly.GetPoint(idx_pre_theta *2))), color='blue', label='Pre')
            # # p.add_mesh(pv.Sphere(center=np.array(srep_poly.GetPoint(i *2))), color='cyan', label='Curr')
            # p.add_mesh(input_mesh, color='white', show_edges=True, label='SPHARM-PDM', opacity=0.5)
            # p.add_mesh(pv.Arrow(base_pt, new_dir, scale=10), color='red', label='New')
            # p.add_mesh(pv.Arrow(base_pt, direction, scale=10), color='blue', label='Old')
            # p.add_legend()
            # p.add_axes(box=True)
            # p.show()

            dir_array[i, :] = new_dir
            # bdry_pt = base_pt + radius * new_dir
            # spokes.append(Spoke(base_pt=base_pt, bdry_pt=bdry_pt))
            
        # from scipy import optimize as opt
        # minimum = opt.fmin(obj_func, radii_array)
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
        # minimizer = minimum[0]
        # dir_list = dir_array.flatten()
        # opt = nlopt.opt(nlopt.LN_NEWUOA, len(dir_list))
        # opt.set_min_objective(self._compute_loss)
        # opt.set_maxeval(2000)
        # minimizer = opt.optimize(dir_list)
        opt = nlopt.opt(nlopt.LN_NEWUOA, len(radii_array))
        opt.set_min_objective(obj_func)
        opt.set_maxeval(2000)
        minimizer = opt.optimize(radii_array)

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
            direction = dir_array[i, :]

            new_bdry_pt = base_pt + radius * direction
            arr_length.InsertNextValue(radius)
            arr_dirs.InsertNextTuple(direction)
            srep_poly.GetPoints().SetPoint(id_bdry_pt, new_bdry_pt)

            ### relocate base points for fold spokes such that their lengths are reciprocal of boundary mean curvature 
            # if i >= num_spokes - num_crest_points:
            #     new_radius = min(radius - 1, self.relocate(new_bdry_pt, input_mesh))
                
            #     new_base_pt = new_bdry_pt - new_radius * direction
            #     srep_poly.GetPoints().SetPoint(id_base_pt, new_base_pt)

        srep_poly.GetPointData().AddArray(arr_length)
        srep_poly.GetPointData().AddArray(arr_dirs)
        srep_poly.Modified()
        p = pv.Plotter()
        p.add_mesh(srep_poly, color='#ffc100', line_width=4, label='Spokes')
        p.add_mesh(input_mesh, color='white', show_edges=True, label='SPHARM-PDM', opacity=0.5)
        p.add_legend()
        p.add_axes(box=True)
        p.show()
        return srep_poly