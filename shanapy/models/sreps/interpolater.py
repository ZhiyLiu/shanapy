"""Interpolate s-rep to have a denser (continuous) spoke field.

The theory can be found in the paper Z. Liu et al. Fitting unbranching skeletal structures to objects, 2021

Author: Zhiyuan Liu
Date: Feb 5, 2022

"""
import collections
import sys
from turtle import color
from shanapy.models.sreps.spoke import Spoke
import numpy as np
from numpy import sin
import numpy.linalg as LA
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import vtk
import pyvista as pv

# set the upper limit for recursion
sys.setrecursionlimit(10**6)
epsilon = 1e-7
## Definition of Hermitian spline functions
def h1(s):
    return 2*(s * s * s) - 3*(s * s) + 1
def h2(s):
    return -2*(s * s * s) + 3*(s * s)
def h3(s):
    return (s * s * s) - 2*(s * s) + s
def h4(s):
    return (s * s * s) - (s * s)

class Interpolater(object):
    """
    Interpolate an srep read by the function readSrepFromXML (organized srep).
    Return in the function interpolate(...) a list of spokes.
    """
    def __init__(self, interpolate_level=3):
        assert interpolate_level >= 0, "The interpolate_level has to be non-negative."
        self.interpolate_level = interpolate_level
        self.interpolate_up = True
        
    def _compute_derivative(self, input_srep, r, c, num_crest_points, num_samples_outward):
        """Use finite difference to compute derivatives of skeletal positions"""
        id = r * num_samples_outward + c
        pt0 = pt1 = [0] * 3 # pt1 - pt0 for dxdu
        pt0v = pt1v = [0] * 3 # for dxdv
        base_pts_array = self._get_base_points_array(input_srep, num_crest_points)
        if r == 0:
            # first row
            pt0 = base_pts_array[id]
            pt1 = base_pts_array[id+num_samples_outward]
            factor = 1.0
        elif r == num_crest_points - 1:
            # last row
            pt1 = base_pts_array[id]
            pt0 = base_pts_array[id-num_samples_outward]
            factor = 1.0
        else:
            # otherwise
            pt1 = base_pts_array[id + num_samples_outward]
            pt0 = base_pts_array[id - num_samples_outward]
            factor = 0.5
        dxdu = [(pt1i - pt0i) * factor for pt1i, pt0i in zip(pt1, pt0)]
        # p = pv.Plotter()
        # p.add_mesh(input_srep, color='white', line_width=4)
        # p.add_mesh(pv.Sphere(radius=0.1, center=np.array(pt0)), color='red')
        # p.add_mesh(pv.Sphere(radius=0.1, center=np.array(pt1)), color='yellow')
        # p.show()
        if c == 0:
            pt1v = base_pts_array[id+1]
            pt0v = base_pts_array[id]
            factor = 1.0
        elif c == num_samples_outward - 1:
            pt1v = base_pts_array[id]
            pt0v = base_pts_array[id-1]
            factor = 1.0
        else:
            pt1v = base_pts_array[id + 1]
            pt0v = base_pts_array[id - 1]
            factor = 0.5
        dxdv = [(pt1iv - pt0iv) * factor for pt1iv, pt0iv in zip(pt1v, pt0v)]

        return dxdu, dxdv
    def _interpolate_skeleton(self, relative_position, corner_pts, corner_deriv):
        """Interpolating skeletal sheet using Hermite spline functions, see Vicory's dissertation for the notation.
        The interpolated position at (u, v) is given by p = H(u) * H_c * H(v), where H_c is determined by control points.
        """
        u, v = relative_position
        dxdu11, dxdv11, dxdu21, dxdv21, dxdu12, dxdv12, dxdu22, dxdv22 = corner_deriv
        x11, x21, x22, x12 = corner_pts

        hx = [[0 for i in range(4)] for j in range(4)]
        hy = [[1 for i in range(4)] for j in range(4)]
        hz = [[2 for i in range(4)] for j in range(4)]
        hx[0][0] = x11[0];          hx[0][1] = x12[0]
        hx[1][0] = x21[0];          hx[1][1] = x22[0]
        hx[2][0] = dxdu11[0];       hx[2][1] = dxdu12[0]
        hx[3][0] = dxdu21[0];       hx[3][1] = dxdu22[0]
        hx[0][2] = dxdv11[0];       hx[0][3] = dxdv12[0]
        hx[1][2] = dxdv21[0];       hx[1][3] = dxdv22[0]
        hx[2][2] = 0;               hx[2][3] = 0
        hx[3][2] = 0;               hx[3][3] = 0

        hy[0][0] = x11[1];          hy[0][1] = x12[1]
        hy[1][0] = x21[1];          hy[1][1] = x22[1]
        hy[2][0] = dxdu11[1];       hy[2][1] = dxdu12[1]
        hy[3][0] = dxdu21[1];       hy[3][1] = dxdu22[1]
        hy[0][2] = dxdv11[1];       hy[0][3] = dxdv12[1]
        hy[1][2] = dxdv21[1];       hy[1][3] = dxdv22[1]
        hy[2][2] = 0;               hy[2][3] = 0
        hy[3][2] = 0;               hy[3][3] = 0

        hz[0][0] = x11[2];       hz[0][1] = x12[2]
        hz[1][0] = x21[2];       hz[1][1] = x22[2]
        hz[2][0] = dxdu11[2];    hz[2][1] = dxdu12[2]
        hz[3][0] = dxdu21[2];    hz[3][1] = dxdu22[2]
        hz[0][2] = dxdv11[2];    hz[0][3] = dxdv12[2]
        hz[1][2] = dxdv21[2];    hz[1][3] = dxdv22[2]
        hz[2][2] = 0;            hz[2][3] = 0
        hz[3][2] = 0;            hz[3][3] = 0

        ## H(u) and H(v)
        hu = [0] * 4
        huThx = [0] * 4
        huThy = [0] * 4
        huThz = [0] * 4
        hv = [0] * 4
        hu[0] = h1(u)
        hu[1] = h2(u)
        hu[2] = h3(u)
        hu[3] = h4(u)
        hv[0] = h1(v)
        hv[1] = h2(v)
        hv[2] = h3(v)
        hv[3] = h4(v)

        ## Perform A = H(u)' * H_c
        huThx[0] = hu[0] * hx[0][0] + hu[1] * hx[1][0] + hu[2] * hx[2][0] + hu[3] * hx[3][0]
        huThx[1] = hu[0] * hx[0][1] + hu[1] * hx[1][1] + hu[2] * hx[2][1] + hu[3] * hx[3][1]
        huThx[2] = hu[0] * hx[0][2] + hu[1] * hx[1][2] + hu[2] * hx[2][2] + hu[3] * hx[3][2]
        huThx[3] = hu[0] * hx[0][3] + hu[1] * hx[1][3] + hu[2] * hx[2][3] + hu[3] * hx[3][3]

        huThy[0] = hu[0] * hy[0][0] + hu[1] * hy[1][0] + hu[2] * hy[2][0] + hu[3] * hy[3][0]
        huThy[1] = hu[0] * hy[0][1] + hu[1] * hy[1][1] + hu[2] * hy[2][1] + hu[3] * hy[3][1]
        huThy[2] = hu[0] * hy[0][2] + hu[1] * hy[1][2] + hu[2] * hy[2][2] + hu[3] * hy[3][2]
        huThy[3] = hu[0] * hy[0][3] + hu[1] * hy[1][3] + hu[2] * hy[2][3] + hu[3] * hy[3][3]

        huThz[0] = hu[0] * hz[0][0] + hu[1] * hz[1][0] + hu[2] * hz[2][0] + hu[3] * hz[3][0]
        huThz[1] = hu[0] * hz[0][1] + hu[1] * hz[1][1] + hu[2] * hz[2][1] + hu[3] * hz[3][1]
        huThz[2] = hu[0] * hz[0][2] + hu[1] * hz[1][2] + hu[2] * hz[2][2] + hu[3] * hz[3][2]
        huThz[3] = hu[0] * hz[0][3] + hu[1] * hz[1][3] + hu[2] * hz[2][3] + hu[3] * hz[3][3]

        ## Perform A * H(v), i.e., H(u)' * H_c * H(v)
        output = [0] * 3
        output[0] = huThx[0] * hv[0] + huThx[1] * hv[1] + huThx[2] * hv[2]
        output[1] = huThy[0] * hv[0] + huThy[1] * hv[1] + huThy[2] * hv[2]
        output[2] = huThz[0] * hv[0] + huThz[1] * hv[1] + huThz[2] * hv[2]

        return output
    
    def _finite_difference_2nd_derivative(self, spoke, ref_spoke):
        """Approximate 2nd derivative of spokes' directions """
        curr_ri, curr_ci = spoke.coords
        curr_dir = spoke.U
        ref_ri, ref_ci = ref_spoke.coords
        Uvv = [0] * 3
        total_ci = len(self.interp_circular_dirs)
        total_ri = self.interp_circular_dirs[0].shape[0]
        if curr_ri == ref_ri:
            ## target and reference are on the same radial line, then compute derivative along this radial line
            if curr_ci + 2 >= total_ci:
                ## second order backward 
                Uvv = curr_dir - 2 * self._get_interpolated_dirs(curr_ri, curr_ci - 1) + self._get_interpolated_dirs(curr_ri, curr_ci - 2)
            elif curr_ci < 2:
                ## second order forward
                Uvv = self._get_interpolated_dirs(curr_ri, curr_ci + 2) - 2 * self._get_interpolated_dirs(curr_ri, curr_ci + 1) + curr_dir
            else:
                ## second order central
                Uvv = self._get_interpolated_dirs(curr_ri, curr_ci + 1) + self._get_interpolated_dirs(curr_ri, curr_ci - 1) - 2 * curr_dir
        elif curr_ci == ref_ci:
            ## Compute 2nd derivative along the circular direction using circular central difference
            Uvv = self._get_interpolated_dirs((curr_ri + 1)%total_ri, curr_ci) + self._get_interpolated_dirs((curr_ri -1) % total_ri, curr_ci) - 2 * curr_dir
        else:
            assert False, "Not valid pair of target and reference spokes for computing 2nd derivatives."
        return Uvv

    def _interpolate_middle(self, spoke_start, spoke_end, dist_from_start):
        """
        Interpolate radii of middle spoke between spoke_start and
        spoke_end, which distant from spoke_start dist_from_start
        This function looks up the directions of spokes being interpolated according to (total_ri, total_ci)
        """
        ## 1. 2nd derivatives at two ends
        assert(isinstance(spoke_start,Spoke) and isinstance(spoke_end, Spoke))
        Uvv_end = self._finite_difference_2nd_derivative(spoke_end, spoke_start)
        Uvv_start = self._finite_difference_2nd_derivative(spoke_start, spoke_end)

        ## 2. compute the middle spoke
        # sum_rU = spoke_start.add(spoke_end)
        # avg_rU = sum_rU / 2
        avg_rU = (spoke_start.r + spoke_end.r) / 2
        
        ## 3. compute the direction of the middle spoke
        half_dist = dist_from_start / 2
        # if start_dir == end_dir set middle_dir to the same
        start_dir = spoke_start.U
        end_dir = spoke_end.U
        start_ri, start_ci = spoke_start.coords
        end_ri, end_ci = spoke_end.coords
        if np.linalg.norm(start_dir - end_dir) < epsilon:
            middle_dir = start_dir
            Uvv_start = np.zeros_like(Uvv_start)
            Uvv_end = np.zeros_like(Uvv_end)
        else:
            middle_dir = self._get_interpolated_dirs((start_ri + end_ri)//2, (start_ci+end_ci) // 2)

        ## 4. compute the radius of the middle spoke
        inner_prod1 = np.dot(middle_dir, avg_rU)
        inner_prod2 = np.dot(start_dir, Uvv_start)
        inner_prod3 = np.dot(end_dir, Uvv_end)
        middle_r = avg_rU #- half_dist ** 2 * 0.25 * (inner_prod2 + inner_prod3)

        middle_coords = (start_ri + end_ri)//2, (start_ci+end_ci) // 2
        middle_spoke = Spoke(middle_r, middle_dir, None, middle_coords)
        # middle_spoke.p = (spoke_start.p + spoke_end.p)/2
        # p= pv.Plotter()
        # p.add_mesh(spoke_start.visualize(), line_width=4, color='red')
        # p.add_mesh(spoke_end.visualize(), line_width=4, color='yellow')
        # p.add_mesh(middle_spoke.visualize(), line_width=4, color='green')
        # p.show()
        return middle_spoke
    def _interpolate_quad(self, relative_position, corner_spokes, p_lambda):
        """
        Interpolate the center of a quad surrounded by 4 corner_spokes to approach the relative_position.
        The relative_position (type: float) specifies the target position want to interpolate,
        while p_lambda (2 ^ k, k <= 0) indicates how many subdivisions have been through, initialy 1
        """
        sp11, sp12, sp21, sp22 = corner_spokes
        u, v = relative_position # ri, ci
        ### 1. interpolate center positions on edges
        top_middle_spoke   = self._interpolate_middle(sp11, sp12, p_lambda)
        left_middle_spoke  = self._interpolate_middle(sp11, sp21, p_lambda)
        bot_middle_spoke   = self._interpolate_middle(sp21, sp22, p_lambda)
        right_middle_spoke = self._interpolate_middle(sp22, sp12, p_lambda)

        ### 2. interpolate center of the quad
        vertical_center = self._interpolate_middle(top_middle_spoke, bot_middle_spoke, p_lambda)
        horizont_center = self._interpolate_middle(left_middle_spoke, right_middle_spoke, p_lambda)

        ### 3. average vertical_center and horizont_center to get a better estimation of center
        assert(not vertical_center.isnan())
        assert(not horizont_center.isnan())

        ri_11, ci_11 = sp11.coords
        ri_22, ci_22 = sp22.coords
        quad_center = vertical_center.avg(horizont_center)
        quad_center.coords = (ri_11 + ri_22)//2, (ci_11 + ci_22) // 2

        ### 4. solve the spoke at relative_position if close. Subdivide otherwise.
        half_dist = 0.5 * p_lambda

        if abs(u - half_dist) <= epsilon and abs(v-half_dist) <= epsilon:
            # close the quad center
            result_spoke = quad_center
        elif abs(u) <= epsilon and abs(v) <= epsilon:
            # close to the left top corner
            result_spoke = sp11
        elif abs(u - p_lambda) < epsilon and abs(v) < epsilon:
            # close to the left bot corner
            result_spoke = sp21
        elif abs(v - p_lambda) < epsilon and abs(u) < epsilon:
            # close to the right top corner
            result_spoke = sp12
        elif abs(v - p_lambda) < epsilon and abs(u - p_lambda) < epsilon:
            # right bot 
            result_spoke = sp22
        elif abs(u - half_dist) <= epsilon and abs(v) < epsilon:
            # left_middle_spoke
            result_spoke = left_middle_spoke
        elif abs(u) < epsilon and abs(v - half_dist) < epsilon:
            # top_middle_spoke
            result_spoke = top_middle_spoke
        elif abs(u - half_dist) < epsilon and abs(v - p_lambda) < epsilon:
            # right_middle_spoke
            result_spoke = right_middle_spoke
        elif abs(u - p_lambda) < epsilon and abs(v - half_dist) < epsilon:
            # bot_middle_spoke
            result_spoke = bot_middle_spoke
        else:
            # subdivide to approach (u, v)
            interpolated_spokes = \
                top_middle_spoke, right_middle_spoke, bot_middle_spoke, left_middle_spoke, quad_center
            result_spoke = self._new_quad_interpolation(relative_position, corner_spokes, interpolated_spokes, half_dist, p_lambda)
        return result_spoke

    def _new_quad_interpolation(self, relative_position, prime_spokes, interpolated_spokes, half_dist, p_lambda):
        new_corner = None
        u, v = relative_position # ri, ci
        sp11, sp12, sp21, sp22 = prime_spokes
        top_middle_spoke, right_middle_spoke, bot_middle_spoke, left_middle_spoke, center = \
                                            interpolated_spokes
        if u < half_dist and v > half_dist:
            ## bot left quad # 11, 12, 21, 22
            new_corner = left_middle_spoke, center, sp21, bot_middle_spoke 
            new_relative_position = u, v - half_dist
            return self._interpolate_quad(new_relative_position, new_corner, p_lambda / 2)
        elif u < half_dist and v < half_dist:
            ## top left quad
            new_corner = sp11, top_middle_spoke, left_middle_spoke, center
            return self._interpolate_quad(relative_position, new_corner, p_lambda / 2)
        elif u > half_dist and v < half_dist:
            # top right quad
            new_corner = top_middle_spoke, sp12, center, right_middle_spoke
            new_relative_position = u - half_dist, v
            return self._interpolate_quad(new_relative_position, new_corner, p_lambda / 2)
        elif u > half_dist and v > half_dist:
            # bot right quad
            new_corner = center, right_middle_spoke, bot_middle_spoke, sp22
            new_relative_position = u - half_dist, v - half_dist
            return self._interpolate_quad(new_relative_position, new_corner, p_lambda / 2)
        else:
            # interpolate on a line segment, subdivide the segment
            if abs(v - half_dist) < epsilon:
                new_corner = top_middle_spoke, bot_middle_spoke
                new_relative_position = u
                return self._interpolate_segment(new_relative_position, new_corner, 1, is_horizontal=False)
            elif abs(u - half_dist) < epsilon:
                new_corner = left_middle_spoke, right_middle_spoke
                new_relative_position = v
                return self._interpolate_segment(new_relative_position, new_corner, 1, is_horizontal=True)

        return new_corner
    def _interpolate_segment(self, dist, end_spokes, p_lambda, is_horizontal):
        """
        p_lambda represents the total distance from start to end of the segment
        dist is the target position of the desired interpolation
        """
        start_spoke, end_spoke = end_spokes
        middle_spoke = self._interpolate_middle(start_spoke, end_spoke, p_lambda)

        half_dist = p_lambda / 2
        if abs(dist - half_dist) < epsilon:
            result_spoke = middle_spoke
        elif dist < half_dist:
            new_ends = start_spoke, middle_spoke
            result_spoke = self._interpolate_segment(dist, new_ends, half_dist, is_horizontal)
        elif dist > half_dist:
            new_ends = middle_spoke, end_spoke
            result_spoke = self._interpolate_segment(dist-half_dist, new_ends, half_dist, is_horizontal)
        return result_spoke
    
    def _get_spoke_lens(self, srep, num_crest_points):
        """Return spokes' length array"""
        ret_array = []
        # the number of smooth point in both sides
        num_smooth_skeletal_pts = num_crest_points*3*2
        start, end = 0, num_smooth_skeletal_pts
        if not self.interpolate_up:
            start, end = start + num_smooth_skeletal_pts, end + num_smooth_skeletal_pts
        for i in range(start, end, 2):
            spoke = np.array(srep.GetPoint(i+1)) - np.array(srep.GetPoint(i))
            spoke_len = np.linalg.norm(spoke)

            ret_array.append(spoke_len)
        return ret_array
    def _get_spoke_dirs(self, srep, num_crest_points=24):
        """Return spokes' directions array"""
        ret_array = []
        # the number of smooth point in both sides
        num_smooth_skeletal_pts = num_crest_points*3*2
        start, end = 0, num_smooth_skeletal_pts
        if not self.interpolate_up:
            start, end = start + num_smooth_skeletal_pts, end + num_smooth_skeletal_pts
        for i in range(start, end, 2):
            spoke = np.array(srep.GetPoint(i+1)) - np.array(srep.GetPoint(i))
            spoke_len = np.linalg.norm(spoke)
            spoke_dir = spoke / spoke_len
            ret_array.append(spoke_dir)
        return ret_array
    def _get_base_points_array(self, srep, num_crest_points):
        """Return the base points of spokes given by srep"""
        ret_base_pts = []
        # the number of smooth point in both sides
        num_smooth_skeletal_pts = num_crest_points*3*2
        start, end = 0, num_smooth_skeletal_pts
        if not self.interpolate_up:
            start, end = start + num_smooth_skeletal_pts, end + num_smooth_skeletal_pts
        for i in range(start, end, 2):
            ret_base_pts.append(np.array(srep.GetPoint(i)))
        return ret_base_pts
    
    def no_interpolate(self, input_srep):
        ret_spokes = []
        for i in range(0, input_srep.GetNumberOfPoints(), 2):
            base_pt = np.array(input_srep.GetPoint(i))
            bdry_pt = np.array(input_srep.GetPoint(i+1))
            ret_spokes.append(Spoke(base_pt=base_pt, bdry_pt=bdry_pt))
        return ret_spokes
    def _interpolate_dir_on_radial_line(self, input_srep, interpolate_level, num_crest_points, num_samples_outward):
        """Interpolate directions of spokes that are based on radial lines of a skeleton."""
        # num_crest_points = num_rows * 2 + (num_cols - 2) * 2
        num_steps = np.power(2, interpolate_level)
        # num_samples_outward = 1 + num_rows // 2
        step_size = 1.0 / num_steps
        dirs_da = self._get_spoke_dirs(input_srep, num_crest_points)
        pts = self._get_base_points_array(input_srep, num_crest_points)
        # k x n x 3, where k is the number of radial lines, n is the number of interpolated spokes on each radial line
        # Particularly , n = (num_samples_outward - 1) * (num_steps - 1) + num_samples_outward
        ret_dirs = [] 
        
        for r in range(num_crest_points):
            # p = pv.Plotter()
            # p.add_mesh(input_srep, color='white', line_width=2)
        
            key_times = []
            primary_spokes_existed = []
            for c in range(num_samples_outward):
                key_times.append(c)
                primary_spokes_existed.append(dirs_da[r * num_samples_outward + c])
                # p.add_mesh(pv.Arrow(pts[r*num_samples_outward+c], dirs_da[r * num_samples_outward + c], scale=3), color='red')
            
            key_rots = R.from_rotvec(np.array(primary_spokes_existed))
            
            slerp = Slerp(key_times, key_rots)
            interp_times = np.arange(0, num_samples_outward + step_size - 1, step_size)
            interp_radial_rots = slerp(interp_times).as_rotvec()
            ret_dirs.append(interp_radial_rots)
            # p.show()
        return np.array(ret_dirs)
    def _get_interpolated_dirs(self, total_ri, total_ci):
        """ Find the interpolated direction at the position (total_ri, total_ci)
        Both total_ri and total_ci are the absolute coordinates in (u, v) coordinate system.

        """
        return self.interp_circular_dirs[total_ci][total_ri, :]

    def _interpolate_dirs(self, interp_radial_dirs, num_crest_points, num_samples_outward, num_steps):
        """Interpolate all spoke directions according to the interpolate level (or num_steps).
        The input interp_radial_dirs are the directions of existing primary spokes PLUS the directions of spokes based on the radial lines of the skeleton.
        Return directions of interpolated spokes, whose base points are not yet determined in this function.

        """
        self.interp_circular_dirs = collections.defaultdict(list)
        step_size = 1.0 / num_steps
        for r in range(num_crest_points):
            for c in range(num_samples_outward -1):
                # for ri in range(num_steps + 1):
                for ci in range(num_steps + 1):
                    total_ci = c * num_steps + ci
                    # total_ri = r * num_steps + ri
                    control_directions = np.concatenate((interp_radial_dirs[:, total_ci, :], interp_radial_dirs[0:1, total_ci, :]))
                        
                    key_times = [i for i in range(control_directions.shape[0])]
                    key_rots = R.from_rotvec(control_directions)
                    
                    slerp = Slerp(key_times, key_rots)
                    interp_times = np.arange(0, num_crest_points + step_size, step_size)
                    interp_radial_rots = slerp(interp_times).as_rotvec()
                    
                    self.interp_circular_dirs[total_ci] = interp_radial_rots
    def _interp_along_spline(self, pts, num_steps):
        """Interpolate crest points along a spline defined by pts of dimension (n x 3)"""
        radial_spline = vtk.vtkParametricSpline()
            
        spline_pts = vtk.vtkPoints()
        for pt in pts:
            spline_pts.InsertNextPoint(pt)
        radial_spline.SetPoints(spline_pts)
        interp_crest_pts = []
        for k in range(num_steps+1):
            interp_u = [1/ num_steps * k] * 3
            pt = [0] * 3
            radial_spline.Evaluate(interp_u, pt,[0]*9)
            interp_crest_pts.append(pt)
        # func_src = vtk.vtkParametricFunctionSource()
        # func_src.SetParametricFunction(radial_spline)
        # func_src.Update()
        return interp_crest_pts
    
    def interpolate_crest(self, input_srep, top_spokes, bot_spokes, num_crest_points):
        """
        Interpolate spokes around the crest region. 
        Input input_srep is the primary spokes, including top, bot and crest spokes
        Input up_spokes and down_spokes are results from the method self.interpolate
        Input num_crest_points is the number of the primary crest spokes

        """
        num_steps = np.power(2, self.interpolate_level)
        num_interp_crest_pts = len(top_spokes)
        primary_crest_start_idx = num_crest_points * 3 * 2 * 2
        prim_crest_pts = vtk.vtkPoints()
        prim_fold_pts = vtk.vtkPoints()
        for i in range(0, num_interp_crest_pts, num_steps):
            prim_crest_base_pt = np.array(input_srep.GetPoint(primary_crest_start_idx + (i//num_steps) * 2))
            prim_crest_bdry_pt = np.array(input_srep.GetPoint(primary_crest_start_idx + (i//num_steps) * 2 + 1))
            prim_crest_pts.InsertNextPoint(prim_crest_bdry_pt)
            prim_fold_pts.InsertNextPoint(prim_crest_base_pt)
            
        tps = vtk.vtkThinPlateSplineTransform()
        tps.SetSourceLandmarks(prim_fold_pts)
        tps.SetTargetLandmarks(prim_crest_pts)
        tps.SetBasisToR()
        tps.Modified()
        ## interpolate along the crest curve
        crest_pts = []
        p = pv.Plotter()
        top_crest_pts = []
        bot_crest_pts = []
        top_crest_spokes, bot_crest_spokes = [], []
        p = pv.Plotter()
        p.add_mesh(input_srep, line_width=2, color='white', label='Input')
        for i in range(num_interp_crest_pts):
            crest_bdry_pt = tps.TransformPoint(top_spokes[i][-1].p)
            crest_pts.append(crest_bdry_pt)

            ## interpolate crest points along primary crest points
            top_control_pts = [top_spokes[i][-1].getB(), crest_bdry_pt]
            top_interp_crest_pts = self._interp_along_spline(top_control_pts, num_steps)
            bot_control_pts = [bot_spokes[i][-1].getB(), crest_bdry_pt]
            bot_interp_crest_pts = self._interp_along_spline(bot_control_pts, num_steps)
            top_crest_pts += top_interp_crest_pts
            bot_crest_pts += bot_interp_crest_pts

            for j in range(len(top_interp_crest_pts)):
                top_spoke = Spoke(base_pt=top_spokes[i][-1].p, bdry_pt=top_interp_crest_pts[j])
                top_crest_spokes.append(top_spoke)
                p.add_mesh(top_spoke.visualize(), color='orange', line_width=4)

                bot_spoke = Spoke(base_pt=bot_spokes[i][-1].p, bdry_pt=bot_interp_crest_pts[j])
                bot_crest_spokes.append(bot_spoke)
                p.add_mesh(bot_spoke.visualize(), color='blue', line_width=4)
                

            ######## The following visualization helps debugging
            # p = pv.Plotter()
            # p.add_mesh(input_srep, line_width=2, color='white', label='Input')
            
            p.add_mesh(pv.PolyData(np.array(top_interp_crest_pts)), color='orange', label='Interp', point_size=10)
            p.add_mesh(pv.PolyData(np.array(bot_interp_crest_pts)), color='blue', label='Interp', point_size=10)
            
        # p.add_mesh(pv.Spline(np.array(crest_pts + [crest_pts[0]])), color='cyan', label='Crest spline', line_width=4)
        # p.add_mesh(pv.PolyData(np.array(crest_pts)), color='cyan', label='Crest pts', point_size=20)
        # p.add_legend()
        # p.show()
        # p.add_mesh(input_srep, line_width=2, color='white', label='Input')   
        # p.add_mesh(pv.Spline(np.array(crest_pts + [crest_pts[0]])), color='cyan', label='Crest spline', line_width=4)
        # p.add_mesh(pv.PolyData(np.array(crest_pts)), color='cyan', label='Crest pts', point_size=20)
        # p.add_legend()
        # p.show()
            #################
        # crest_poly = self._connect_crest_poly(top_crest_pts, bot_crest_pts, num_steps)
        # p.add_mesh(crest_poly,  color='cyan', label='crest')
        p.show()
        return top_crest_spokes + bot_crest_spokes
    def _connect_crest_poly(self, top_crest_pts, bot_crest_pts, num_steps):
        top_crest_poly = vtk.vtkPolyData()
        top_crest_ps = vtk.vtkPoints()
        top_crest_con = vtk.vtkCellArray()

        bot_crest_poly = vtk.vtkPolyData()
        bot_crest_ps = vtk.vtkPoints()
        bot_crest_con = vtk.vtkCellArray()
        def new_triangle(id_curr_pt, id_pt_below, id_pt_right, id_pt_bot_right):
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, id_curr_pt)
            triangle.GetPointIds().SetId(1, id_pt_below)
            triangle.GetPointIds().SetId(2, id_pt_right)

            triangle2 = vtk.vtkTriangle()
            triangle2.GetPointIds().SetId(0, id_pt_right)
            triangle2.GetPointIds().SetId(1, id_pt_below)
            triangle2.GetPointIds().SetId(2, id_pt_bot_right)
            return triangle, triangle2

        for i in range(len(top_crest_pts)):
            id_curr_pt = top_crest_ps.InsertNextPoint(top_crest_pts[i])
            
            bot_crest_ps.InsertNextPoint(bot_crest_pts[i])
            if i == 0 or (i+1) % (num_steps+1) != 0:
                id_pt_below = (id_curr_pt + 1)% len(top_crest_pts)
                id_pt_right = (id_curr_pt + num_steps + 1) % len(top_crest_pts)
                id_pt_bot_right = (id_pt_right+1) % len(top_crest_pts)
                top_tri1, top_tri2 = new_triangle(id_curr_pt, id_pt_below, id_pt_right, id_pt_bot_right)
                bot_tri1, bot_tri2 = new_triangle(id_curr_pt, id_pt_below, id_pt_right, id_pt_bot_right)
                
                top_crest_con.InsertNextCell(top_tri1)
                bot_crest_con.InsertNextCell(bot_tri1)

                top_crest_con.InsertNextCell(top_tri2)
                bot_crest_con.InsertNextCell(bot_tri2)
                
        top_crest_poly.SetPoints(top_crest_ps)
        bot_crest_poly.SetPoints(bot_crest_ps)
        top_crest_poly.SetPolys(top_crest_con)
        bot_crest_poly.SetPolys(bot_crest_con)
        top_crest_poly.Modified()
                
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(top_crest_poly)
        appender.AddInputData(bot_crest_poly)
        appender.Update()
        return appender.GetOutput()


    def interpolate(self, input_srep, num_crest_points, num_samples_outward):
        """
        main entry of interpolation
        """
        print("Interpolate an s-rep with interpolation_level = " + str(self.interpolate_level))
        interpolate_level = self.interpolate_level
        if interpolate_level == 0:
            return self.no_interpolate(input_srep)
        # steps of interpolation
        num_steps = np.power(2, interpolate_level)
        step_size = 1.0 / num_steps

        interpolated_spokes = vtk.vtkAppendPolyData()
        num_interp_each_radial_line = (num_steps - 1) * (num_samples_outward - 1) + num_samples_outward
        ## Dimension num_crest_point x num_interp_each_radial_line x 3
        interp_radial_dirs = self._interpolate_dir_on_radial_line(input_srep, interpolate_level, num_crest_points, num_samples_outward)
        self._interpolate_dirs(interp_radial_dirs, num_crest_points, num_samples_outward, num_steps)
        ## save outmost interior spokes  along radial lines for interpolating crest spokes
        ret_spokes = collections.defaultdict(list)
        # p = pv.Plotter()
        # p.add_mesh(input_srep, color='white')
        for r in range(num_crest_points):
            for c in range(num_samples_outward -1):
                next_row = r + 1 # next radial line index
                # compute positional derivatives for 4 corners
                if r == num_crest_points - 1:
                    dxdu11, dxdv11 = self._compute_derivative(input_srep, r, c,   num_crest_points, num_samples_outward)
                    dxdu21, dxdv21 = self._compute_derivative(input_srep, 0, c,   num_crest_points, num_samples_outward)
                    dxdu12, dxdv12 = self._compute_derivative(input_srep, r, c+1, num_crest_points, num_samples_outward)
                    dxdu22, dxdv22 = self._compute_derivative(input_srep, 0, c+1, num_crest_points, num_samples_outward)
                    next_row = 0

                else:
                    dxdu11, dxdv11 = self._compute_derivative(input_srep, r, c,    num_crest_points, num_samples_outward)
                    dxdu21, dxdv21 = self._compute_derivative(input_srep, r+1, c,  num_crest_points, num_samples_outward)
                    dxdu12, dxdv12 = self._compute_derivative(input_srep, r, c+1,  num_crest_points, num_samples_outward)
                    dxdu22, dxdv22 = self._compute_derivative(input_srep, r+1, c+1,num_crest_points, num_samples_outward)

                corner_deriv = dxdu11, dxdv11, dxdu21, dxdv21, dxdu12, dxdv12, dxdu22, dxdv22
                
                # Form 4 corners of interpolation quad
                idx11 = r * num_samples_outward + c
                idx21 = next_row * num_samples_outward + c
                idx22 = next_row * num_samples_outward + (c+1)
                idx12 = r * num_samples_outward + (c+1)

                radii_da = self._get_spoke_lens(input_srep, num_crest_points)
                dirs_da  = self._get_spoke_dirs(input_srep, num_crest_points)
                r0 = radii_da[idx11]
                r1 = radii_da[idx21]
                r2 = radii_da[idx22]
                r3 = radii_da[idx12]

                u11 = dirs_da[idx11]
                u21 = dirs_da[idx21]
                u22 = dirs_da[idx22]
                u12 = dirs_da[idx12]

                ### Note that input_srep stores base points and bdry points for spokes
                base_pts_array = self._get_base_points_array(input_srep, num_crest_points)
                pt11 = pt12 = pt21 = pt22 = [0] * 3
                pt11 = base_pts_array[idx11]
                pt21 = base_pts_array[idx21]
                pt22 = base_pts_array[idx22]
                pt12 = base_pts_array[idx12]
                corner_pts = pt11, pt21, pt22, pt12

                absolute_uv11 = np.array([r, c]) * num_steps
                absolute_uv21 = np.array([next_row, c]) * num_steps
                absolute_uv22 = np.array([next_row, c+1]) * num_steps
                absolute_uv12 = np.array([r, c+1]) * num_steps
                sp11 = Spoke(r0, u11, pt11, absolute_uv11)
                sp21 = Spoke(r1, u21, pt21, absolute_uv21)
                sp22 = Spoke(r2, u22, pt22, absolute_uv22)
                sp12 = Spoke(r3, u12, pt12, absolute_uv12)
                corner_spokes = sp11, sp12, sp21, sp22

                ######## The following visualization helps debugging
                # if r != num_crest_points - 1: continue
                
                
                # p.add_mesh(sp11.visualize(), line_width=4, color='red')
                # p.add_mesh(sp12.visualize(), line_width=4, color='yellow')
                # p.add_mesh(sp21.visualize(), line_width=4, color='blue')
                # p.add_mesh(sp22.visualize(), line_width=4, color='green')
                # p.add_mesh(pv.Sphere(radius=0.1, center=np.array(pt11)), color='red')
                # p.add_mesh(pv.Sphere(radius=0.1, center=np.array(pt12)), color='yellow')
                # p.add_mesh(pv.Sphere(radius=0.1, center=np.array(pt21)), color='blue')
                # p.add_mesh(pv.Sphere(radius=0.1, center=np.array(pt22)), color='green')
                ##########################
                for ri in range(num_steps + 1):
                    for ci in range(num_steps + 1):
                        if c > 0 and ci == 0:
                            ## To avoid redundent interpolation on the boundary of quads
                            continue
                        if r > 0 and ri == 0:
                            ## To avoid redundent interpolation on the boundary of quads
                            continue
                        if r == num_crest_points - 1 and ri == num_steps:
                            ## To avoid redundent interpolation at the circular starting radial line
                            continue
                        if r > num_crest_points//2 and c == 0 and ci == 0:
                            ## To avoid redundent interpolation on the spine
                            continue
                        
                        total_ci = c * num_steps + ci
                        total_ri = r * num_steps + ri
                        
                        relative_position = ri * step_size, ci * step_size
                        
                        ## Interpolate spoke base points
                        interp_skeletal_pt = self._interpolate_skeleton(relative_position, corner_pts, corner_deriv)
                        
                        ## Look up the pre-interpolated directions
                        interpolated_dir = self._get_interpolated_dirs(total_ri, total_ci)
                        # if ri in [0, num_steps]:
                        #     p.add_mesh(pv.Arrow(interp_skeletal_pt, interpolated_dir, scale=2), color='red', label='tau_1')
                        # elif ci in [0, num_steps] and c == 1:
                        #     p.add_mesh(pv.Arrow(interp_skeletal_pt, interpolated_dir, scale=2), color='cornflowerblue', label='theta')

                        ## interpolate spoke radii by a successive subdivision
                        interpolated_spoke = self._interpolate_quad(relative_position, corner_spokes, 1.0)
                        interpolated_spoke.p = interp_skeletal_pt
                        interpolated_spoke.U = interpolated_dir
                        interpolated_spokes.AddInputData(interpolated_spoke.visualize())
                        interpolated_spokes.Update()

                        ## save outmost interior spokes and middle spokes along radial lines for interpolating crest spokes
                        # if total_ci == num_interp_each_radial_line - 1 or ci == num_steps // 2 or ci == num_steps:
                        ret_spokes[total_ri].append(interpolated_spoke)
                        
        #     p.add_mesh(interpolated_spokes.GetOutput(), color='red', line_width=4)
        
        # p.show()
                
        return interpolated_spokes.GetOutput(), ret_spokes
    
