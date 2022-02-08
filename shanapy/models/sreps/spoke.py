import vtk
import numpy as np
import math
class Spoke(object):
    def __init__(self, radius=None, direction=None,
                 base_pt=None, absolute_uv=None, bdry_pt=None):
        self.r = radius
        self.U = np.array(direction, dtype=np.float64)
        self.p = np.array(base_pt, dtype=np.float64)
        self.coords = absolute_uv # coords in (u, v) coords system, (offset_r, offset_c, r0, c0)
        self.ext = np.inf       # extension (i.e., boundary to the medial axis of between-object)
        self.link_dir = None

        ## This is predefined threshold. Only links that are smaller than this threshold are considered
        self.delta_min = 0.5 ## distance from z to z_prime, where z is the end of the extension of this spoke
        self.ext_pt = None
        
        if bdry_pt is not None:
            ## compute r, U from base_pt and bdry_pt
            assert base_pt is not None, "Need both the skeletal point and bdry point"
            s = np.array(bdry_pt) - base_pt
            self.r = np.linalg.norm(s)
            self.U = s / self.r
    def scale(self, scale=1):
        # if np.isinf(self.ext):
        #     return None
        if scale < 1: scale += 1
        return Spoke(scale * self.r, self.U, self.p)

    def add(self, another):
        assert(isinstance(another, Spoke))
        return self.r * self.U + another.r * another.U
    def isnan(self):
        if math.isnan(self.r) or np.any(np.isnan(self.U)):
           return True
        return False
    def getB(self):
        return self.p + self.r * self.U
    def extend(self, l):
        return self.p + self.U * (self.r + l)
    def avg(self, another):
        assert(isinstance(another, Spoke))
        avg_r = 0.5 * (self.r + another.r)
        avg_U = 0.5 * (self.U + another.U)
        avg_p = 0.5 * (self.p + another.p)
        return Spoke(avg_r, avg_U, avg_p, None)
    def visualize(self):
        spoke_polydata = vtk.vtkPolyData()
        spoke_pts = vtk.vtkPoints()
        spoke_lines = vtk.vtkCellArray()
        arrow = vtk.vtkLine()

        bdry_pt = self.p + self.r * self.U
        id0 = spoke_pts.InsertNextPoint(self.p)
        id1 = spoke_pts.InsertNextPoint(bdry_pt)
        arrow.GetPointIds().SetId(0, id0)
        arrow.GetPointIds().SetId(1, id1)
        spoke_lines.InsertNextCell(arrow)
        spoke_polydata.SetPoints(spoke_pts)
        spoke_polydata.SetLines(spoke_lines)

        # set color
        named_colors = vtk.vtkNamedColors()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.InsertNextTypedTuple(named_colors.GetColor3ub("Blue"))
        spoke_polydata.GetCellData().SetScalars(colors)

        return spoke_polydata

# s1 = Spoke(0.2, np.array([1, 1, 1]), np.array([2, 3, 4]))
# s2 = Spoke(1, np.array([5, 5, 5]), np.array([2, 3, 4]))
# sum_spoke = (s1.add(s2))
# print(sum_spoke / 2)