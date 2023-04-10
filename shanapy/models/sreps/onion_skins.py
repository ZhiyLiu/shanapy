"""This is a geometric model for the interior geometry."""
import vtk
import numpy as np
import pyvista as pv
class Onion:
    def __init__(self, num_steps, tau_values=[0.2, 0.4, 0.6, 0.8, 1], num_fold_pts = 24):
        """A onion skin is a level surface of radial distance"""
        ## factions of spoke lengths that define the level surface of radial distance
        self.levels = tau_values
        self.num_crest_pts = num_fold_pts
        # interior skeletal points, assuming sample 3 points along tau_1 direction
        self.num_smooth_skel_pts = num_fold_pts * 3
        ## Interpolated spokes within each quad
        self.num_steps = num_steps

    def _get_implied_pts(self, spokes, level):
        """Return points along spokes given the fraction (level) of spokes' lengths"""
        ret_pts = []
        for spoke in spokes:
            bdry_pt = np.array(spoke.p + level * spoke.r * spoke.U)
            ret_pts.append(bdry_pt)
        return ret_pts
    
    def _get_implied_surface(self, implied_pts):
        """Return onion skins (level surfaces according to the given levels) of the s-rep
        Input: spokes of a list of Spkes: it is essentially a set of spokes, 
        including vtkPoints(base_point, end_point) and vtkLine poly connecting them.
        The ordering of the spokes in a set is using parameters of (tau_1, theta). 
        When tau_1 = 0 and theta = 0, the corresponding (first) spoke emanates from a spine end.
        The second spoke is (tau_1=1, theta=0), corresponding the skeletal point on the extension of the spine, so on so forth
        
        The set of spokes in the input is concatenation of top, bottom and crest spokes.
        """
        top_crest_pts = implied_pts[-2*self.num_crest_pts:-self.num_crest_pts]
        bot_crest_pts = implied_pts[-self.num_crest_pts:]
        smooth_pts = implied_pts[:-2*self.num_crest_pts]
        top_smooth_pts = smooth_pts[:len(smooth_pts)//2]
        bot_smooth_pts = smooth_pts[len(smooth_pts)//2:]
        crest_poly = self._connect_poly(top_crest_pts, bot_crest_pts, self.num_steps)
        smooth_poly = self._connect_poly(top_smooth_pts, bot_smooth_pts, self.num_steps *2)
        ########## Debug code for visualization
        # p = pv.Plotter()
        # p.add_mesh(pv.PolyData(top_crest_pts), label='Crest', color='cyan', point_size=10)
        # p.add_mesh(pv.PolyData(bot_crest_pts), label='Crest', color='green', point_size=10)
        # p.add_mesh(pv.PolyData(top_smooth_pts),label='Top',  color='red', point_size=10)
        # p.add_mesh(pv.PolyData(bot_smooth_pts),label='Bot',  color='blue', point_size=10)
        # p.add_mesh(crest_poly, color='cyan', show_edges=True)
        # p.add_mesh(smooth_poly, color='orange', show_edges=True)
        # p.show()
        ################
        appender = vtk.vtkAppendPolyData()
        appender.AddInputData(smooth_poly)
        appender.AddInputData(crest_poly)
        appender.Update()
        return appender.GetOutput()

    def _connect_poly(self, top_crest_pts, bot_crest_pts, num_steps):
        """
        Connect the interpolated crest points into a renderable triangle mesh vtkPolyData.
        """
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
    def get_skins(self, spokes):
        """Return onion skins (level surfaces according to the given levels) of the s-rep
        Input: spokes of a list of Spkes: it is essentially a set of spokes, 
        including vtkPoints(base_point, end_point) and vtkLine poly connecting them.
        The ordering of the spokes in a set is using parameters of (tau_1, theta). 
        When tau_1 = 0 and theta = 0, the corresponding (first) spoke emanates from a spine end.
        The second spoke is (tau_1=1, theta=0), corresponding the skeletal point on the extension of the spine, so on so forth
        
        The set of spokes in the input is concatenation of top, bottom and crest spokes.
        """
        onion_skins = []
        for level in self.levels:
            implied_pts = self._get_implied_pts(spokes, level)
            skin = self._get_implied_surface(implied_pts)
            onion_skins.append(skin)
        return onion_skins



