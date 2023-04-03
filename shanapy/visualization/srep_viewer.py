import pyvista as pv
class SrepViewer:
    def __init__(self):
        """TODO: Allow users to config the params of visualization"""
        pass
    def srep_in_surface_mesh(self, srep, mesh):
        plt = pv.Plotter()
        plt.add_mesh(mesh, color='white', opacity=0.2)
        plt.add_mesh(srep)
        plt.show()
    def srep_with_fold_in_surface(self, fold_pts, refined_srep, mesh):
        plt = pv.Plotter()
        plt.add_mesh(mesh, color='white', opacity=0.2)
        plt.add_mesh(refined_srep, color='orange',line_width=2)
        fold_curve = pv.Spline(fold_pts, 1000)
        plt.add_mesh(fold_curve, line_width=4, color='cornflowerblue')

        plt.show()
