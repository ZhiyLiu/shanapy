import pyvista as pv
class SrepViewer:
    def __init__(self):
        """TODO: Allow users to config the params of visualization"""
        pass
    def view(self, srep, mesh):
        plt = pv.Plotter()
        plt.add_mesh(mesh, color='white', opacity=0.2)
        plt.add_mesh(srep)
        plt.show()

