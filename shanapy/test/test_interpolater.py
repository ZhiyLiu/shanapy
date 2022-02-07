import vtk
from shanapy.models.sreps import Initializer, Interpolater
import pyvista as pv

## Read the input surface mesh (produced by SPHARM-PDM)
reader = vtk.vtkPolyDataReader()
reader.SetFileName('data/example_hippocampus.vtk')
reader.Update()
input_mesh = reader.GetOutput()

## Initialize an s-rep for the input mesh
initializer = Initializer()
srep = initializer.fit(input_mesh)

interp = Interpolater()
interp_spokes = interp.interpolate(srep, 5, 9)

p = pv.Plotter()
p.add_mesh(input_mesh, color='white', opacity=0.3, label='Surface')
p.add_mesh(interp_spokes, color='cyan', line_width=3, label='Interpolated')
p.add_mesh(srep, color='red', line_width=4, label='Primary')
p.add_legend()
p.add_axes(box=True)
p.show()