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

num_crest_pt = 24
num_samples_outward = 3

## Interpolate up spokes
interp = Interpolater(interpolate_level=3)
interp_spokes, up_spokes = interp.interpolate(srep, num_crest_pt, num_samples_outward)

## Interpolate down spokes
interp.interpolate_up = False
interp_down_spokes, bot_spokes = interp.interpolate(srep, num_crest_pt, num_samples_outward)

## interpolate fold spokes
crest_spokes = interp.interpolate_crest(srep, up_spokes, bot_spokes, num_crest_pt)

p = pv.Plotter()
p.add_mesh(input_mesh, color='white', opacity=0.3, label='Surface')
p.add_mesh(interp_spokes, color='orange', line_width=3, label='Interp Up')
# p.add_mesh(interp_down_spokes, color='cyan', line_width=3, label='Interp Down')
p.add_mesh(srep, color='red', line_width=4, label='Primary')
p.add_legend()
p.add_axes(box=True)
p.show()