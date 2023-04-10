import vtk
from shanapy.models.sreps import Initializer, Interpolater, Onion
import pyvista as pv
import numpy as np
pv.set_plot_theme('document')
## Read the input surface mesh (produced by SPHARM-PDM)
reader = vtk.vtkPolyDataReader()
reader.SetFileName('data/example_caud.vtk')
reader.Update()
input_mesh = reader.GetOutput()

## Initialize an s-rep for the input mesh
initializer = Initializer()
srep = initializer.fit(input_mesh)

num_crest_pt = 24
num_samples_outward = 3

## Interpolate up spokes
interpolate_level = 1
interp = Interpolater(interpolate_level=interpolate_level)
interp_spokes, up_spokes = interp.interpolate(srep, num_crest_pt, num_samples_outward)

## Interpolate down spokes
interp.interpolate_up = False
interp_down_spokes, down_spokes = interp.interpolate(srep, num_crest_pt, num_samples_outward)

## interpolate fold spokes
crest_spokes = interp.interpolate_crest(srep, up_spokes, down_spokes, num_crest_pt)

num_steps = np.power(2, interpolate_level)
onion_skins = Onion(num_steps, num_fold_pts=len(crest_spokes)//2)
top_spokes, bot_spokes = [], []

symm_ids = (num_steps + 1) * 2
top_spine_spokes, bot_spine_spokes = [], []
for total_ri in up_spokes.keys():
    if len(up_spokes[total_ri]) != (num_steps - 1) * (num_samples_outward - 1)  + num_samples_outward:
        top_spokes += [up_spokes[total_ri - symm_ids][0]]
        bot_spokes += [down_spokes[total_ri-symm_ids][0]]
        symm_ids += 2
    top_spokes += up_spokes[total_ri]
    bot_spokes += down_spokes[total_ri]
    
interior_surfs = onion_skins.get_skins(top_spokes + bot_spokes + crest_spokes)

### visualize onion skins
colors = ['#063852', '#636466', '#1e9adf', '#ffc100', '#D3D3D3']
labels = ['0.2', '0.4', '0.6', '0.8', '1']
obj_polydata = pv.PolyData(input_mesh)
cube_center = [obj_polydata.center[0] + 6, obj_polydata.center[1] + 9, obj_polydata.center[2]]
cube = pv.Cube(center=(cube_center), x_length=10, y_length=20, z_length=12)
# cube.rotate_x(2)

p = pv.Plotter()
# p.add_mesh(cube, color='red', opacity=0.3, label='Cube')
for i, level_surf in enumerate(interior_surfs):
    clipped = pv.PolyData(level_surf).clip_box(cube)
    p.add_mesh(clipped, color=colors[i], label=labels[i])

p.add_legend(bcolor=(1, 1, 1), border=True)
p.add_axes(line_width=5)
p.show()