
"""
Run this script in the root directory, e.g., ~/shanapy/
"""
import vtk
from shanapy.models.sreps import Initializer
from shanapy.visualization import SrepViewer

## Read the input surface mesh (produced by SPHARM-PDM)
reader = vtk.vtkPolyDataReader()
# reader.SetFileName('data/example_hippocampus.vtk')
reader.SetFileName('/Users/zhiy/playpen/hippocampi/nick_hipp/neg_vtk/stx_noscale_830319_V06_t1w_RAI_Bias_label_pp_surf_tMeanSPHARM_procalign.vtk') # expected orientation
reader.Update()
input_mesh = reader.GetOutput()

## Initialize an s-rep for the input mesh
initializer = Initializer()
srep = initializer.fit(input_mesh)

## Visualize the s-rep and the input mesh
viewer = SrepViewer()
viewer.srep_in_surface_mesh(srep=srep, mesh=input_mesh)
print('Done')
