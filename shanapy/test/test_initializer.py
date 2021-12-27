
"""This file is testing the initializer implemented in shanapy/models/sreps"""
import vtk
from shanapy.models.sreps import Initializer
from shanapy.visualization import SrepViewer

## Read the input surface mesh (produced by SPHARM-PDM)
reader = vtk.vtkPolyDataReader()
reader.SetFileName('shanapy/data/example_hippocampus.vtk')
reader.Update()
input_mesh = reader.GetOutput()

## Initialize an s-rep for the input mesh
initializer = Initializer()
srep = initializer.fit(input_mesh)

## Visualize the s-rep and the input mesh
viewer = SrepViewer()
viewer.view(srep, input_mesh)
print('Done')