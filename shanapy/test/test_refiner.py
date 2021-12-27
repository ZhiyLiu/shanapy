"""This file is testing the refiner implemented in shanapy/models/sreps"""
from shanapy.models.sreps.refiner import Refiner
import vtk
from shanapy.models.sreps import Initializer, refiner
from shanapy.visualization import SrepViewer

## Read the input surface mesh (produced by SPHARM-PDM)
reader = vtk.vtkPolyDataReader()
reader.SetFileName('shanapy/data/example_hippocampus.vtk')
reader.Update()
input_mesh = reader.GetOutput()

## Initialize an s-rep for the input mesh
initializer = Initializer()
srep = initializer.fit(input_mesh)

## Refine the above s-rep
refiner = Refiner()
refined_srep = refiner.refine(srep, input_mesh)

## Visualize the s-rep and the input mesh
viewer = SrepViewer()
viewer.view(srep, refined_srep, input_mesh)