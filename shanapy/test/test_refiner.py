"""This file is testing the refiner implemented in shanapy/models/sreps"""
from shanapy.models.sreps.refiner import Refiner
import vtk
from shanapy.models.sreps import Initializer, refiner
from shanapy.visualization import SrepViewer


## Read the input surface mesh (produced by SPHARM-PDM)
## Test data is located in ~/shanapy/data/
reader = vtk.vtkPolyDataReader()
reader.SetFileName('data/example_hippocampus.vtk')
reader.Update()
input_mesh = reader.GetOutput()

## Initialize an s-rep for the input mesh
num_crest_points=24
initializer = Initializer(num_crest_points)
srep = initializer.fit(input_mesh)

## Refine the above s-rep
refiner = Refiner()
refined_srep = refiner.refine(srep, input_mesh, num_crest_points)

## Collect fold points from refined srep to form a loop (i.e., fold curve)
fold_pts = []
num_pts = refined_srep.GetNumberOfPoints()
for fold_pt_id in range(num_pts - 1*2, num_pts - 25 * 2, -2):
    fold_pts.append(refined_srep.GetPoint(fold_pt_id))
fold_pts.append(fold_pts[0])

## Visualize the s-rep and the input mesh
viewer = SrepViewer()
viewer.view(fold_pts, refined_srep, input_mesh)