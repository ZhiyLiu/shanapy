import vtk
import os
import numpy as np
import pyvista as pv
from shanapy.models.sreps import Initializer, Refiner

## initialization + refinement of s-reps
root = '/path/to/population/mesh/vtks/'
class_name = ['pos', 'neg']
for class_label in class_name:
    mesh_folder = root + '/' + class_label + '_vtk/'
    os.chdir(root)
    for file_name in os.listdir(mesh_folder):
        if file_name.split('.')[-1] != 'vtk': continue
        print(class_label + " : " + file_name)
        file_path = mesh_folder + file_name
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_path)

        reader.Update()
        input_mesh = reader.GetOutput()

        ## Initialize an s-rep for the input mesh
        num_crest_points=24 # DON'T CHANGE THIS NUMBER FOR NOW
        initializer = Initializer(num_crest_points=num_crest_points)
        initial_srep = initializer.fit(input_mesh)

        output_file_name = file_name.split('.')[0]
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(root + "/" + class_label + "_initial_sreps/" + output_file_name + "_init.vtk")
        writer.SetInputData(initial_srep)
        writer.Update()

        # refiner = Refiner(input_mesh)
        # refined_srep = refiner.refine(initial_srep, num_crest_points)
        # writer.SetFileName(root + "/" + class_label + "_sreps/" + output_file_name + ".vtk")
        # writer.SetInputData(refined_srep)
        # writer.Update()