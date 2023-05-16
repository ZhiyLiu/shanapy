import vtk
import os
import numpy as np
import pyvista as pv
from shanapy.models.sreps import Initializer
from shanapy.visualization import SrepViewer
from shanapy.models.stats import SrepFeatures
from sklearn.decomposition import PCA
import plotly.express as px
# distribution of initial s-reps features
root = '/path/to/population/mesh/vtks/'
os.chdir(root)
for file_name in os.listdir(root):
    if file_name.split('.')[-1] != 'vtk': continue
    print(file_name)
    file_path = root + file_name
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)

    reader.Update()
    input_mesh = reader.GetOutput()

    ## Initialize an s-rep for the input mesh
    initializer = Initializer()
    srep = initializer.fit(input_mesh)

    ## Visualize the s-rep and the input mesh
    # skel_first_pt = np.array(srep.GetPoint(0))
    # bdry_first_pt = np.array(srep.GetPoint(1))
    # skel_second_pt = np.array(srep.GetPoint(2*8))
    # bdry_second_pt = np.array(srep.GetPoint(2*8+1))

    # plt = pv.Plotter(off_screen=True)
    # plt.add_mesh(input_mesh, color='white', opacity=0.2)
    # plt.add_mesh(srep)
    # plt.add_mesh(pv.Sphere(center=np.array(input_mesh.GetPoint(0))), color='orange')
    # plt.add_mesh(pv.Sphere(center=np.array(input_mesh.GetPoint(1))), color='blue')
    # plt.add_mesh(pv.Arrow(skel_first_pt, bdry_first_pt-skel_first_pt, scale='auto'), color='orange')
    # plt.add_mesh(pv.Arrow(skel_second_pt, bdry_second_pt-skel_second_pt, scale='auto'), color='blue')
    # plt.show(screenshot='/path/to/output_orientation_screenshots.jpg')
    # plt.deep_clean()
    