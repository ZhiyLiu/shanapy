# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:57:11 2014

@author: jvicory
modified by: Zhiyuan Liu, in order to output multiple figures in one model
"""

from shanapy.models.srep import *

import numpy as np
def organize_srep(spoke_polydata, num_rows, num_cols):
    spokePointData = spoke_polydata.GetPointData()
    numberOfArrays = spokePointData.GetNumberOfArrays()
    if numberOfArrays is 0:
        print('empty array')

    spokePoints = vtk.vtkPoints()
    spokeLines = vtk.vtkCellArray()

    arr_length = spokePointData.GetArray('spokeLength')
    arr_dirs = spokePointData.GetArray('spokeDirection')
    base_pts_array = vtk.vtkDoubleArray()
    base_pts_array.SetNumberOfComponents(3)
    base_pts_array.SetName("basePoints")
    for i in range(spoke_polydata.GetNumberOfPoints()):
        pt = [0] * 3
        spoke_polydata.GetPoint(i, pt)
        # base point of up arrows
        id0 = spokePoints.InsertNextPoint(pt)
        base_pts_array.InsertNextTuple(pt)

        # head of up arrows
        spoke_length = arr_length.GetValue(i)
        baseIdx = i * 3
        dirX = arr_dirs.GetValue(baseIdx)
        dirY = arr_dirs.GetValue(baseIdx + 1)
        dirZ = arr_dirs.GetValue(baseIdx + 2)
        pt1 = [0] * 3
        pt1[0] = pt[0] + spoke_length * dirX
        pt1[1] = pt[1] + spoke_length * dirY
        pt1[2] = pt[2] + spoke_length * dirZ
        id1 = spokePoints.InsertNextPoint(pt1)

        up_arrow = vtk.vtkLine()
        up_arrow.GetPointIds().SetId(0, id0)
        up_arrow.GetPointIds().SetId(1, id1)
        spokeLines.InsertNextCell(up_arrow)

    renderable_srep = vtk.vtkPolyData()
    renderable_srep.SetPoints(spokePoints)
    renderable_srep.SetLines(spokeLines)
    renderable_srep.GetPointData().AddArray(arr_length)

    renderable_srep.GetPointData().AddArray(arr_dirs)
    renderable_srep.GetPointData().AddArray(base_pts_array)
    # organized_srep = {"skeletal_points_vtk": spokePoints,
    #                   "radii_da": arr_length,
    #                   "dirs_da": arr_dirs,
    #                   "num_rows": num_rows,
    #                   "num_cols": num_cols}
    return SrepWrapper(renderable_srep, num_rows, num_cols)

def convert_vtk_polydata(input_poly, start_pt_id, end_pt_id):
    new_srep_vtk = vtk.vtkPolyData()
    new_srep_pts = vtk.vtkPoints()
    radii_array = vtk.vtkDoubleArray()
    radii_array.SetNumberOfComponents(1)
    radii_array.SetName('spokeLength')

    u_array = vtk.vtkDoubleArray()
    u_array.SetNumberOfComponents(3)
    u_array.SetName('spokeDirection')
    for i in range(start_pt_id, end_pt_id, 2):
        bdry_pt_id = i + 1
        base_pt = np.array(input_poly.GetPoint(i))
        bdry_pt = np.array(input_poly.GetPoint(i+1))

        r = np.linalg.norm(bdry_pt - base_pt)
        u = (bdry_pt - base_pt) / r
        radii_array.InsertNextValue(r)
        u_array.InsertNextTuple(u)

        new_srep_pts.InsertNextPoint(base_pt)

    new_srep_vtk.GetPointData().AddArray(radii_array)
    new_srep_vtk.GetPointData().AddArray(u_array)
    new_srep_vtk.SetPoints(new_srep_pts)
    new_srep_vtk.Modified()
    return organize_srep(new_srep_vtk, 5, 9)

def read_from_simulation_vtk(file_path):
    """
    Read sreps that saved in vtk files for simulation of ellipsoids
    Need to separate up, down and crest spokes
    Layout information: num_rows = 5, num_cols = 9    <===>  num_crest_points = 24, num_steps = 3
    """
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    srep_vtk = reader.GetOutput()
    total_num_pts = srep_vtk.GetNumberOfPoints()
    num_crest_points = 24  ## num of crest spokes
    end_of_down_spokes = total_num_pts - 2 * num_crest_points
    end_of_up_spokes = end_of_down_spokes // 2

    up_renderable = convert_vtk_polydata(srep_vtk, 0, end_of_up_spokes)
    down_renderable = convert_vtk_polydata(srep_vtk, end_of_up_spokes, end_of_down_spokes)
    crest_renderable = convert_vtk_polydata(srep_vtk, end_of_down_spokes, total_num_pts)
    return up_renderable, down_renderable, crest_renderable
def readSrepFromXML(filename):
    """ Parse header.xml file, create models from the data, and visualize it. """
    # 1. parse header file
    tree = ET.parse(filename)
    upFileName = ''
    crestFileName = ''
    downFileName = ''
    nCols = 0
    nRows = 0
    headerFolder = os.path.dirname(filename)
    for child in tree.getroot():
        if child.tag == 'upSpoke':
            # if os.path.isabs(child.text):
            #     upFileName = os.path.join(headerFolder, child.text.split('/')[-1])
            upFileName = os.path.join(headerFolder, child.text.split('/')[-1])
        elif child.tag == 'downSpoke':
            downFileName = os.path.join(headerFolder, child.text.split('/')[-1])
        elif child.tag == 'crestSpoke':
            crestFileName = os.path.join(headerFolder, child.text.split('/')[-1])
        elif child.tag == 'nRows':
            nRows = (int)(child.text)
        elif child.tag == 'nCols':
            nCols = (int)(child.text)

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(upFileName)
    reader.Update()

    # GetPoint == base point
    # spokeLength and spokeDirection are in PointData
    upSpokes = reader.GetOutput()

    down_reader = vtk.vtkXMLPolyDataReader()
    down_reader.SetFileName(downFileName)
    down_reader.Update()
    downSpokes = down_reader.GetOutput()

    crest_reader = vtk.vtkXMLPolyDataReader()
    crest_reader.SetFileName(crestFileName)
    crest_reader.Update()
    crestSpokes = crest_reader.GetOutput()

    up_renderable = organize_srep(upSpokes, nRows, nCols)
    down_renderable = organize_srep(downSpokes, nRows, nCols)
    crest_renderable = organize_srep(crestSpokes, nRows, nCols)
    return up_renderable, down_renderable, crest_renderable, nRows, nCols

def export_non_linked_region(srep_poly, filename, pad_zeros=True):
    """
    Save srep_poly (vtkPolyData) to filename
    """
    with open(filename, 'a') as f:
        radii_da = srep_poly.GetPointData().GetArray('spokeLength')
        dirs_da = srep_poly.GetPointData().GetArray('spokeDirection')
        for i in range(radii_da.GetNumberOfValues()):
            r = radii_da.GetValue(i)
            pt_id = i * 2
            base_pt = srep_poly.GetPoint(pt_id)
            u_pt = dirs_da.GetTuple3(i)
            if pad_zeros:
                f.write('%f,%f,%f,%f,%f,%f,%f;%f,%f,%f,%f,%f,%f,%f\n' % (base_pt[0], base_pt[1], base_pt[2], u_pt[0], u_pt[1], u_pt[2], r, 0, 0, 0, 0, 0, 0, 0))
            else:
                f.write('%f,%f,%f,%f,%f,%f,%f\n' % (base_pt[0], base_pt[1], base_pt[2], u_pt[0], u_pt[1], u_pt[2], r))
def readSrepFromM3D(filename):
    f = open(filename,'r')
    
    lines = [line.strip() for line in f.readlines()]
    
    figidx = lines.index('figure[0] {')
    coloridx = lines.index('color {')
    endidx = lines.index('}',coloridx)

    figparams = getSectionDict(lines,figidx+1,coloridx)
    numrows = int(figparams['numRows'])
    numcols = int(figparams['numColumns'])

    fig = srep.figure(numrows,numcols)

    primidx = endidx + 1;
    
    for row in range(numrows):
        for col in range(numcols):
            endprimidx = lines.index('}',primidx)
            primsection = getSectionDict(lines,primidx+1,endprimidx)
            fig.addAtomFromDict(row,col,primsection)
            primidx = endprimidx + 1;
    
    return fig

# read select (the number of figure) from multi-figure srep
def readFigFromSrep(filename, figNum):
    f = open(filename,'r')
    
    lines = [line.strip() for line in f.readlines()]
    
    figidx = lines.index('figure['+ str(figNum) + '] {')
    coloridx = lines.index('color {', figidx)
    endidx = lines.index('}',coloridx)

    figparams = getSectionDict(lines,figidx+1,coloridx)
    numrows = int(figparams['numRows'])
    numcols = int(figparams['numColumns'])

    fig = srep.figure(numrows,numcols)

    primidx = endidx + 1;
    
    for row in range(numrows):
        for col in range(numcols):
            endprimidx = lines.index('}',primidx)
            primsection = getSectionDict(lines,primidx+1,endprimidx)
            fig.addAtomFromDict(row,col,primsection)
            primidx = endprimidx + 1;
    
    return fig


def writeSingleFigure(f, fig, figId):
    f.write('\tfigure[%s] {\n'.expandtabs(4) % figId);
    f.write('\t\tname = test;\n'.expandtabs(4));
    
    f.write('\t\tnumColumns = %s;\n'.expandtabs(4) % fig.numCols);
    
    f.write('\t\tnumLandmarks = 0;\n'.expandtabs(4));
    
    f.write('\t\tnumRows = %s;\n'.expandtabs(4) % fig.numRows);
    
    f.write('\t\tpositivePolarity = 1;\n'.expandtabs(4));
    f.write('\t\tpositiveSpace = 1;\n'.expandtabs(4));
    f.write('\t\tsmoothness = 50;\n'.expandtabs(4));
    f.write('\t\ttype = QuadFigure;\n'.expandtabs(4));
    
    f.write('\t\tcolor {\n'.expandtabs(4));
    f.write('\t\t\tblue = 0;\n'.expandtabs(4));
    f.write('\t\t\tgreen = 1;\n'.expandtabs(4));
    f.write('\t\t\tred = 0;\n'.expandtabs(4));
    f.write('\t\t}\n'.expandtabs(4));
    
    for row in range(fig.numRows):
        for col in range(fig.numCols):
            
            isCrest = fig.atoms[row,col].isCrest();
            f.write('\t\tprimitive[{0}][{1}] {{\n'.format(row,col).expandtabs(4));

            f.write('\t\t\tr[0] = {0};\n'.format(fig.atoms[row,col].topSpoke.r).expandtabs(4));
            f.write('\t\t\tr[1] = {0};\n'.format(fig.atoms[row,col].botSpoke.r).expandtabs(4));
            if isCrest:
                f.write('\t\t\tr[2] = {0};\n'.format(fig.atoms[row,col].crestSpoke.r).expandtabs(4));
            
            f.write('\t\t\tselected = 1;\n'.format(fig.atoms[row,col].selected).expandtabs(4));
            
            if isCrest:
                f.write('\t\t\ttype = EndPrimitive;\n'.expandtabs(4));
            else:
                f.write('\t\t\ttype = StandardPrimitive;\n'.expandtabs(4));
            
            f.write('\t\t\tux[0] = {0};\n'.format(fig.atoms[row,col].topSpoke.U[0]).expandtabs(4));
            f.write('\t\t\tux[1] = {0};\n'.format(fig.atoms[row,col].botSpoke.U[0]).expandtabs(4));
            
            if isCrest:
                f.write('\t\t\tux[2] = {0};\n'.format(fig.atoms[row,col].crestSpoke.U[0]).expandtabs(4));
            else:
                f.write('\t\t\tux[2] = 1;\n'.expandtabs(4));
                
            f.write('\t\t\tuy[0] = {0};\n'.format(fig.atoms[row,col].topSpoke.U[1]).expandtabs(4));
            f.write('\t\t\tuy[1] = {0};\n'.format(fig.atoms[row,col].botSpoke.U[1]).expandtabs(4));
            
            if isCrest:
                f.write('\t\t\tuy[2] = {0};\n'.format(fig.atoms[row,col].crestSpoke.U[1]).expandtabs(4));
            else:
                f.write('\t\t\tuy[2] = 0;\n'.expandtabs(4));
                
            f.write('\t\t\tuz[0] = {0};\n'.format(fig.atoms[row,col].topSpoke.U[2]).expandtabs(4));
            f.write('\t\t\tuz[1] = {0};\n'.format(fig.atoms[row,col].botSpoke.U[2]).expandtabs(4));
            
            if isCrest:
                f.write('\t\t\tuz[2] = {0};\n'.format(fig.atoms[row,col].crestSpoke.U[2]).expandtabs(4));
            else:
                f.write('\t\t\tuz[2] = 0;\n'.expandtabs(4));
                
            f.write('\t\t\tx = {0};\n'.format(fig.atoms[row,col].hub.P[0]).expandtabs(4));
            f.write('\t\t\ty = {0};\n'.format(fig.atoms[row,col].hub.P[1]).expandtabs(4));
            f.write('\t\t\tz = {0};\n'.format(fig.atoms[row,col].hub.P[2]).expandtabs(4));
            
            f.write('\t\t}\n'.expandtabs(4));
            
    f.write('\t}\n'.expandtabs(4));


def writeSrepToM3D(filename, model):    
    f = open(filename,'w')
    
    f.write('pabloVersion = 9974 2009/07/24 19:36:23;\n'.expandtabs(4));
    f.write('coordSystem {\n'.expandtabs(4));
    f.write('\tyDirection = 1;\n'.expandtabs(4));
    f.write('}\n'.expandtabs(4));

    numFigs = 1
    if isinstance(model, srep.model):
        numFigs = len(model.figures)
    f.write('model {\n'.expandtabs(4));
#    f.write('\tfigureCount = 1;\n'.expandtabs(4));
    f.write('\tfigureCount = %s;\n'.expandtabs(4) % numFigs);
    f.write('\tname = test;\n'.expandtabs(4));
    # f.write('\tfigureTrees {\n'.expandtabs(4));
    # f.write('\t\tcount=1;\n'.expandtabs(4));
    # f.write('\t\ttree[0] {\n'.expandtabs(4));
    # f.write('\t\t\tattachmentMode = 0;\n'.expandtabs(4));
    # f.write('\t\t\tblendAmount = 0;\n'.expandtabs(4));    
    # f.write('\t\t\tblendExtent = 0;\n'.expandtabs(4));
    # f.write('\t\t\tchildCount = 0;\n'.expandtabs(4));
    # f.write('\t\t\tfigureId = 0;\n'.expandtabs(4));
    # f.write('\t\t\tlinkCount = 0;\n'.expandtabs(4));
    # f.write('\t\t}\n'.expandtabs(4));
    # f.write('\t}\n'.expandtabs(4));
    if isinstance(model, srep.model):
        i = 0
        for fig in model.figures:
            writeSingleFigure(f, fig, i)
            i = i+1
    elif isinstance(model, srep.figure):
        writeSingleFigure(f, model, 0)
    f.write('}');
    f.close()

def getSectionDict(lines, start, stop):
    section = [line.strip(';') for line in lines[start:stop]]
    sectiondict = dict(line.split(' = ') for line in section)
    return sectiondict
    
    