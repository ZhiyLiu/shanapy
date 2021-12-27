# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:42:25 2014

@author: jvicory
modified by: Zhiyuan Liu, in order to output multiple figures in one model
"""

import numpy as np
import math
import datetime
import xml.etree.ElementTree as ET
import os
import vtk

class srep_io:
    def __init__(self):
        pass
    def organize_srep(self, spoke_polydata, num_rows, num_cols):
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

class SrepWrapper(object):
    def __init__(self, polydata, num_rows=5, num_cols=9):
        self.data = polydata
        self.rows = num_rows
        self.cols = num_cols
class spoke:
    def __init__(self, ux, uy, uz, r):
        self.U = np.array([ux,uy,uz])
        self.r = r
    def updateSpoke(self, bdryPt, skeletalPt):
        spokeVector = bdryPt - skeletalPt
        self.r = np.linalg.norm(spokeVector)
        self.U = spokeVector / self.r
class hub:
    def __init__(self, x, y, z):
        self.P = np.array([x,y,z])

class atom:
    def __init__(self, hub, selected = 0):
        self.hub = hub
        self.selected = selected

    def addSpoke(self, spoke, side):
        if side == 0:
            self.topSpoke = spoke
        elif side == 1:
            self.botSpoke = spoke
        else:
            self.crestSpoke = spoke;

    def setLocation(self, row, col):
        self.row = row;
        self.col = col;

    def isCrest(self):
        return hasattr(self,'crestSpoke')

class figure:
    def __init__(self, numRows, numCols):
        self.numRows = numRows
        self.numCols = numCols
        self.atoms = np.ndarray([numRows,numCols],dtype=object)

    def addAtom(self, row, col, atom):
        self.atoms[row,col] = atom

    def addAtomFromDict(self, row, col, primdict):
        px = float(primdict['x'])
        py = float(primdict['y'])
        pz = float(primdict['z'])

        newhub = hub(px,py,pz)
        newatom = atom(newhub)

        ux0 = float(primdict['ux[0]'])
        uy0 = float(primdict['uy[0]'])
        uz0 = float(primdict['uz[0]'])
        r0 = float(primdict['r[0]'])
        spoke0 = spoke(ux0,uy0,uz0,r0)
        newatom.addSpoke(spoke0,0)

        ux1 = float(primdict['ux[1]'])
        uy1 = float(primdict['uy[1]'])
        uz1 = float(primdict['uz[1]'])
        r1 = float(primdict['r[1]'])
        spoke1 = spoke(ux1,uy1,uz1,r1)
        newatom.addSpoke(spoke1,1)

        primtype = primdict['type']

        if primtype == 'EndPrimitive':
            ux2 = float(primdict['ux[2]'])
            uy2 = float(primdict['uy[2]'])
            uz2 = float(primdict['uz[2]'])
            r2 = float(primdict['r[2]'])
            spoke2 = spoke(ux2,uy2,uz2,r2)
            newatom.addSpoke(spoke2,2)

        self.addAtom(row,col,newatom)

    # get center of the s-rep
    def getCenter(self):
        centerRow = int(self.numRows / 2)
        centerCol = int(self.numCols / 2)
        center = self.atoms[centerRow, centerCol].hub.P
        return center

    # get range X,Y coordinate system of bbx
    def getRange(self):
        boundary_points = []
        for i in range(self.numRows):
            for j in range(self.numCols):
                currAtom = self.atoms[i,j]
                currPoint = currAtom.hub.P
                upSpoke = currAtom.topSpoke
                downSpoke = currAtom.botSpoke
                bdryPoint0 = currPoint + upSpoke.U * upSpoke.r
                bdryPoint1 = currPoint + downSpoke.U * downSpoke.r
                boundary_points.append(bdryPoint0)
                boundary_points.append(bdryPoint1)
                if currAtom.isCrest():
                    crestSpoke = currAtom.crestSpoke
                    bdryPoint2 = currPoint + crestSpoke.U * crestSpoke.r
                    boundary_points.append(bdryPoint2)

        x = [p[0] for p in boundary_points]
        y = [p[1] for p in boundary_points]
        return [min(x), max(x), min(y), max(y)]

    # scale s-rep in the place
    def scale(self, scale_amount):
        center_pt = self.getCenter()
        center_x = center_pt[0]
        center_y = center_pt[1]
        center_z = center_pt[2]
        for i in range(self.numRows):
            for j in range(self.numCols):
                currAtom = self.atoms[i,j]
                currPoint = currAtom.hub.P
                # move to origin
                currPoint[0] -= center_x
                currPoint[1] -= center_y
                currPoint[2] -= center_z

                # scale
                currPoint[0] *= scale_amount
                currPoint[1] *= scale_amount
                currPoint[2] *= scale_amount

                # move back
                currPoint[0] += center_x
                currPoint[1] += center_y
                currPoint[2] += center_z

                currAtom.hub.P = currPoint
                # shrink boundary
                upSpoke = currAtom.topSpoke
                downSpoke = currAtom.botSpoke
                bdryPointUp = currPoint + upSpoke.U * upSpoke.r
                bdryPointDown = currPoint + downSpoke.U * downSpoke.r

                bdryPointUp[0] = (bdryPointUp[0] - center_x) * scale_amount + center_x
                bdryPointUp[1] = (bdryPointUp[1] - center_y) * scale_amount + center_y
                bdryPointUp[2] = (bdryPointUp[2] - center_z) * scale_amount + center_z

                bdryPointDown[0] = (bdryPointDown[0] - center_x) * scale_amount + center_x
                bdryPointDown[1] = (bdryPointDown[1] - center_y) * scale_amount + center_y
                bdryPointDown[2] = (bdryPointDown[2] - center_z) * scale_amount + center_z

                # currAtom.botSpoke.r *= scale_amount
                # currAtom.topSpoke.r *= scale_amount
                currAtom.topSpoke.r = np.sqrt((bdryPointUp[0] - currPoint[0]) ** 2 + (bdryPointUp[1] - currPoint[1]) ** 2 + (bdryPointUp[2] - currPoint[2]) ** 2)
                currAtom.botSpoke.r = np.sqrt((bdryPointDown[0] - currPoint[0]) ** 2 + (bdryPointDown[1] - currPoint[1]) ** 2 + (bdryPointDown[2] - currPoint[2]) ** 2)

                # oldU = currAtom.topSpoke.U
                # oldU[0] = (bdryPointUp[0] - currPoint[0]) / currAtom.topSpoke.r
                # oldU[1] = (bdryPointUp[1] - currPoint[1]) / currAtom.topSpoke.r
                # currAtom.topSpoke.U = oldU
                # oldU = currAtom.botSpoke.U
                # oldU[0] = (bdryPointDown[0] - currPoint[0]) / currAtom.botSpoke.r
                # oldU[1] = (bdryPointDown[1] - currPoint[1]) / currAtom.botSpoke.r
                # currAtom.botSpoke.U = oldU

                if currAtom.isCrest():
                    bdryPointCrest = currAtom.hub.P + currAtom.crestSpoke.U * currAtom.crestSpoke.r
                    bdryPointCrest[0] = (bdryPointCrest[0] - center_x) * scale_amount + center_x
                    bdryPointCrest[1] = (bdryPointCrest[1] - center_y) * scale_amount + center_y
                    bdryPointCrest[2] = (bdryPointCrest[2] - center_z) * scale_amount + center_z

#                    currAtom.crestSpoke.r *= scale_amount
                    currAtom.crestSpoke.r = np.sqrt((bdryPointCrest[0] - currPoint[0]) ** 2 + (bdryPointCrest[1] - currPoint[1]) ** 2 + (bdryPointCrest[2] - currPoint[2]) ** 2)

    def translate(self, trans_dir, trans_amount):
        trans_mat = np.array([[1, 0, 0, trans_dir[0] * trans_amount],
                              [0, 1, 0, trans_dir[1] * trans_amount],
                              [0, 0, 1, trans_dir[2] * trans_amount],
                              [0, 0, 0, 0]])
        for i in range(self.numRows):
            for j in range(self.numCols):
                currAtom = self.atoms[i, j]
                currPoint = currAtom.hub.P
                LHS = np.array([currPoint[0], currPoint[1], 1.0, 1.0])
                after_rotate = np.matmul(trans_mat, LHS)
                # translate skeletal points
                currPoint[0] = after_rotate[0]
                currPoint[1] = after_rotate[1]
                currAtom.hub.P = currPoint

    # rotate around the center of this figure
    def rotate(self, rot_matrix):
        for i in range(self.numRows):
            for j in range(self.numCols):
                currAtom = self.atoms[i, j]
                currPoint = currAtom.hub.P
                LHS = np.array([currPoint[0], currPoint[1], 1.0])
                after_rotate = np.matmul(LHS, rot_matrix)

                # rotate skeletal points
                currPoint[0] = after_rotate[0]
                currPoint[1] = after_rotate[1]
                currAtom.hub.P = currPoint

                # rotate boundary points and update U
                upSpoke = currAtom.topSpoke
                downSpoke = currAtom.botSpoke
                bdryPointUp = currPoint + upSpoke.U * upSpoke.r
                bdryPointDown = currPoint + downSpoke.U * downSpoke.r
                LHS = np.array([bdryPointUp[0], bdryPointUp[1], 1.0])
                after_rotate = np.matmul(upSpoke.U, rot_matrix)
                currAtom.topSpoke.U = after_rotate
#                currAtom.topSpoke.updateSpoke(after_rotate, currPoint)

                LHS = np.array([bdryPointDown[0], bdryPointDown[1], 1.0])
                after_rotate = np.matmul(downSpoke.U, rot_matrix)
                currAtom.botSpoke.U = after_rotate
 #               currAtom.botSpoke.updateSpoke(after_rotate, currPoint)

                if currAtom.isCrest():
                    crestSpoke = currAtom.crestSpoke
                    bdryPointCrest = currPoint + crestSpoke.U + crestSpoke.r
                    LHS = np.array([bdryPointCrest[0], bdryPointCrest[1], 1.0])
                    after_rotate = np.matmul(crestSpoke.U, rot_matrix)
                    currAtom.crestSpoke.U = after_rotate
  #                  currAtom.crestSpoke.updateSpoke(after_rotate, currPoint)

class model:
    def __init__(self):
        self.figures = []
        
    def addFigure(self, fig):
        self.figures.append(fig)
