SHANAPY: SHape ANAlysis PYthon package using s-reps
===
![downloads](https://img.shields.io/github/downloads/atom/atom/total.svg)
![build](https://img.shields.io/appveyor/ci/:user/:repo.svg)
![chat](https://img.shields.io/discord/:serverId.svg)

## Table of Contents

1. [Introduction](#intro)
2. [Installation](#use)
3. [Examples](#example)

<a name="intro"></a>

## Introduction

### What are s-reps?

An s-rep is a skeletal representation of an object. In the continuous regime, an s-rep consists of a skeleton and a radial vector field defined (based) on the skeleton. 
Though a continuous s-rep is desirable in some situations (e.g., refinement), a discrete s-rep is favorable in statistical analysis because of good correspondences across a population.
A discrete s-rep (also referred to an s-rep in some places) consists of skeletal points, and spokes that connect the skeletal points and boundary points.
The skeletal points form quadrilaterals on the skeleton. A continuous skeleton is thus approximated by the combination of these quadrilaterals.

This package provides an algorithm to obtain a discrete s-rep that best fit to a 3D object.
Often, the object of interest has a close and smooth boundary with no holes. 
The discretized skeletal sheet sits in the near-medial place of the object. These skeletal points are sampled in a consistent way to provide good correspondences.
At these skeletal points, spokes are correspondingly sampled from the radial vector field. These spokes define the mapping from the skeletal geometry to the boundary geometry.
A reader can refer to chapter 3 in the book
> K. Siddiqi and S. Pizer, Medial representations: mathematics, algorithms and applications, 2008
for more details about the relation between the skeletal and boundary geometry.

A formal definition of an s-rep and the discretization of an s-rep can be found in the paper
> Z. Liu et al., Fitting unbranching skeletal structures to objects, Medical Image Analysis, 2021

### Anatomical shape analysis with s-reps

Anatomical shapes are typically obtained from segmentation of medical images. It is often problematic to use boundary geometry in analyzing a population of anatomical shapes for the following reasons.
First, the boundary is often noisy, sometimes correupted (see the figure (a) below). These noise and corruptions unexpectedly draw too much attention in statistical analysis. 
Therefore, the results from analyzing boundaries can be biased and difficult to generalize to new shapes.
Second, it is difficult to establish good correspondences on 3D objects. There exist methods (e.g., SPHARM-PDM) producing mathematical landmarks of an object. 
However, the correspondences of these landmarks across a population can be not anatomically reasonable (see the figure (b) below).
![Problems with boundary geometry](figures/problems_in_boundary_geometry.png)

In contrast, s-reps take the interior geometry of an object into consideration, making the geometric features more robust. 
The skeleton shape and the differential properties of spokes allow us to reconstruct the boundary geometry from the skeletal geometry.
This reconstructed boundary is also referred to as the `implied boundary` or `onion skin` (see the figure below).
![3D onion skins](figures/onion_skins_3d.png)

S-reps provide rich anatomical shape features. A user can select appropriate features according to the data and tasks.
The features provided by s-reps include:

(1) Geometric features of implied boundaries. S-reps can produce corruption-free smooth boundary mesh. Moreover, the implied boundary points have good correspondences across a population.
See e.g.
> Z. Liu et al., Non-Euclidean Analysis of Joint Variations in Multi-Object Shapes, 2021

(2) Radial geometry from spokes. A spoke is represented by a tuple containing (a) coordinates of the base (skeletal) point (b) a unit direction vector in $\mathbb{R}^3$ and 
(c) a positive scalar value that indicates the spoke's length.

The workflow of using s-reps can be summarized in the following chart.
![Flowchart](figures/srep_fitting_workflow.png)

---
<a name="use"></a>
## Installation
1. (Optional) Download [SPHARM-PDM](https://www.nitrc.org/projects/spharm-pdm) according to your platform into `third_party/spharm_bin`.
2. Install shanapy module as follows
```bash=
## clone the source code
$ cd ~
$ git clone https://github.com/ZhiyLiu/shanapy.git
$ cd shanapy

## create & activate a virtual envirionment
## assume the anaconda has been installed
$ conda create -n shanapy python=3.7 anaconda
$ source activate shanapy

## install required packages
$ python -m pip install -r requirements.txt

## install pyshanalysis
$ python -m pip install -e .
```
---
<a name="example"></a>
## Example

### 1. Initialize and visualize a discrete s-rep for a hippocampus
The example data (a hippocampus) is provided [here](data/example_hippocampus.vtk). The example code is in [this file](shanapy/test/test_initializer.py). To run the example,
```bash=
## Run the above installation before the following executions.
## make sure the current directory is ~/shanapy/
$ pwd
$ python shanapy/test/test_initializer.py
```
The above commands result in a fitted s-rep the example hippocampus. Also, a visualization window should show as follows.
![InitResult](figures/test_initializer_result.png)
The transparent surface is the boundary of the hippocampus. The white line segments are discrete spokes connecting the skeleton and the boundary.
### 2. Refine the above s-rep
The example code for the refinement is in [this file](shanapy/test/test_refiner.py). To run the example,
```bash=
## Run the above installation before the following executions.
## make sure the current directory is ~/shanapy/
$ pwd
$ python shanapy/test/test_refiner.py
```
The refinement yields a better fit s-rep, as shown below.
![RefineResult](figures/test_refiner_result.png)
The white line segments are from the above initial s-rep. The red line segments are from the refined s-rep. 
The black arrow points to a spoke that was initially longer than expected but was refined due to the refinement algorithm.
## Acknowledgement
This project is adviced by Stephen M. Pizer, J. S. Marron and James N. Damon.
J. Hong initiates this project. J. Vicory and B. Paniagua significantly contributed to this project.
Thanks to M. Styner for providing insightful comments and experimental data.
Special thanks to my great colleagues M. Taheri, N. Tapp-Hughes, A. Sharma and J. Schulz for their feedback and contributions. 

###### tags: `Simulation` `Shape models` `Shape analysis`
