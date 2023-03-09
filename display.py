from mpl_toolkits import mplot3d
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import interpnd, RegularGridInterpolator
import numpy as np
from potential_initialization import *


def potential_slicez(potential, srange, z):
    xtick, ytick, ztick = get_field_point_ticks(srange)
    interp = RegularGridInterpolator((xtick, ytick, ztick), potential, meth)
    slice = interp(np.array(np.meshgrid(xtick, ytick, np.array([z]), indexing='ij')).transpose((1, 2, 3, 0)))
    grid = np.meshgrid(xtick, ytick, indexing='ij')
    plt.contourf(grid[0], grid[1], slice.reshape(grid[0].shape), levels=30)
    plt.title("z=%.2f" % z)
    plt.show()
    return slice

def potential_slicex(potential, srange, x):
    xtick, ytick, ztick = get_field_point_ticks(srange)
    interp = RegularGridInterpolator((xtick, ytick, ztick), potential)
    slice = interp(np.array(np.meshgrid(np.array([x]), ytick, ztick, indexing='ij')).transpose((1, 2, 3, 0)))
    grid = np.meshgrid(ytick, ztick, indexing='ij')
    plt.contourf(grid[0], grid[1], slice.reshape(grid[0].shape), levels=30)
    plt.title("x=%.2f" % x)
    plt.show()
    return slice


def total_potential_slice(voltage, potential_basis, pseudo_potential, srange, z):
    tp = get_total_potential(voltage, potential_basis, pseudo_potential)
    return potential_slicez(tp, srange, z)

def part(mesh, part_idx):
    part = trimesh.graph.split(mesh, False)[part_idx]
    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.triangles, alpha=0.3))
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(part.triangles, color="orange"))
    # Auto scale to the mesh size
    scale = mesh.vertices.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()