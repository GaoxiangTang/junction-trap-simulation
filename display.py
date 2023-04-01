from mpl_toolkits import mplot3d
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import interpnd, RegularGridInterpolator
import numpy as np
from potential_initialization import *


def potential_slicez(srange, z, potential=None, interp=None, ax=None):
    xtick, ytick, ztick = get_field_point_ticks(srange)
    if interp is None:
        interp = RegularGridInterpolator((xtick, ytick, ztick), potential)
    points = np.array(np.array(np.meshgrid(xtick, ytick, np.array([z]), indexing='ij')).transpose((1, 2, 3, 0)))
    slice = interp(points.reshape((int(points.size / 3), 3)))
    grid = np.meshgrid(xtick, ytick, indexing='ij')
    if ax is None:
        cs = plt.contourf(grid[0], grid[1], slice.reshape(grid[0].shape), levels=30)
        plt.colorbar(cs)
        plt.title("z=%.2f" % z)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    else:
        cs = ax.contourf(grid[0], grid[1], slice.reshape(grid[0].shape), levels=30)
        ax.set_title("z=%.2f" % z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

def potential_slicex(srange, x, potential=None, interp=None, ax=None):
    xtick, ytick, ztick = get_field_point_ticks(srange)
    if interp is None:
        interp = RegularGridInterpolator((xtick, ytick, ztick), potential)
    points = np.array(np.meshgrid(np.array([x]), ytick, ztick, indexing='ij')).transpose((1, 2, 3, 0))
    slice = interp(points.reshape((int(points.size / 3), 3)))
    grid = np.meshgrid(ytick, ztick, indexing='ij')
    if ax is None:
        cs = plt.contourf(grid[0], grid[1], slice.reshape(grid[0].shape), levels=30)
        plt.colorbar(cs)
        plt.title("x=%.2f" % x)
        plt.xlabel('y')
        plt.ylabel('z')
        plt.show()
    else:
        cs = ax.contourf(grid[0], grid[1], slice.reshape(grid[0].shape), levels=30)
        ax.set_title("x=%.2f" % x)
        ax.set_xlabel('y')
        ax.set_ylabel('z')


def total_potential_slice(voltage, potential_basis, pseudo_potential, srange, z):
    tp = get_total_potential(voltage, potential_basis, pseudo_potential)
    return potential_slicez(tp, srange, z)

def part(mesh, part_idx=0, ):
    part_idx = np.array(part_idx).ravel()
    part = trimesh.graph.split(mesh, False)[part_idx]
    # Create a new plot
    figure = plt.figure()
    axes = figure.add_subplot(projection='3d')

    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(mesh.triangles, alpha=0.3))
    if type(part) == np.ndarray or type(part) == list:
        n = len(part)
        for idx, (pid, p) in enumerate(zip(part_idx, part)):
            alpha = (n - idx) / n
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(p.triangles, color="orange", alpha=alpha))
    else:
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(part.triangles, color="orange"))
    # Auto scale to the mesh size
    scale = mesh.vertices.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    return axes
 

