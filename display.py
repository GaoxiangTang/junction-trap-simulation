from mpl_toolkits import mplot3d
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import interpnd, RegularGridInterpolator
import numpy as np
from potential_initialization import *
import plotly.graph_objects as go

def mesh3d(mesh,**kwargs):
    vertices = mesh.vertices
    triangles = mesh.faces
    x, y, z = vertices.T
    I, J, K = triangles.T              
    Xe = []
    Ye = []
    Ze = []
    for T in vertices[triangles] :
        Xe.extend([T[k%3][0] for k in range(4)]+[None])
        Ye.extend([T[k%3][1] for k in range(4)]+[None])
        Ze.extend([T[k%3][2] for k in range(4)]+[None])
    return (go.Mesh3d(x=x,y=y,z=z,i=I,j=J,k=K,flatshading=True,showscale=False,**kwargs),)
            #go.Scatter3d(x=Xe,y=Ye,z=Ze,mode='lines',line=dict(color='rgb(70,70,70)',width=1),name='')) 

def potential_slicez(srange, stepsize, z, potential=None, interp=None, ax=None):
    xtick, ytick, ztick = get_field_point_ticks(srange, stepsize)
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

def potential_slicex(srange, stepsize, x, potential=None, interp=None, ax=None):
    xtick, ytick, ztick = get_field_point_ticks(srange, stepsize)
    if interp is None:
        interp = RegularGridInterpolator((xtick, ytick, ztick), potential)
    points = np.array(np.meshgrid(np.array([x]), ytick, ztick, indexing='ij')).transpose((1, 2, 3, 0))
    slice = interp(points.reshape((int(points.size / 3), 3)))
    grid = np.meshgrid(ytick, ztick, indexing='ij')
    if ax is None:
        cs = plt.contourf(grid[0], grid[1], slice.reshape(grid[0].shape), levels=100)
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


def total_potential_slice(voltage, potential_basis, pseudo_potential, srange, stepsize, z, PSCoef=None):
    tp = get_total_potential(voltage, potential_basis, pseudo_potential, PSCoef)
    return potential_slicez(srange, stepsize, z, tp)

def part(mesh, part_idx=0, ):
    part_id = np.array(part_idx).ravel()
    # part = trimesh.graph.split(mesh, False)[part_idx]
    # Create a new plot
    electrodes = trimesh.graph.split(mesh,False)

    scene = dict(aspectratio=dict(x=3,y=2,z=1),zaxis=dict(range=[0,0.1]),camera=dict(eye=dict(x=0,y=-1.5,z=3)))
    layout = go.Layout(scene=scene,margin=dict(r=0,l=0,b=0,t=0), autosize=False)#,height=400)
    data = []
    for i in range(len(electrodes)):
        data += mesh3d(electrodes[i],color='goldenrod' if i in part_id else 'grey',hoverinfo='name',name=str(i),opacity=1 if i in part_id else 0.5)
    return go.Figure(data=data,layout=layout)


def part0(mesh, part_idx=0):
    part_id = np.array(part_idx).ravel()
    part = trimesh.graph.split(mesh, False)[part_idx]
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

