from int_green3d import *
import trimesh
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def get_mesh(model):
    return trimesh.load_mesh("models/%s.stl" % model)

def gauss_electrostatic_propagator(Pglo, V):
    I, Igrad = int_green3d_tri_multiple(Pglo, V)
    return I / 4 / np.pi, Igrad / 4 / np.pi

def get_charge_basis(model, regenerate=False):

    mesh = get_mesh(model)
    cb_path = "data/%s_charge_basis.npy" % model
    if os.path.exists(cb_path) and not regenerate:
        return np.load(cb_path)   
    nov, nof, nop = len(mesh.vertices), len(mesh.faces), len( trimesh.graph.connected_components(mesh.edges))
    cendroids = mesh.triangles_center

    # indexing 
    label_vertices = trimesh.graph.connected_component_labels(mesh.edges)
    label_triagles = [label_vertices[triagle[0]] for triagle in mesh.faces]

    print("Calculating charge basis")
    alpha = gauss_electrostatic_propagator(cendroids, mesh.triangles).reshape((nof, nof))
    inv_alpha = np.linalg.inv(alpha)
    charge_basis = []
    for part_index in range(nop):
        voltage = np.array([(1 if label_triagles[triagle_index] == part_index else 0) for triagle_index in range(nof)])
        charge_basis.append(inv_alpha @ voltage)
    charge_basis = np.array(charge_basis)
    np.save(cb_path, charge_basis)
    return charge_basis

def get_potential_basis(model, PSCoef=None, field_points=None, regenerate=False):

    pb_path = "data/%s_potential.npz" % model
    if os.path.exists(pb_path) and not regenerate:
        data = np.load(pb_path)
        return data["potential_basis"], data["pseudo_potential"].astype(np.longdouble)

    if PSCoef is None:
        q = 1.6e-19
        m = 0.171/6.02e23
        omega_rf = 2 * np.pi * 30e6
        V_rf = 170
        PSCoef = (q / (4 * m * omega_rf ** 2)) * V_rf ** 2
    
    mesh = trimesh.load_mesh("models/%s.stl" % model)
    nov, nof, nop = len(mesh.vertices), len(mesh.faces), len( trimesh.graph.connected_components(mesh.edges))

    charge_basis = get_charge_basis(mesh)

    if field_points is None:
        field_points = np.mgrid[-1:1:0.005, -0.1:0.1:0.005, 0.05:0.10:0.005]
        field_points = np.transpose(field_points, (1, 2, 3, 0))
    
    grid = field_points.shape[:-1]

    print("Calculating propagators")
    potential_propagators, grad_propagators = gauss_electrostatic_propagator(field_points, mesh.triangles, require_grad=True)
    grad_propagators = np.transpose(grad_propagators, (2, 0, 1))
    field_points_grad = grad_propagators @ charge_basis[0]
    pseudo_potential = ((field_points_grad ** 2).sum(axis=0) * PSCoef).reshape(grid)
    potential_basis = (potential_propagators @ charge_basis.T).reshape(list(grid) + [nop])

    np.savez_compressed(pb_path, 
                        potential_propagators=potential_propagators,
                        potential_basis=potential_basis,
                        pseudo_potential=pseudo_potential,
                        grad_propagators=grad_propagators )

    return potential_basis, pseudo_potential


def get_total_potential(voltage, potential_basis, pseudo_potential):
    dc_potential = np.dot(potential_basis, voltage)
    return dc_potential + pseudo_potential

def get_field_point_ticks(srange, stepsize=0.005):
    xrange, yrange, zrange = srange
    x = np.mgrid[xrange[0]:xrange[1]:stepsize]
    y = np.mgrid[yrange[0]:yrange[1]:stepsize]
    z = np.mgrid[zrange[0]:zrange[1]:stepsize]
    return x, y, z

def get_field_points(srange, stepsize=0.005):
    x, y, z = get_field_point_ticks(srange, stepsize)
    return np.array(np.meshgrid(x, y, z, indexing='ij')).transpose((1, 2, 3, 0))
