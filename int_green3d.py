# -*- coding: utf-8 -*-
# INT_GREEN3D_TRI integrates '1/r' and 'grad(1/r)' over a triangle
#
# USE:
# I1, Igrad = int_green3d_TRI(Pglo, V)
#
# INPUTS:
# 'Pglo': point(s) to calculate integral (in in global 3d coordinates)
# 'V': vertices of triangle (in global 3d coordinates)
#
# OUTPUTS:
# 'I1': value of the integral of '1/r'
# 'Igrad': value of the integral of 'grad(1/r)' in global 3d coordinates
#
# WARNING:
# Igrad will be diverge if the respective point lies on edges. We use a
# threshold to keep points away from edges. So if a input point is too
# close to the triangle edges, a huge error will be introduced.
#
# NOTE:
# See R. Graglia, "Numerical Integration of the linear shape functions
# times the 3-d Green's function or its gradient on a plane triangle", IEEE
# Transactions on Antennas and Propagation, vol. 41, no. 10, Oct 1993,
# pp. 1448--1455
#
# Adapted from MATLAB Version:
# Date: 03.03.2010
# Copyright(C) 2010-2014: Fabio Freschi (fabio.freschi@polito.it)
#
# This is a Python Version:
# Date: 09.07.2017
# by 唐高翔 （995316072@qq.com)

import numpy as np
from numpy import log, exp, abs, arctan
try:
    NB = get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
except:
    NB = False
if NB:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def versor(v):
    v = np.array(v)
    return v / np.sqrt((v ** 2).sum(axis=-1))[..., np.newaxis]


def int_green3d_tri(Pglo, V):

    # number of field points
    nfp = int(np.size(Pglo)/3)
    Pglo = Pglo.reshape((nfp, 3))

    
    # turn the input to array
    Pglo = np.array(Pglo)
    Pglo = Pglo.reshape(nfp, 3).transpose() # use different way to store P
    V = np.array(V)
    
    # local-to-global rotation matrix
    vec1 = V[1]-V[0]
    vec2 = V[2]-V[0]
    vec3 = V[2]-V[1]
    dx = versor(vec1)                      # local x-axis
    dz = versor(np.cross(vec1, vec2))      # local z-axis
    dy = versor(np.cross(dz, dx))          # local y-axis
    R = (np.mat([dx,dy,dz]).T).I  # the inverse of matrix [dx,dy,dz]
    
    # field points in local coordinates
    Ploc = Pglo-V[0].reshape(3,1)        # translation of field points Pglo
    Ploc = np.array(R*np.mat(Ploc))    # rotate the coordinate
    u0 = Ploc[0]                        # notation according to Graglia's paper
    v0 = Ploc[1]
    w0 = Ploc[2]
    
    # vertices in local coordinates
    Vloc = V-V[0]  # translation
    Vloc = np.array((R*np.mat(Vloc).T).T)      # rotate the coordinate
    l3 = Vloc[1][0]                     # notation according to Graglia's paper
    u3 = Vloc[2][0]
    v3 = Vloc[2][1]
    
    # edge lengths
    l1 = np.linalg.norm(vec3)
    l2 = np.linalg.norm(vec2)
    
    # threshold for small numbers
    threshold = 1e-6 * np.min([l1,l2,l3])
    
    # make sure Igrad behaves well near the plane
    w0[abs(w0) < threshold] = 0   
    
    # versors normal to edges Fig. 1(b)
    m = np.array([    # m = s x w, with w = [0 0 1]
        versor([v3,l3-u3,0]),  # m1
        versor([-v3,u3,0]),    # m2
        versor([0,-l3,0])      # m3
    ])     

    # useful quantities for integration
    sminus = np.array([-((l3-u3)*(l3-u0)+v3*v0)/l1, 
                        -(u3*(u3-u0)+v3*(v3-v0))/l2, -u0]) # eq. (3)
    
    splus = np.array([((u3-l3)*(u3-u0)+v3*(v3-v0))/l1, 
                                 (u3*u0+v3*v0)/l2, l3-u0]) # eq. (3)
    
    t0 = np.array([((u3-l3)*v0+v3*(l3-u0))/l1, (v3*u0-u3*v0)/l2, v0]) # eq. (4)
    
    tplus = np.array([np.sqrt((u3-u0)**2+(v3-v0)**2), 
                    np.sqrt(u0**2+v0**2), np.sqrt((l3-u0)**2+v0**2)]) # eq. (5)

    tminus = np.array([tplus[2],tplus[0], tplus[1]])  # eq. (5)
    
    R0 = np.sqrt(t0**2+w0**2)                # line 1 pp. 1450
    
    Rminus = np.sqrt(tminus**2+w0**2)        # line 2 pp. 1450
    
    Rplus = np.sqrt(tplus**2+w0**2)          # line 2 pp. 1450
    
    # return Vloc, tminus
    def get_f2_beta(w0, Rplus, Rminus, splus, sminus, tplus, tminus, t0, R0):

        # field point not in the plane of triangle
        if abs(w0) >= threshold:
            f2 = log(Rplus + splus) - log(Rminus + sminus) # eq. (11)
            beta = arctan(t0 * splus / (R0 ** 2 + abs(w0) * Rplus)) - arctan(t0 * sminus / (R0 ** 2 + abs(w0) * Rminus)) # eq. (14)

        # field point in the plane of triangle but not aligned with edges
        if abs(w0) < threshold and abs(t0) >= threshold:
            f2 = log((tplus + splus) / (tminus + sminus)) # eq. (15)
            beta = arctan(splus / t0) - arctan(sminus / t0) # eq. (17)

        # field point in the plane of triangle and aligned with edges
        if abs(w0) < threshold and abs(t0) < threshold:
            beta = 0
            f2 = abs(log(splus / sminus)) # abs(lim t->0 eq. 15)
            
        # fix value for point on triangle corners (undocumented)
        if not np.isfinite(f2):
            f2 = 0
        return f2, beta

    # use vectorized functions to speed up calculation.
    f2, beta = np.vectorize(get_f2_beta)(w0, Rplus, Rminus, splus, sminus, tplus, tminus, t0, R0)
                
    # integral value of '1/r'
    I1 = np.sum(t0*f2-abs(w0)*beta, axis=0).reshape(nfp,1) # eq. (19)

    # integral value of grad(1/r)
    Igradloc = np.array([                                  # eq. (34)
    -m[0][0]*f2[0]-m[1][0]*f2[1]-m[2][0]*f2[2],
    -m[0][1]*f2[0]-m[1][1]*f2[1]-m[2][1]*f2[2],
    -np.sign(w0)*np.sum(beta, axis=0)])     
    Igrad = np.array((np.mat([dx,dy,dz]).T*np.mat(Igradloc)).transpose())
    
    return I1, Igrad

def int_green3d_vectorized(Pglo, V):

    # number of field points
    nfp = int(np.size(Pglo)/3)

    # number of field points
    nft = int(np.size(V)/9)

    
    # turn the input to array
    Pglo = np.array(Pglo)
    Pglo = Pglo.reshape(nfp, 3).transpose() # use different way to store P
    V = V.reshape((nft, 3, 3))
    
    # local-to-global rotation matrix
    vec1 = V[:, 1]-V[:, 0]
    vec2 = V[:, 2]-V[:, 0]
    vec3 = V[:, 2]-V[:, 1]
    dx = versor(vec1)                      # local x-axis
    dz = versor(np.cross(vec1, vec2))      # local z-axis
    dy = versor(np.cross(dz, dx))          # local y-axis
    # R = 

    R = np.linalg.inv(np.array([dx,dy,dz]).transpose((1, 2, 0)))  # the inverse of matrix [dx,dy,dz]
    
    # field points in local coordinates
    Ploc = Pglo[None, :, :] - V[:, 0][:, :, None]        # translation of field points Pglo
    Ploc = R @ Ploc  # rotate the coordinate
    u0 = Ploc[:, 0]                        # notation according to Graglia's paper
    v0 = Ploc[:, 1]
    w0 = Ploc[:, 2]
    
    # vertices in local coordinates
    Vloc = V - V[:, 0][:, None, :]  # translation
    print(Vloc.shape)
    Vloc = (R @ Vloc.transpose((0, 2, 1))).transpose((0, 2, 1))      # rotate the coordinate
    l3 = Vloc[:, 1, 0]                     # notation according to Graglia's paper
    u3 = Vloc[:, 2, 0]
    v3 = Vloc[:, 2, 1]
    # edge lengths
    l1 = (vec3 ** 2).sum(axis=-1) ** 0.5
    l2 = (vec2 ** 2).sum(axis=-1) ** 0.5
    
    # threshold for small numbers
    threshold = 1e-6 * np.min(np.array([l1,l2,l3]), axis=0)
    # make sure Igrad behaves well near the plane
    w0[abs(w0) < threshold[:, None]] = 0   
    
    # versors normal to edges Fig. 1(b)
    m = np.array([    # m = s x w, with w = [0 0 1]
        versor(np.array([v3,l3-u3,np.zeros(nft)]).T).T,  # m1
        versor(np.array([-v3,u3,np.zeros(nft)]).T).T,    # m2
        versor(np.array([np.zeros(nft),-l3,np.zeros(nft)]).T).T      # m3
    ])     
    def broadcast(v):
        return np.tile(v, (nfp, 1)).T
    l1 = broadcast(l1)
    l2 = broadcast(l2)
    l3 = broadcast(l3)
    u3 = broadcast(u3)
    v3 = broadcast(v3)
    threshold = broadcast(threshold)


    # useful quantities for integration
    sminus = np.array([-((l3-u3)*(l3-u0)+v3*v0)/l1, 
                        -(u3*(u3-u0)+v3*(v3-v0))/l2, -u0]) # eq. (3)
    
    splus = np.array([((u3-l3)*(u3-u0)+v3*(v3-v0))/l1, 
                                 (u3*u0+v3*v0)/l2, l3-u0]) # eq. (3)
    
    t0 = np.array([((u3-l3)*v0+v3*(l3-u0))/l1, (v3*u0-u3*v0)/l2, v0]) # eq. (4)
    
    tplus = np.array([np.sqrt((u3-u0)**2+(v3-v0)**2), 
                    np.sqrt(u0**2+v0**2), np.sqrt((l3-u0)**2+v0**2)]) # eq. (5)

    tminus = np.array([tplus[2],tplus[0], tplus[1]])  # eq. (5)
    
    R0 = np.sqrt(t0**2+w0**2)                # line 1 pp. 1450
    
    Rminus = np.sqrt(tminus**2+w0**2)        # line 2 pp. 1450
    
    Rplus = np.sqrt(tplus**2+w0**2)          # line 2 pp. 1450
    
    # return Vloc, tminus
    
    def get_f2_beta(w0, Rplus, Rminus, splus, sminus, tplus, tminus, t0, R0, threshold):

        # field point not in the plane of triangle
        if abs(w0) >= threshold:
            f2 = log(Rplus + splus) - log(Rminus + sminus) # eq. (11)
            beta = arctan(t0 * splus / (R0 ** 2 + abs(w0) * Rplus)) - arctan(t0 * sminus / (R0 ** 2 + abs(w0) * Rminus)) # eq. (14)

        # field point in the plane of triangle but not aligned with edges
        if abs(w0) < threshold and abs(t0) >= threshold:
            f2 = log((tplus + splus) / (tminus + sminus)) # eq. (15)
            beta = arctan(splus / t0) - arctan(sminus / t0) # eq. (17)

        # field point in the plane of triangle and aligned with edges
        if abs(w0) < threshold and abs(t0) < threshold:
            beta = 0
            f2 = abs(log(splus / sminus)) # abs(lim t->0 eq. 15)
            
        # fix value for point on triangle corners (undocumented)
        if not np.isfinite(f2):
            f2 = 0
        return f2, beta

    # use vectorized functions to speed up calculation.
    f2, beta = np.vectorize(get_f2_beta)(w0, Rplus, Rminus, splus, sminus, tplus, tminus, t0, R0, threshold)
                
    # integral value of '1/r'
    I1 = np.sum(t0*f2-abs(w0)*beta, axis=0) # eq. (19)


    m = np.repeat(m[..., np.newaxis], nfp, axis=-1)
    # integral value of grad(1/r)
    Igradloc = np.array([                                  # eq. (34)
    -m[0][0]*f2[0]-m[1][0]*f2[1]-m[2][0]*f2[2],
    -m[0][1]*f2[0]-m[1][1]*f2[1]-m[2][1]*f2[2],
    -np.sign(w0)*np.sum(beta, axis=0)])     
    Igrad = np.linalg.inv(R) @ Igradloc.transpose((1, 0, 2))
    
    
    return I1, Igrad.transpose((0, 2, 1))

def int_green3d_tri_multiple(Pglo, Vs, require_grad=False):
    '''
    the column vectors of alpha are Propagators of Green's function from V to P
    alpha: [len(Pglo), len(Vs)]
    Igrads: [len(Pglo), len(Vs), 3]
    '''
    alpha = []
    Igrads = []
    for V in tqdm(Vs):
        I, Igrad = int_green3d_tri(Pglo, V)
        alpha.append(I) # Pglo.shape
        Igrads.append(Igrad) # [Pglo.shape, 3]
    alpha = np.array(alpha).T
    Igrads = np.moveaxis(Igrads, 0, 1)
    if require_grad:
        return alpha, Igrads
    return alpha

if __name__ == '__main__':

    import matlab
    from matlab import engine
    import matplotlib.pyplot as plt
    import time
    # test if the result is consistent with Matlab.
    eng = engine.start_matlab()
    s = eng.genpath('matlab')
    eng.addpath(s, nargout=0)
    errors, errors_grad = [], []
    for i in range(1000):
        V = np.random.uniform(-50, 50, (3, 3))
        Pglo = np.random.uniform(-50, 50, (5, 3))
        I_m, Igrad_m = eng.int_green3d_tri(matlab.double(Pglo.tolist()), matlab.double(V.tolist()), nargout=2)
        I, Igrad = int_green3d_tri(Pglo, V)
        # print(Igrad.shape)
        errors.append((I_m - I).mean())
        errors_grad.append((Igrad_m - Igrad).mean())
    plt.plot(errors)
    plt.show()
    plt.plot(errors_grad)
    plt.show()

    # V = np.random.uniform(-50, 50, (200, 3, 3))
    # Pglo = np.random.uniform(-50, 50, (5000, 3))

    # start = time.time()
    # I, Igrad = int_green3d_vectorized(Pglo, V)
    # end = time.time()
    # print(end - start)

    # start = time.time()
    # Igrads = []
    # for v in V:
    #     I, Igrad1 = int_green3d_tri(Pglo, v)
    #     Igrads.append(Igrad1)
    # end = time.time()
    # print(end - start)

    # print(Igrad - np.array(Igrads))