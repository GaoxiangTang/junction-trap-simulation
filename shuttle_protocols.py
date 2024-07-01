import numpy as np
from numpy import pi
from scipy import constants
from scipy.misc import derivative
import matplotlib.pyplot as plt
import ndsplines

def qtanhN(t, tspan, qspan=(0, 120), N=5):
    ''' t is a specific time
        tspan is a tuple with the start and end time'''
    t0, tf = tspan
    y0, yf = qspan
    T = tf-t0
    if (t-t0) < T:
        return y0 + max((yf-y0)/2*(np.tanh(N*(2*(t-t0)-T)/T)+np.tanh(N))/np.tanh(N), 0)
    else:
        return yf
    
def qsin(t, tspan, qspan=(0, 120)):
    ''' t is a specific time
        tspan is a tuple with the start and end time'''
    t0, tf = tspan
    y0, yf = qspan
    T = tf-t0
    if t < t0:
        return y0
    elif (t-t0) < T:
        return y0 + max((yf-y0)/2*(1-np.cos(pi*(t-t0)/T)), 0)
    else:
        return yf

def qsta(t, tspan, qspan=(0, 120), f=1):
    ''' t is a specific time
        tspan is a tuple with the start and end time'''
    t0, tf = tspan
    y0, yf = qspan
    T, ti = tf-t0, t-t0 
    s = ti/T
    if s < 0:
        return y0
    elif s <= 1:
        return y0 + (yf-y0)*((1/(2*pi*f)**2)*(60*s - 180*s**2 + 120*s**3) + 10*s**3 - 15*s**4 + 6*s**5)
    else:
        return yf
    
def qsta2(t, tspan, qspan=(0, 120), f=1):
    ''' t is a specific time
        tspan is a tuple with the start and end time'''
    t0, tf = tspan
    y0, yf = qspan
    T, ti = tf-t0, t-t0 
    s = ti/T
    if s < 0:
        return y0
    elif s <= 1:
        return y0 + (yf-y0)*((1/(2*pi*f)**2)*(2520*s**3-12600*s**4+22680*s**5-17640*s**6+5040*s**7)
                             + 126*s**5-420*s**6+540*s**7-315*s**8+70*s**9)
    else:
        return yf
    
def qtrig(t, tspan, qspan=(0, 120), f=1.2e6, mu=1):
    ''' args should be f (axialfreq in Hz), mu (mass ratio)'''
    t0, tf = tspan
    y0, yf = qspan
    T, ti, d = tf-t0, t-t0, yf-y0
    s = ti/T
    Op, On = 2*np.pi*f*(1+1/mu+(1-1/mu+1/mu**2)**0.5)**0.5, 2*np.pi*f*(1+1/mu-(1-1/mu+1/mu**2)**0.5)**0.5
    b3 = -49*((T*Op)**2 - (5*np.pi)**2)*((T*On)**2 - (5*np.pi)**2)/(2048*(T*T*On*Op)**2)
    b4 = 5*((T*Op)**2 - (7*np.pi)**2)*((T*On)**2 - (7*np.pi)**2)/(2048*(T*T*On*Op)**2)
    
    if s < 0:
        return y0
    elif s <= 1:
        return y0 + d*(0.5 + (-9/16+2*b3+5*b4)*np.cos(np.pi*s) + 1/16*(1-48*b3-96*b4)*np.cos(3*np.pi*s) + b3*np.cos(5*np.pi*s) + b4*np.cos(7*np.pi*s))
    else:
        return yf
    
def d_t(t, tspan, qspan=(0, 120)):
    ''' t is a specific time
        tspan is a tuple with the start and end time'''
    t0, tf = tspan
    y0, yf = qspan
    T, ti = tf-t0, t-t0 
    s = ti/T
    if s < 0:
        return y0
    elif s <= 1:
        return y0 + (yf-y0)*(10*s**3 - 15*s**4 + 6*s**5)
    else:
        return yf
    
def const_speed(t, tspan, qspan=(0, 120)):
    ''' t is a specific time
        tspan is a tuple with the start and end time
    '''
    t0, tf = tspan
    y0, yf = qspan
    T, ti = tf-t0, t-t0 
    s = ti/T
    if s < 0:
        return y0
    elif s <= 1:
        return y0 + (yf-y0)*s
    else:
        return yf
    
def split_sta_lukeqi(s, a10=-780, a11=127):
    f0 = 5e5*2*np.pi
    mass = 171 * constants.atomic_mass
    eps0 = constants.epsilon_0
    K = constants.e**2/(4*np.pi*eps0)
    a0 = 0.5*mass*f0**2
    l0 = 2*a0/mass*np.array([1, 3]) ## (-, +)
    lf = 2*a0/mass*np.array([1, 1.00002])
    gp, gm = (l0/lf)**0.25
    rho_m_t = lambda s, a10=a10, a11=a11: 1-(126*(1-gm) + a10 + 5*a11)*s**5 + \
        (420*(1-gm) + 5*a10 + 24*a11)*s**6 - (540*(1-gm) + 10*a10 + 45*a11)*s**7 + \
            (315*(1-gm) + 10*a10 + 40*a11)*s**8 - (70*(1-gm) + 5*a10 + 15*a11)*s**9 + a10*s**10 + a11*s**11
     
    rho_m = rho_m_t(s)
    d2rho_pdt2 = derivative(rho_m_t, s, n=2)
    lm_t = l0[1]*rho_m**(-4) - d2rho_pdt2/rho_m
    lp_t = l0[0]
    d_ideal = (4*K/(mass*(lm_t-lp_t)))**(1/3) ## SI units [m]
    a_ideal = (mass/8*(3*lm_t - 5*lp_t)) ## SI units [J/m^2]
    b_ideal = (2*K*(d_ideal)**(-5) - 2*a_ideal*(d_ideal)**(-2)) ## SI units [J/m^4]
    return a_ideal*1e-6*2/constants.e, b_ideal*1e-12*24/constants.e, d_ideal

def split_sta_palmero(s, a10=-637.36273847, a11=101.30581049, f0=0.15):
    f0 = f0 * 1e6*2*np.pi
    mass = 171 * constants.atomic_mass
    eps0 = constants.epsilon_0
    K = constants.e**2/(4*np.pi*eps0)
    a0 = 0.5*mass*f0**2
    l0 = 2*a0/mass*np.array([1, 3]) ## (-, +)

    lf = 2*a0/mass*np.array([1, 1.002])
    # ebeta = 6e4
    # beta = constants.e * ebeta / 24 * 1e12
    # d=(2*K/beta)**(1/5)
    # lf = np.array([3*beta * d**2/mass, (3*beta*d**2+4*K/d**3)/mass])

    gp, gm = (l0/lf)**0.25
    rho_m_t = lambda s, a10=a10, a11=a11: 1-(126*(1-gm) + a10 + 5*a11)*s**5 + (420*(1-gm) + 5*a10 + 24*a11)*s**6 - (540*(1-gm) + 10*a10 + 45*a11)*s**7 + (315*(1-gm) + 10*a10 + 40*a11)*s**8 - (70*(1-gm) + 5*a10 + 15*a11)*s**9 + a10*s**10 + a11*s**11
    rho_p_t = lambda s: 1+(gp-1)*(126*s**5-420*s**6+540*s**7-315*s**8+70*s**9)
    rho_m = rho_m_t(s)
    rho_p = rho_p_t(s)
    ss = np.linspace(0, 1, 1000)
    rho_m_t = ndsplines.make_interp_spline(ss, rho_m_t(ss))
    rho_p_t = ndsplines.make_interp_spline(ss, rho_p_t(ss))

    d2rho_mdt2 = rho_m_t.derivative(0).derivative(0)(s)
    d2rho_pdt2 = rho_p_t.derivative(0).derivative(0)(s)
    lm_t = l0[1]*rho_m**(-4) - d2rho_mdt2/rho_m
    lp_t = l0[0]*rho_p**(-4) - d2rho_pdt2/rho_p
    # plt.plot(s, lm_t)
    # plt.plot(s, lp_t)
    # plt.show()
    d_ideal = (4*K/(mass*(lm_t-lp_t)))**(1/3) ## SI units [m]
    a_ideal = (mass/8*(3*lm_t - 5*lp_t)) ## SI units [J/m^2]
    b_ideal = (2*K*(d_ideal)**(-5) - 2*a_ideal*(d_ideal)**(-2)) ## SI units [J/m^4]
    return a_ideal*1e-6*2/constants.e, b_ideal*1e-12*24/constants.e, d_ideal


def split_naive(s):
    f0 = 5e5*2*np.pi
    mass = 171 * constants.atomic_mass
    eps0 = constants.epsilon_0
    K = constants.e**2/(4*np.pi*eps0)
    a0 = 0.5*mass*f0**2

    return 
    