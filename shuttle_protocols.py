import numpy as np
from numpy import pi

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