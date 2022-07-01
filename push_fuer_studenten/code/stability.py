from scipy.linalg import eigvals
from scipy.optimize import root
import numpy as np


def softplus(H,pbase,pcap,pepsilon):
    slope = (1-pbase)/pcap
    return slope*pepsilon*np.log(np.exp(1/pepsilon*(pcap-H))+1)+pbase

def softplus_deriv(H,pbase,pcap,pepsilon):
    slope = (1-pbase)/pcap
    return -slope*np.exp(1/pepsilon*(pcap-H))/(1+np.exp(1/pepsilon*(pcap-H)))

def FP_I(I, *args):
    b = args[0]
    o = args[1]
    g = args[2]
    pcap = args[3]
    pbase = args[4]
    pepsilon = args[5]
    return b*I+o/softplus(I,pbase,pcap,pepsilon)-b*o/g*(1-I) 

def FPparams(p,iguess=0.001):
    pcap = p['pcap']
    pbase = p['pbase']
    pepsilon = p['pepsilon']
    b = p['beta']
    o = p['nu']
    g = p['gamma']
    
    sol = root(FP_I,iguess, args=(b,o,g,pcap,pbase,pepsilon))

    ISTAR = sol.x[0]
    RSTAR = g/o*ISTAR
    SSTAR = g/b*1/softplus(ISTAR,pbase,pcap,pepsilon)
    
    return SSTAR,ISTAR,RSTAR

def jacobian(p):
    pcap = p['pcap']
    pbase = p['pbase']
    pepsilon = p['pepsilon']
    b = p['beta']
    o = p['nu']
    g = p['gamma']
    tau = p['tau']
    SSTAR,ISTAR,RSTAR = FPparams(p)
    spstar = softplus(ISTAR,pbase,pcap,pepsilon)
    spderivstar = softplus_deriv(ISTAR,pbase,pcap,pepsilon)

    row1 = [-b*spstar*ISTAR-o, -b*spstar*SSTAR-o,0,-b*spderivstar*SSTAR*ISTAR]
    row2 = [b*spstar*ISTAR, b*spstar*SSTAR-g, 0, b*spderivstar*SSTAR*ISTAR]
    row3 = [0,2/tau,-2/tau,0]
    row4 = [0,0,2/tau,-2/tau]
    J = [row1,row2,row3,row4]
    return J

def largestEW(p):
    J = jacobian(p)
    EW = eigvals(J)
    return np.max(EW.real)
