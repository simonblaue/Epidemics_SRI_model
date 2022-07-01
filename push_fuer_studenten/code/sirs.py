import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class model:
    def __init__(
        self,
        y0, beta, gamma, nu, tau, pbase, pcap, pepsilon, s, w,
        tmin, tmax, stepsize, maxstep, direction, threshold
    ):
        
        self.y0 = y0
        self.beta = beta
        self.gamma = gamma
        self.nu = nu
        self.tau = tau 
        
        self.pbase = pbase
        self.pcap = pcap
        self.pepsilon = pepsilon
        self.s = s
        self.w = w
        
        self.tmin = tmin
        self.tmax = tmax
        self.stepsize = stepsize
        self.maxstep = maxstep
        self.direction = direction
        self.threshold = threshold
        
    
    def FOI_softplus(self,M,I):
        return self.softplus(M)*I*self.beta
    
    def softplus(self,M):
        slope = (1-self.pbase)/self.pcap
        return slope*self.pepsilon*np.log(np.exp(1/self.pepsilon*(self.pcap-M))+1)+self.pbase
    
    def seasonalforcing(self,t):
        return (1+self.s*np.sin(self.w*t))
    
    def i_peaks(self,t,y):
        FOI = self.FOI_softplus(y[4],y[1])
        return FOI*y[0]*self.seasonalforcing(t) - self.gamma*y[1]
    
    def seasonal_peaks(self,t,y):
        return np.cos(self.w*t)
    
    def event_threshold(self,t,y):
        return y[1]-self.threshold
        
    def fun(self,t,y):
        S,I,R,M1,M = y

        FOI = self.FOI_softplus(M,I)
        
        dS = -FOI*S*self.seasonalforcing(t) + self.nu*R 
        dI = FOI*S*self.seasonalforcing(t) - self.gamma*I
        dR = self.gamma*I - self.nu*R
        dM1 = 2/self.tau*(I-M1)
        dM = 2/self.tau*(M1-M)
        return [dS,dI,dR,dM1,dM]

    def run(self):
        event1 = lambda t,x: self.i_peaks(t,x)
        if self.direction:   
            event1.direction = -1
            
        event2 = lambda t,x: self.seasonal_peaks(t,x)
        event2.direction = -1
        
        event3_thres = lambda t,x: self.event_threshold(t,x)
        event3_thres.direction = 1
        
        event4_thres = lambda t,x: self.event_threshold(t,x)
        event4_thres.direction = -1

        toutput = np.arange(self.tmin, self.tmax, self.stepsize)
        res = solve_ivp(self.fun, (self.tmin,self.tmax), self.y0, t_eval=toutput, max_step = self.maxstep,events=[event1,event2,event3_thres,event4_thres])
        self.times = res['t']
        self.data = res['y']
        self.events = res['y_events']
        self.t_events = res['t_events']
        return self.times, self.data, self.events, self.t_events