import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as int
from sympy import difference_delta
import general_methods as gm
from scipy.signal import find_peaks
from scipy.integrate import simps
# plt.rc('font', size=18) #controls default text size


class SIRModels:
    def __init__(self, gamma=0.1, beta=0.5, mue=0, nue=1/100, p_base = 0.1, s=0, T=25, dI_thresh = 0.001, reductionfactor = 1/10, p_cap = 0.06):
        self.gamma = gamma
        self.beta = beta
        self.mue = mue
        self.nue = nue
        self.p_base = p_base # Strongest possible factor that reduces the transmission rate
        self.p_cap = p_cap # percieved risk beyond which no further reduction of the reansmission rate takes palce
        self.epsilon = 1e-4 # Curvature parameter
        self.s = s # Amplitude of seasonal forcing
        self.omega = 2*np.pi/360 #Frequency of yearly seasonsl variation
        self.T = T # mittlerer Erinnerungszeit
        self.P = lambda H: self.p_base+(1-self.p_base)/self.p_cap*self.epsilon*np.log(1+np.exp(1/self.epsilon*(self.p_cap-H)))
        self.Gamma = lambda t : 1+self.s*np.cos(self.omega*t)
        self.dP = lambda H : -(((1-self.p_base)/self.p_cap)*np.exp((self.p_cap-H)/self.epsilon))/(1+np.exp((self.p_cap-H)/self.epsilon))
        self.Fix = self.FindFixpoint(self.MemoryIncrementForStability)
        
        self.direction= True
        self.threshold = 0.03 #only for eventthreshold of joel
        # new params
        self.Ti = 25

        #zu variieren
        self.dI_thresh = dI_thresh
        self.d_pcap_min = self.p_cap*reductionfactor
        self.delta_pcap = None

    def ClassicIncrement (self,t,state):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R) vector
            - beta    - Required: transmission rate
            - gamma   - Required: recovery rate
        Return: 
            - temporal derivative of the sate vector
        """
        dS = -self.beta*state[1]*state[0]
        dI = self.beta*state[1]*state[0] - self.gamma*state[1]
        dR = self.gamma*state[1]
        return [dS,dI,dR,0,0]

    def VitalIncrement(self,t,state):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R) vector
            - beta    - Required: transmission rate
            - gamma   - Required: recovery rate
        Return: 
            - temporal derivative of the sate vector
        """
        dS = -self.beta*state[1]*state[0] + self.mue - self.mue*state[0] + self.nue*state[2]
        dI = self.beta*state[1]*state[0] - self.gamma*state[1] - self.mue*state[1]
        dR = self.gamma*state[1] - self.mue*state[2] - self.nue*state[2]
        return [dS,dI,dR,0,0]

    def SeasonalyIncrement(self, t, state):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R,H1,H) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
            - t    - Required: transmission rate

        Return: 
            - state after one step
        """
        P = self.P(state[4])
        Gamma = self.Gamma(t)
        S , I, R, H1, H = state

        dS = -self.beta*P*Gamma*I*S + self.nue*R
        dI = self.beta*P*Gamma*I*S - self.gamma*I
        dR = self.gamma*I - self.nue*R
        dH1 = 2/self.T*(I-H1)
        dH = 2/self.T*(H1-H)

        return [dS,dI,dR,dH1,dH]
    
    def DifferentPIncrement(self, t, state):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R,H1,H) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
            - t    - Required: transmission rate

        Return: 
            - state after one step
        """
        
        Gamma = self.Gamma(t)
        S , I, R, H1, H = state

        dH = 2/self.T*(H1-H)
        
        P = self.P_new(H,dH)

        dS = -self.beta*P*Gamma*I*S + self.nue*R
        dI = self.beta*P*Gamma*I*S - self.gamma*I
        dR = self.gamma*I - self.nue*R
        dH1 = 2/self.T*(I-H1)
        
        Hdot.append(dH)
        Idot.append(dI)

        return [dS,dI,dR,dH1,dH]

    def MemoryIncrement(self, t=0, state = [1,0,0,0,0]):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R,H1,H) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
            - t    - Required: transmission rate

        Return: 
            - state after one step
        """
        P = self.P(state[4])
        S , I, R, H1, H = state

        dS = -self.beta*P*1*I*S + self.nue*R
        dI = self.beta*P*1*I*S - self.gamma*I
        dR = self.gamma*I - self.nue*R
        dH1 = 2/self.T*(I-H1)
        dH = 2/self.T*(H1-H)

        return [dS,dI,dR,dH1,dH]

    def AdaptivePIncrement(self, t, state):

        S , I, R, H1, H, Hi1, Hi = state
        Gamma = self.Gamma(t)
        P = self.P_new(H,Hi)
        dS = -self.beta*P*Gamma*I*S + self.nue*R
        dI = self.beta*P*Gamma*I*S - self.gamma*I
        dR = self.gamma*I - self.nue*R
        dH1 = 2/self.T*(I-H1)
        dH = 2/self.T*(H1-H)
        dHi1 = 2/self.Ti*(dI-Hi)
        dHi = 2/self.Ti*(Hi1-Hi)
        Idot.append(dI)
        Hdot.append(Hi)
        return [dS,dI,dR,dH1,dH,dHi1,dHi]

    def Jacobi(self,state, func):
        """
        The function returns the Jacobian.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
        Return: 
            - JacobiMatrix
        """
        if func.__name__ == 'MemoryIncrement':
            P = self.P(state[4])
            dP = self.dP(state[4])
        elif func.__name__ == 'DifferentPIncrement':
            if self.delta_pcap is None:
                raise ValueError(
                    "self.delta_pcap is None but needs to be of shape float"
                )
            else:
                P = self.P_new(state[4], 0)
                dP = self.dPdH_new(state[4],0)


        Jac = np.zeros((4,4))
        Jac[0,0] = - self.beta*P*state[1]-self.nue
        Jac[0,1] = - self.beta*P*state[0]-self.nue

        Jac[0,3] = - self.beta*dP*state[0]*state[1]
        Jac[1,0] = self.beta*P*state[1]
        Jac[1,1] = self.beta*P*state[0]-self.gamma

        Jac[1,3] = self.beta*dP*state[0]*state[1]

        Jac[2,1] = 2/self.T
        Jac[2,2] = - 2/self.T



        Jac[3,2] = 2/self.T
        Jac[3,3] = - 2/self.T
        return Jac
########## FIXPOINTS ############
    def FindFixpoint(self,fun,I0 = 0.9):
        Fix = opt.root(fun,I0)
        I = Fix.x
        S = self.gamma/(self.beta*self.P(I))
        R = self.gamma/self.nue*I
        H = I
        H1 = I
        Fix = np.array([S,I,R,H1,H])
        return Fix

    def FindFixpointDifferentP(self,fun,I0 = 0.001):
        Fix = opt.root(fun, I0)
        I = Fix.x
        H = I
        S = self.gamma/(self.beta*self.P_new(H,0))
        R = self.gamma/self.nue*I
        H1 = I
        Fix = np.array([S,I,R,H1,H])
        return Fix

    def MemoryIncrementForStability(self, I):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R,H1,H) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
            - t    - Required: transmission rate

        Return: 
            - state after one step
        """
        return self.nue/(self.gamma+self.nue)*(1-self.gamma/(self.beta*self.P(I)))-I

    def DifferentPIncrementForStability(self, H):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R,H1,H) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
            - t    - Required: transmission rate

        Return: 
            - state after one step
        """
        return self.nue/(self.gamma+self.nue)*(1-self.gamma/(self.beta*self.P_new(H,0)))-H

######### Helping Functions ##########
    def P_new(self,H,dH):
        if not self.delta_pcap:
            if dH<=0:
                delta_p_cap = self.p_cap/2
            elif dH < self.dI_thresh:
                delta_p_cap = -(self.p_cap/2-self.d_pcap_min)/self.dI_thresh*dH+self.p_cap/2
            else:
                delta_p_cap = self.d_pcap_min
        else:
            delta_p_cap = self.delta_pcap

        pcap1 = self.p_cap/2-delta_p_cap
        pcap2 = self.p_cap/2+delta_p_cap
        epsilon = (pcap2-pcap1)
        if H>=pcap2:
            return self.p_base
        elif H<=pcap1: 
            return 1
        elif H>pcap1:
            #if pcap1 == 0:
            #    print('pcap1=',pcap1,'pcap2=',pcap2,'H=', H,'dH=', dH)
            return (1-self.p_base)/(1+np.exp(epsilon*((1/(pcap2-H)+1/(pcap1-H)))))+self.p_base

    def dPdH_new(self,H,dH):
        if not self.delta_pcap:
            if dH<=0:
                delta_p_cap = self.p_cap/2
            elif dH < self.dI_thresh:
                delta_p_cap = -(self.p_cap/2-self.d_pcap_min)/self.dI_thresh*dH+self.p_cap/2
            else:
                delta_p_cap = self.d_pcap_min
        else:
            delta_p_cap = self.delta_pcap
        pcap1 = self.p_cap/2-self.delta_pcap
        pcap2 = self.p_cap/2+self.delta_pcap
        epsilon = (pcap2-pcap1)
        if H>=pcap2:
            return 0
        elif H<=pcap1: 
            return 0
        elif H>pcap1:
            exp = np.exp(epsilon*((1/(pcap2-H)+1/(pcap1-H))))
            return (1-self.p_base)*epsilon*exp*(1/(pcap2-H)**2+1/(pcap1-H)**2)/(1+exp)**2

    def event_threshold_p_adapt(self,t,y):
        # y = [dS,dI,dR,dH1,dH,dHi1,dHi]
        FOI = self.P_new(y[4],y[6])
        return self.beta*FOI*y[0]*y[1]*self.seasonalforcing(t) - self.gamma*y[1]
    
    def different_p_event(self,t,y):
        # y = [dS,dI,dR,dH1,dH,dHi1,dHi]
        FOI = self.P_new(y[4],2/self.T*(y[3]-y[4]))
        return self.beta*FOI*y[0]*y[1]*self.seasonalforcing(t) - self.gamma*y[1]
    
    def FOI_softplus(self,M,I):
        return self.softplus(M)*I*self.beta

    def softplus(self,M):
        slope = (1-self.p_base)/self.p_cap
        return slope*self.epsilon*np.log(np.exp(1/self.epsilon*(self.p_cap-M))+1)+self.p_base

    def seasonalforcing(self,t):
        return (1+self.s*np.cos(self.omega*t))

    def i_peaks(self,t,y):
        FOI = self.FOI_softplus(y[4],y[1])
        return FOI*y[0]*self.seasonalforcing(t) - self.gamma*y[1]

    def event_threshold(self,t,y):
        return y[1]-self.threshold

######## Stability #############
    def CheckStabilityNumerically(self,Func,T_max, y0,eps_max=1e-8,eps_minmax=0.004):
        """
        Returns  True if Convergences is achieved
        """

        #evtl eine vorgegebene schrittdichte
        SolveDict = int.solve_ivp(Func,[0,T_max],y0=y0, max_step = 1)#,events=[event1])
        
        t = SolveDict.t
        state = SolveDict.y
        
        I = state[1,:]
        max_ind = find_peaks(I)[0]
        min_ind = find_peaks(-I)[0]
        #print(I[max_ind])
        if max_ind.shape[0]<50:
            #print("Almost no maxima or minima")
            return True
        Difference = I[max_ind][-7:-1]-I[min_ind][-7:-1]
        PeakDifference = I[max_ind][1:-1]-I[max_ind][0:-2]
        if all(PeakDifference<0) and Difference[-1]<0.05:
            #print(PeakDifference[-5:-1], eps_max,"\n",Difference,eps_minmax)
            return True
        elif any(PeakDifference[-8:]>eps_max) or all(Difference>eps_minmax):
            #print(PeakDifference[-5:-1], eps_max,"\n",Difference,eps_minmax)
            return False
        else:
            #print(PeakDifference.shape,PeakDifference[-5:-1], eps_max,"\n",Difference,eps_minmax)
            return True
        #min_shapes = np.min([max_ind.shape[0],min_ind.shape[0]])
            

        #Difference = I[max_ind[:min_shapes]]-I[min_ind[:min_shapes]]
        if Difference.shape[0]<10:
            #print("Almost no maxima or minima")
            return True
        elif any(Difference[-5:-1]>eps): #or Difference[-1]>Difference[-2]:
            #print(Difference[-5:-1], eps)
            return False
        else: 
            #print(Difference[-1],eps,Difference[-2])
            return True
        """
        if Func.__name__ == 'AdaptivePIncrement':
            event1 = lambda t,x: self.event_threshold_p_adapt(t,x)    
        if Func.__name__ == 'DifferentPIncrement':
            event1 = lambda t,x: self.different_p_event(t,x)     
        else:
            event1 = lambda t,x: self.i_peaks(t,x)
        event1.direction = -1
        t_events = SolveDict.t_events[0]
        
        Period = t_events[-1]-t_events[-2]
        NumberPeriod = 10
        MinMeanTime = t[-1]-NumberPeriod*Period
        Mean = np.mean(I[t>MinMeanTime])


        DistToMean = np.abs(I-Mean)
        #plt.plot(DistToMean)
        #plt.vlines([MinMeanTime,(t[-1]-2*NumberPeriod*Period)], ymin=0, ymax=0.02)
        #plt.show()
        #plt.close()
        time_ind = (t<t_events[-1])
        last_ind =  t>(t[-1]-NumberPeriod*Period)
        secondLast_ind = t>(t[-1]-2*NumberPeriod*Period)
        last_ind = np.all(np.array([time_ind,last_ind]),0)
        secondLast_ind = np.all(np.array([time_ind,secondLast_ind]),0)
        Last = np.sum(DistToMean[last_ind])
        SecondLast = 1/2*np.sum(DistToMean[secondLast_ind])
        print(Last,SecondLast)
        if Last+eps > SecondLast:
            return False
        else: 
            return True
        """

    def CheckStabilityAnalytically(self,func):
        if func.__name__ == 'MemoryIncrement':
            self.Fix = self.FindFixpoint(self.MemoryIncrementForStability)
        elif func.__name__ == 'DifferentPIncrement':
            self.Fix = self.FindFixpointDifferentP(self.DifferentPIncrementForStability)
        Jacobi = self.Jacobi(self.Fix, func)
        w = np.linalg.eigvals(Jacobi)
        
        w_real=np.real(w)
        max = np.max(w_real)
        if max>0:
            return False
        else:
            return True

    def getInterestingData(self,func,T_min,T_max,y0, plot = False):
        
        if func.__name__ == 'AdaptivePIncrement':
            event1 = lambda t,x: self.event_threshold_p_adapt(t,x)    
        if func.__name__ == 'DifferentPIncrement':
            event1 = lambda t,x: self.different_p_event(t,x)     
        else:
            event1 = lambda t,x: self.i_peaks(t,x)
        event1.direction = -1 
        
        t = np.arange(T_min,T_max,1)
        
        SolveDict = int.solve_ivp(func,[0,T_max],y0, max_step = 1,t_eval=t, events=[event1])

        t = SolveDict.t
        state = SolveDict.y
        events0 = SolveDict.y_events[0]
        t_events = SolveDict.t_events[0]

        events0 = events0[t_events>T_min,:]
        t_events = t_events[t_events>T_min]
    
        I_max = np.mean(events0[:,1])
        T = np.mean(t_events[1:]-t_events[:-1])

        Years = np.arange(0,T_max-T_min,365)
        TotalYears = np.shape(Years)
        TotalInfected = np.zeros(TotalYears)
        for (year,day_ind) in enumerate(Years):
            #print(TotalInfected.shape, state.shape, day_ind, year)
            TotalInfected[year] = simps(state[1, :day_ind+1])*self.gamma
        InfectedperYear = TotalInfected[1:]-TotalInfected[0:-1]
        MeanIperYear = np.mean(InfectedperYear)
        
        if plot:
            fig, axes = plt.subplots(1,1)
            axes.plot(t[t>T_min],state[1,t>T_min], label = 'Infected')
            #axes.scatter(t_events[t_events>T_min],events0[t_events>T_min,1],label='Maxima',c='red') #t_events>T_min
            axes.set_xlabel(r"Time $t$ in days")
            axes.set_ylabel(r"Currently infected $I$")
            ax2 = axes.twinx()
            ax2.set_ylabel('Accumulated infections')
            ax2.plot((np.arange(T_min,T_max,365))[1:],InfectedperYear,label = 'Infected per year', color = 'green')
            ax2.plot((np.arange(T_min,T_max,365)),TotalInfected,label = 'Total Infected', color = 'red')
            axes.legend(loc = 'center left')
            ax2.legend(loc = 'center right')
            plt.show()
            plt.close()


        return I_max,T, MeanIperYear

class plots():
    def PlotTrajectoryAndFix(Model,Func,save=None, y0 =[0.99,0.01,0,0,0],T_min=0, T_max=10000):
        plt.figure(figsize=(12,10))
        if Func.__name__ == 'MemoryIncrement':
            Model.Fix = Model.FindFixpoint(Model.MemoryIncrementForStability)
        elif Func.__name__ == 'DifferentPIncrement':
            Model.Fix = Model.FindFixpointDifferentP(Model.DifferentPIncrementForStability)
        SolveDict = int.solve_ivp(Func,[0,T_max],y0, max_step = 1)
        #int(np.shape(X) , X)
        t = SolveDict.t
        state = SolveDict.y
        #f = Model.P(state)
        #plt.plot(t,f)
        #plt.plot(t,state[0,:], label = 'S')
        #plt.plot(t,state[1,:], label = 'I')
        #plt.plot(t,state[2,:], label = 'R')
        
        plt.plot(state[0,t>T_min],state[1,t>T_min],label = 'Trajectory')
        plt.scatter(Model.Fix[0],Model.Fix[1], label = 'Fixpoint',color = 'r')
        #plt.hlines(y = 0 , xmin=np.min(state[0,:]),xmax=np.max(state[0,:]), color = 'black', linestyles='dotted')
        plt.xlabel('S')
        plt.ylabel('I')
        plt.legend()
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()
        return t,state

    def PlotSRITime(model,func, y0=[0.99,0.01,0,0,0], save=None, T_min = 0, T_max=10000):
        plt.figure(figsize=(12,10))
        if func.__name__ == 'AdaptivePIncrement':
            event1 = lambda t,x: model.event_threshold_p_adapt(t,x)    
        if func.__name__ == 'DifferentPIncrement':
            event1 = lambda t,x: model.different_p_event(t,x)     
        else:
            event1 = lambda t,x: model.i_peaks(t,x)
        event1.direction = -1 
        
        t = np.arange(T_min,T_max, 1)    
        SolveDict = int.solve_ivp(func,[0,T_max], y0, max_step = 1, events=[event1], t_eval=t)#,event2,event3_thres,event4_thres])
        t = SolveDict.t
        state = SolveDict.y
        events0 = SolveDict.y_events[0]
        t_events = SolveDict.t_events[0]
        #events2_ind = find_peaks(state[1,t_events>T_min])[0]
        #print(events.shape,t_events.shape)
        #plt.plot(t,state[0,:], label = 'S')
        plt.plot(t[t>T_min],state[1,t>T_min], label = 'I')
        plt.scatter(t_events[t_events>T_min],events0[t_events>T_min,1],label='Maxima',c='red') #t_events>T_min
        #plt.scatter(t[events2_ind],state[1,events2_ind],label='Maxima2nd',c='green')
        #plt.plot(t,state[2,:], label = 'R')
        #plt.ylim(0.02025, 0.0204)
        plt.xlabel(r"Time $t$ in days")
        plt.ylabel("Infected fraction $I$ ")
        #plt.legend()

        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()
        return t,state

    def PlotStabilityAnalytically(Model,func,save = None, dT=0.1, MeanTMax=60, dp=0.0001, pBaseMax=0.25):
        Ts=np.arange(1,MeanTMax,dT)
        p_bases = np.arange(0.01,pBaseMax,dp)
        Stable = np.ones((np.shape(Ts)[0],np.shape(p_bases)[0]))
        for j,Model.T in enumerate(Ts):
            for i, Model.p_base in enumerate(p_bases):
                Stable[j, i] = Model.CheckStabilityAnalytically(func)
                gm.progress_featback.printProgressBar(j*np.shape(p_bases)[0]+i,np.shape(p_bases)[0]*np.shape(Ts)[0])
        im = plt.imshow(Stable[-1:0:-1,:],vmin=0,vmax=1,extent=[p_bases.min(),p_bases.max(),Ts.min(),Ts.max()],aspect='auto')
        
        #plt.title('Stability Diagramm', fontweight ="bold")
        plt.xlabel(r'Stubberness $ p_{base}$')
        plt.ylabel(r'Mean memory $T$')
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()
        return Stable  

    def PlotStabilityNumerically(model, func, T_max=6000, y0=[0.99,0.01,0,0,0], save = None, dT=0.5,dp=0.01, pBaseMax = 0.5, MeanTMax = 100, eps = 0.001):
        Ts=np.arange(1,MeanTMax,dT)
        p_bases = np.arange(0.01,pBaseMax,dp)
        StabilityResults = np.zeros((Ts.shape[0],p_bases.shape[0]))

        for i,model.p_base in enumerate(p_bases):
            for j,model.T in enumerate(Ts):
                StabilityResults[j,i]=model.CheckStabilityNumerically(func,T_max,y0)                
                gm.progress_featback.printProgressBar(i*np.shape(Ts)[0]+j,np.shape(p_bases)[0]*np.shape(Ts)[0])
        im = plt.imshow(StabilityResults[-1:0:-1,:],vmin=0,vmax=1,extent=[p_bases.min(),p_bases.max(),Ts.min(),Ts.max()],aspect='auto')
        #plt.title('Stability Diagramm', fontweight ="bold")
        plt.xlabel(r'Stubberness $ p_{base}$')
        plt.ylabel(r'Mean memory $T$')
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()    
        return StabilityResults  

    def PlotBifurcation(model, func, save=None):
        peaks_list = []
        time_list = []
        precision = 100

        event1 = lambda t,x: model.i_peaks(t,x)  
        event1.direction = -1

        event3_thres = lambda t,x: model.event_threshold(t,x)
        event3_thres.direction = 1
        
        event4_thres = lambda t,x: model.event_threshold(t,x)
        event4_thres.direction = -1

        eventsAdaptivP = lambda t,x: model.event_threshold_p_adapt(t,x)
        eventsAdaptivP.direction = -1

        plt.figure(figsize=(10,10))
        s_array = np.linspace(0,1,precision)
        for (i,s) in enumerate(s_array):
            model.s = s
            S = np.random.random() 
            if func.__name__ == 'AdaptivePIncrement':
                solve_dict_AdaptivePIncrement = int.solve_ivp(func,[0,30000], [S,1-S,0.0, 0, 0,0,0],max_step=1, events=[eventsAdaptivP])#,event3_thres,event4_thres])
                I = solve_dict_AdaptivePIncrement.y[1,:]
                t = solve_dict_AdaptivePIncrement.t
                events = solve_dict_AdaptivePIncrement.y_events[0][-40:]
                t_events = solve_dict_AdaptivePIncrement.t_events[0][-40:]
                one_s_list = [s for _ in range(events.shape[0])]
                plt.scatter(one_s_list,events[:,1],marker=',',lw=0, s=1,c = t_events)
            else:
                solve_dict_seasonal = int.solve_ivp(func,[0,20000], [S,1-S,0.0, 0, 0], max_step=1, events=[event1])#,event3_thres,event4_thres])
                I = solve_dict_seasonal.y[1,:]
                t = solve_dict_seasonal.t
                events = solve_dict_seasonal.y_events[0][-40:]
                t_events = solve_dict_seasonal.t_events[0][-40:]
                one_s_list = [s for _ in range(events.shape[0])]
                plt.scatter(one_s_list,events[:,1],marker=',',lw=0, s=1,c = t_events)
            gm.progress_featback.printProgressBar(i,precision)
        #print((events))
        plt.ylabel('I')
        plt.xlabel('s')
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()     
    
    def PlotPH(model,func, y0=[0.99,0.01,0,0,0],save=None, T_min = 0, T_max=10000):
        plt.figure(figsize=(20,15))
        plt.rc('font', size=18) 
        t = np.arange(T_min,T_max, 1)        
        SolveDict = int.solve_ivp(func,[0,T_max], y0, max_step = 1, t_eval=t)
        t = SolveDict.t
        state = SolveDict.y

        if func.__name__ == 'AdaptivePIncrement':
            H = state[4,:]
            Hi = state[6,:]
            P_H = []
            for h,hi in zip(H,Hi):
                P_H.append(model.P_new(h,hi))
        elif func.__name__ == 'DifferentPIncrement':
            H = state[4,:]
            dH = 2/model.T*(state[3,:]-state[4,:])
            P_H = []
            for h,dh in zip(H,dH):
                P_H.append(model.P_new(h,dh))
        else:
            H = state[4,:]
            P_H = []
            for h in H:
                P_H.append(model.P(h))
        P_H=np.array(P_H)
        plt.scatter(H[t>T_min],P_H[t>T_min], c = t[t>T_min])#, lw = 0.5, s = 2)
        plt.colorbar(label=r'Time $t$ in days')
        plt.xlabel(r'Perceived Risk $H$')
        plt.ylabel(r'Action $P(H,\dot{H})$')
        
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()
        return t,state

    def PlotInterestingQuantities(model,func, y0=[0.99,0.01,0,0,0],save=None, T_min = 1000, T_max=5000, dpSteps = 10, dHSteps = 10):
        
        Delta_pcap_mins=np.linspace(model.p_cap/20,model.p_cap/2, dpSteps)
        dI_threshs = np.linspace(0.01,1/20,dHSteps)
        
        #StabilityResults = np.zeros((Delta_pcap_mins.shape[0],dI_threshs.shape[0]), dtype=bool)
        MeanPeakHight = np.zeros((Delta_pcap_mins.shape[0],dI_threshs.shape[0]))
        MeanPeriod = np.zeros((Delta_pcap_mins.shape[0],dI_threshs.shape[0]))
        I_tot = np.zeros((Delta_pcap_mins.shape[0],dI_threshs.shape[0]))
        
        for i,model.d_pcap_min in enumerate(Delta_pcap_mins):
            for j,model.dI_thresh in enumerate(dI_threshs):
                
                MeanPeakHight_i, MeanPeriod_i, I_tot_i = model.getInterestingData(func,T_min,T_max,y0)
                MeanPeakHight[i,j] = MeanPeakHight_i
                MeanPeriod[i,j] = MeanPeriod_i
                I_tot[i,j] = I_tot_i       
                gm.progress_featback.printProgressBar(i*np.shape(dI_threshs)[0]+j,np.shape(Delta_pcap_mins)[0]*np.shape(dI_threshs)[0])
        np.save('data.npy',(MeanPeakHight,MeanPeriod,I_tot))
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (18,5))
        im0 = axes[0].imshow(MeanPeriod[-1:0:-1,:],vmin=MeanPeriod.min(),vmax=MeanPeriod.max(),extent=[dI_threshs.min(),dI_threshs.max(),Delta_pcap_mins.min(),Delta_pcap_mins.max()],aspect='auto')
        axes[0].set_title('Period', fontweight ="bold")
        axes[0].set_xlabel(r'Hysteresis Strength $\dot{H}_{Hyst}$')
        axes[0].set_ylabel(r'Society inertia $\Delta p_{\mathrm{cap,min}}$')
        fig.colorbar(im0, ax = axes[0],orientation='vertical')

        im1 = axes[1].imshow(MeanPeakHight[-1:0:-1,:],vmin=MeanPeakHight.min(),vmax=MeanPeakHight.max(),extent=[dI_threshs.min(),dI_threshs.max(),Delta_pcap_mins.min(),Delta_pcap_mins.max()],aspect='auto')
        axes[1].set_title('Peak Hight', fontweight ="bold")
        axes[1].set_xlabel(r'Hysteresis Strength $\dot{H}_{Hyst}$')
        #axes[1].set_ylabel(r'Society inertia $\Delta p_{\mathrm{cap,min}}$')
        fig.colorbar(im1, ax = axes[1],orientation='vertical')
        axes[1].set_yticks([], [])
                
        im2 = axes[2].imshow(I_tot[-1:0:-1,:],vmin=I_tot.min(),vmax=I_tot.max(),extent=[dI_threshs.min(),dI_threshs.max(),Delta_pcap_mins.min(),Delta_pcap_mins.max()],aspect='auto')
        axes[2].set_title('Infected per year', fontweight ="bold")
        axes[2].set_xlabel(r'Hysteresis Strength $\dot{H}_{Hyst}$')
        #axes[1].set_ylabel(r'Society inertia $\Delta p_{\mathrm{cap,min}}$')
        axes[2].set_yticks([], [])
        fig.colorbar(im2, ax = axes[2],orientation='vertical')
        plt.tight_layout()
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()   
    
Idot = []
Hdot = []
if __name__ == "__main__":
    Model = SIRModels(T=40,p_cap=0.05, s=0)
    Model.delta_pcap = Model.p_cap/2
    x= plots.PlotTrajectoryAndFix(Model, Model.DifferentPIncrement, T_min= 0,T_max=10000)
    plots.PlotSRITime(Model, Model.DifferentPIncrement, T_max=10000)
    print(Model.CheckStabilityNumerically(Model.DifferentPIncrement,10000,y0=[0.99,0.01,0,0,0]))
    #Model = SIRModels(p_cap=0.06)
    #Model = SIRModels()
    #Model.delta_pcap = Model.p_cap/2
    #plots.PlotStabilityNumerically(Model,Model.MemoryIncrement,dT = 1, pBaseMax=0.3,dp=0.1)#,save = 'Plots/StabilityAnalytical_MemoryIncrement')
    #plots.PlotTrajectoryAndFix(Model,Model.MemoryIncrement)
    #print(Model.CheckStabilityAnalytically(Model.MemoryIncrement))
    #plots.PlotSRITime(Model,Model.DifferentPIncrement)
    #print(Model.CheckStabilityNumerically(Model.DifferentPIncrement,T_min=1000,T_max=10000,eps=0.001))
    
    #plots.PlotStabilityNumerically(Model,Model.DifferentPIncrement,T_min=1000,T_max=10000, save = 'Plots/NumericalStabilityDifferentPIncrement',dT=5,dp=0.1)

    """
    #Model = SIRModels(s=0)
    #plots.PlotBifurcation(Model,Model.AdaptivePIncrement,save='Plots/BifurcatioAdaptiveP_s0_1_MxStep01')#,y0=[0.99,0.01,0,0,0,0,0])

    unser_Model = SIRModels(s=0, p_cap=0.07, dI_thresh=0.05, reductionfactor=1/10)#, dI_thresh=0.0000001, reductionfactor=0.00001, p_cap=0.01)
    Joel_Model = SIRModels(s=0)

    
    Model =unser_Model
    t,state = plots.PlotPH(Model,Model.DifferentPIncrement,T_min = 530, T_max = 600, y0=[0.99,0.01,0,0,0],save = 'Plots/JoelsTrajectory')# T_min = 1920, T_max = 2000, y0=[0.99,0.01,0,0,0])#y0=[0.99,0.01,0,0,0,0,0])
    #plt.plot(Idot, label = 'dI')
    #plt.plot(Hdot, label = 'dH')
    #plt.legend()
    #plt.show()
    plots.PlotSRITime(Model,Model.DifferentPIncrement, y0=[0.99,0.01,0,0,0], T_min = 500, T_max = 700)#T_min = 1500, T_max = 2000)
    plots.PlotTrajectoryAndFix(Model, Model.DifferentPIncrement)
    #plots.PlotTrajectoryAndFix(Model)
    
    
    """
    """
    Model = SIRModels()
    plots.PlotStability(Model, save='Plots/StabilityTry')
    
    """
    
        
    """
    Model=SIRModels()
    plots.PlotBifurcation(model = Model, func = Model.SeasonalyIncrement, save = 'Plots/Bifurcation')
    
    """
    