import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as int
import general_methods as gm
from scipy.signal import find_peaks
# plt.rc('font', size=18) #controls default text size


class SIRModels:
    def __init__(self, gamma=0.1, beta=0.5, mue=0, nue=1/100, p_base = 0.1, s=0.3, T=25, dI_thresh = 0.001, reductionfactor = 1/10, p_cap = 1e-3):
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
                P = self.P_new(state[4], None)
                dP = self.dP_new(state[4])


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
    def FindFixpoint(self,fun,I0 = 0.001):
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
        print(Fix)
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
    def P_new(self,H,dI):

        if not self.delta_pcap:
            if dI<0:
                delta_p_cap = self.p_cap/2
            elif dI < self.dI_thresh:
                delta_p_cap = -(self.p_cap/2-self.d_pcap_min)/self.dI_thresh*dI+self.p_cap/2
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
            return (1-self.p_base)/(1+np.exp(epsilon*((1/(pcap2-H)+1/(pcap1-H)))))+self.p_base
        
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
    def CheckStabilityNumerically(self,Func,T_min,T_max,eps):
        """
        Returns  True if Convergences is achieved
        """
        
        if Func.__name__ == 'AdaptivePIncrement':
            event1 = lambda t,x: self.event_threshold_p_adapt(t,x)    
        if Func.__name__ == 'DifferentPIncrement':
            event1 = lambda t,x: self.different_p_event(t,x)     
        else:
            event1 = lambda t,x: self.i_peaks(t,x)


        SolveDict = int.solve_ivp(Func,[0,T_max],[0.99,0.01,0,0,0], max_step = 1,events=[event1])
        events = SolveDict.y_events[0]
        t_events = SolveDict.t_events[0]
        Ivents=events[t_events>T_min,1]
        MinMaxDist = np.abs(Ivents[1:]-Ivents[:-1])
        dMinMaxDist = (np.roll(MinMaxDist, -1, 0)-np.roll(MinMaxDist, +1, 0))/(2)
        if np.shape(MinMaxDist)[0]<10:
            raise ValueError(
                "Integration Length is to small!"
            )
        MeanInc = np.mean(dMinMaxDist[-100:-2])
        Mean = np.mean(MinMaxDist[-10:])
        if Mean>eps or MeanInc>0:
            print(Mean,MeanInc)
            return False
        else:
            print(self.T,self.p_base)
            return True

    def CheckStabilityAnalytically(self,func):
        if func.__name__ == 'MemoryIncrementForStability':
            self.Fix = self.FindFixpoint(self.MemoryIncrementForStability)
        elif func.__name__ == 'DifferentPIncrement':
            self.Fix = self.FindFixpointDifferentP(self.DifferentPIncrementForStability)
        Jacobi = self.Jacobi(self.Fix, func)
        w = np.linalg.eigvals(Jacobi)
        w_real=np.real(w)
        max = np.max(w_real)
        if max>0:
            return 0
        else:
            return 1

class plots():
    def PlotTrajectoryAndFix(Model,Func,save=None, T_min=0, T_max=10000):
        SolveDict = int.solve_ivp(Func,[0,T_max],[0.99,0.01,0,0,0], max_step = 1)
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

    def PlotSRITime(model,func, y0=[0.99,0.01,0,0,0],save=None, T_min = 0, T_max=10000):
        
        if func.__name__ == 'AdaptivePIncrement':
            event1 = lambda t,x: model.event_threshold_p_adapt(t,x)    
        if func.__name__ == 'DifferentPIncrement':
            event1 = lambda t,x: model.different_p_event(t,x)     
        else:
            event1 = lambda t,x: model.i_peaks(t,x)
        event1.direction = -1 
        """
        
        event2 = lambda t,x: model.seasonal_peaks(t,x)
        event2.direction = -1
        
        event3_thres = lambda t,x: model.event_threshold(t,x)
        event3_thres.direction = 1
        
        event4_thres = lambda t,x: model.event_threshold(t,x)
        event4_thres.direction = -1
        """
        
        
        SolveDict = int.solve_ivp(func,[0,T_max], y0, max_step = 0.1, events=[event1])#,event2,event3_thres,event4_thres])
        t = SolveDict.t
        state = SolveDict.y
        events0 = SolveDict.y_events[0]
        t_events = SolveDict.t_events[0]
        #events2_ind = find_peaks(state[1,t_events>T_min])[0]
        #print(events.shape,t_events.shape)
        #plt.plot(t,state[0,:], label = 'S')
        plt.plot(t[t>T_min],state[1,t>T_min], label = 'I')
        #plt.scatter(t_events[t_events>T_min],events0[t_events>T_min,1],label='Maxima',c='red') #t_events>T_min
        #plt.scatter(t[events2_ind],state[1,events2_ind],label='Maxima2nd',c='green')
        #plt.plot(t,state[2,:], label = 'R')
        #plt.ylim(0.02025, 0.0204)
        plt.xlabel(r"Time $t$ in days")
        plt.ylabel("Infected fraction $I$ ")
        plt.legend()

        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()
        return t,state

    def PlotStabilityAnalytically(Model,func,save = None, dT=0.1, T_max=60, dp=0.0001, P_max=0.25):
        Ts=np.arange(1,T_max,dT)
        p_bases = np.arange(0.01,P_max,dp)
        Stable = np.ones((np.shape(Ts)[0],np.shape(p_bases)[0]))
        for j,Model.T in enumerate(Ts):
            for i, Model.p_base in enumerate(p_bases):
                Stable[j, i] = Model.CheckStabilityAnalytically(func)
                gm.progress_featback.printProgressBar(j*np.shape(p_bases)[0]+i,np.shape(p_bases)[0]*np.shape(Ts)[0])

        im = plt.imshow(Stable[-1:0:-1,-1:0:-1],vmin=0,vmax=1,extent=[p_bases.min(),p_bases.max(),Ts.min(),Ts.max()],aspect='auto')
        
        plt.title('Stability Diagramm', fontweight ="bold")
        plt.xlabel(r'Stubberness $ p_{base}$')
        plt.ylabel(r'Mean memory $T$')
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()  

    def PlotStabilityNumerically(model, func, T_min, T_max, save = None, dT=0.5,dp=0.01,eps = 0.001):
        Ts=np.arange(1,80,dT)
        p_bases = np.arange(0.01,0.5,dp)
        StabilityResults = np.zeros((Ts.shape[0],p_bases.shape[0]))

        for i,model.p_base in enumerate(p_bases):
            for j,model.T in enumerate(Ts):
                StabilityResults[j,i]=model.CheckStabilityNumerically(func,T_min,T_max,eps)                
                gm.progress_featback.printProgressBar(i*np.shape(Ts)[0]+j,np.shape(p_bases)[0]*np.shape(Ts)[0])
        im = plt.imshow(StabilityResults[-1:0:-1,-1:0:-1],vmin=0,vmax=1,extent=[p_bases.min(),p_bases.max(),Ts.min(),Ts.max()],aspect='auto')
        plt.title('Stability Diagramm', fontweight ="bold")
        plt.xlabel(r'Stubberness $ p_{base}$')
        plt.ylabel(r'Mean memory $T$')
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()
    

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
        # plt.figure(figsize=(10,8))
        # plt.rc('font', size=18) 
        t = np.arange(T_min,T_max, 0.5)        
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
            H1 = state[3,:]
            P_H = []
            for h,h1 in zip(H,H1):
                P_H.append(model.P_new(h,h1))
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

    
Idot = []
Hdot = []
