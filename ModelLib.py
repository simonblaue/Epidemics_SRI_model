import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as int
import general_methods as gm
from scipy.signal import find_peaks


class SIRModels:
    def __init__(self, gamma=0.1, beta=0.5, mue=0, nue=1/100, p_base = 0.1, s=0.3):
        self.gamma = gamma
        self.beta = beta
        self.mue = mue
        self.nue = nue
        self.p_base = p_base # Strongest possible factor that reduces the transmission rate
        self.p_cap = 1e-3 # percieved risk beyond which no further reduction of the reansmission rate takes palce
        self.epsilon = 1e-4 # Curvature parameter
        self.s = s # Amplitude of seasonal forcing
        self.omega = 2*np.pi/360 #Frequency of yearly seasonsl variation
        self.T = 60 # mittlerer Erinnerungszeit
        self.P = lambda H: self.p_base+(1-self.p_base)/self.p_cap*self.epsilon*np.log(1+np.exp(1/self.epsilon*(self.p_cap-H)))
        self.Gamma = lambda t : 1+self.s*np.cos(self.omega*t)
        self.dP = lambda H : -(((1-self.p_base)/self.p_cap)*np.exp((self.p_cap-H)/self.epsilon))/(1+np.exp((self.p_cap-H)/self.epsilon))
        self.Fix = self.FindFixpoint(self.MemoryIncrementForStability)

        self.direction= True
        self.threshold = 0.03
        # new params
        self.Ti = 2
        self.dI_thresh = 0.001 
        self.d_pcap_min = self.p_cap/10


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


    def P_new(self,H,dI):
  
        if dI<0:
            delta_p_cap = self.p_cap/2
        elif dI < self.dI_thresh:
            delta_p_cap = -(self.p_cap/2-self.d_pcap_min)/self.dI_thresh*dI+self.p_cap/2
        else:
            delta_p_cap = self.d_pcap_min

        pcap1 = self.p_cap/2-delta_p_cap
        pcap2 = self.p_cap/2+delta_p_cap
        epsilon = pcap2

        if H>=pcap2:
            return self.p_base
        if H<=pcap1: 
            return 1
        if H>pcap1:
            return (1-self.p_base)/(1+np.exp(epsilon*(1/(pcap2-H)+1/(pcap1-H))))+self.p_base

    def AdaptivePIncrement(self, t, state):

        S , I, R, H1, H, Hi1, Hi = state
        Gamma = self.Gamma(t)
        # TODO:  neeed the last dI in here :(
        P = self.P_new(H,Hi)

        dS = -self.beta*P*Gamma*I*S + self.nue*R
        dI = self.beta*P*Gamma*I*S - self.gamma*I
        dR = self.gamma*I - self.nue*R
        dH1 = 2/self.T*(I-H1)
        dH = 2/self.T*(H1-H)
        dHi1 = 2/self.Ti*(dI-Hi)
        dHi = 2/self.Ti*(Hi1-Hi)

        return [dS,dI,dR,dH1,dH,dHi1,dHi]

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



    def JacobiMemory(self,state):
        """
        The function returns the Jacobian.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
        Return: 
            - JacobiMatrix
        """
        Jac = np.zeros((4,4))
        Jac[0,0] = - self.beta*self.P(state[4])*state[1]-self.nue
        Jac[0,1] = - self.beta*self.P(state[4])*state[0]-self.nue

        Jac[0,3] = - self.beta*self.dP(state[4])*state[0]*state[1]
        Jac[1,0] = self.beta*self.P(state[4])*state[1]
        Jac[1,1] = self.beta*self.P(state[4])*state[0]-self.gamma

        Jac[1,3] = self.beta*self.dP(state[4])*state[0]*state[1]

        Jac[2,1] = 2/self.T
        Jac[2,2] = - 2/self.T



        Jac[3,2] = 2/self.T
        Jac[3,3] = - 2/self.T
        return Jac

    def FindFixpoint(self,fun,I0 = 1):
        Fix = opt.root(fun,I0)
        I = Fix.x
        S = self.gamma/(self.beta*self.P(I))
        R = self.gamma/self.nue*I
        H = I
        H1 = I
        Fix = np.array([S,I,R,H1,H])
        return Fix
    
    def FOI_softplus(self,M,I):
        return self.softplus(M)*I*self.beta
    
    def softplus(self,M):
        slope = (1-self.p_base)/self.p_cap
        return slope*self.epsilon*np.log(np.exp(1/self.epsilon*(self.p_cap-M))+1)+self.p_base

    def seasonalforcing(self,t):
        return (1+self.s*np.sin(self.omega*t))

    def i_peaks(self,t,y):
        FOI = self.FOI_softplus(y[4],y[1])
        return FOI*y[0]*self.seasonalforcing(t) - self.gamma*y[1]

    def event_threshold(self,t,y):
        return y[1]-self.threshold


    ####

    def FOI_our_p(self,H,Hi):
        return self.P_new(H,Hi)

    def event_threshold_p_adapt(self,t,y):
        FOI = self.FOI_our_p(y[4],y[6],)
        return FOI*y[0]*self.seasonalforcing(t) - self.gamma*y[1]


class plots():
    def PlotTrajectoryAndFix(Model,save=None):
        print(Model.Fix)
        SolveDict = int.solve_ivp(Model.MemoryIncrement,[0,10000],[0.99,0.01,0,0,0], max_step = 1)
        #int(np.shape(X) , X)
        t = SolveDict.t
        state = SolveDict.y
        #f = Model.P(state)
        #plt.plot(t,f)
        #plt.plot(t,state[0,:], label = 'S')
        #plt.plot(t,state[1,:], label = 'I')
        #plt.plot(t,state[2,:], label = 'R')
        plt.plot(state[0,:],state[1,:],label = 'Trajectory')
        plt.scatter(Model.Fix[0],Model.Fix[1], label = 'Fixpoint',color = 'r')
        #plt.hlines(y = 0 , xmin=np.min(state[0,:]),xmax=np.max(state[0,:]), color = 'black', linestyles='dotted')
        plt.xlabel('S')
        plt.ylabel('I')
        plt.legend()
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()

    def PlotSRITime(func, y0=[0.99,0.01,0,0,0],save=None):
        SolveDict = int.solve_ivp(func,[10000,100000], y0, max_step = 1)
        t = SolveDict.t
        state = SolveDict.y
        #plt.plot(t,state[0,:], label = 'S')
        plt.figure(figsize=(30,23))
        plt.plot(t[1000:],state[1,:][1000:], label = 'I')
        #plt.plot(t,state[2,:], label = 'R')
        plt.ylim(0.02025, 0.0204)
        #plt.ylim(0.0383,0.038335) for adjusted P
        plt.legend()
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()

    def PlotStability(Model,save = None):
        
        Ts=np.arange(1,60,0.1)
        p_bases = np.arange(0.01,0.25,0.0001)
        Stable = np.ones((np.shape(Ts)[0],np.shape(p_bases)[0]))
        for j,T in enumerate(Ts):
            for i, p_base in enumerate(p_bases):
                Model.T = T
                Model.p_base = p_base
                Model.Fix = Model.FindFixpoint(Model.MemoryIncrementForStability)
                Jacobi = Model.JacobiMemory(Model.Fix)
                w = np.linalg.eigvals(Jacobi)
                w_real=np.real(w)
                max = np.max(w_real)
                if max>0:
                    Stable[j, i] = 0
                gm.progress_featback.printProgressBar(j*np.shape(p_bases)[0]+i,np.shape(p_bases)[0]*np.shape(Ts)[0])

        im = plt.imshow(Stable[-1:0:-1,:],vmin=0,vmax=1,extent=[p_bases.min(),p_bases.max(),Ts.min(),Ts.max()],aspect='auto')
        plt.colorbar(im)
        plt.title('Stability Diagramm', fontweight ="bold")
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()   

    def PlotBifurcation(model, func, save=None):
        peaks_list = []
        time_list = []
        precision = 250

        event1 = lambda t,x: model.i_peaks(t,x)  
        event1.direction = -1

        event3_thres = lambda t,x: model.event_threshold(t,x)
        event3_thres.direction = 1
        
        event4_thres = lambda t,x: model.event_threshold(t,x)
        event4_thres.direction = -1

        p_adapt_events = lambda t,x: model.event_threshold_p_adapt(t,x)

        plt.figure(figsize=(10,10))
        s_array = np.linspace(0,0.3,precision)
        for (i,s) in enumerate(s_array):
            model.s = s
            S = np.random.random() 
            if func.__name__ == 'AdaptivePIncrement':
                solve_dict_seasonal = int.solve_ivp(func,[0,30000], [S,1-S,0.0, 0, 0,0,0],max_step=10)#,event3_thres,event4_thres])
                I = solve_dict_seasonal.y[1,:]
                events_idx = find_peaks(I,height=[0.001,0.2])[0]
                events = I[events_idx]
                one_s_list = [s for _ in range(len(events))]
                plt.scatter(one_s_list,events,marker=',',lw=0, s=1)

            else:
                solve_dict_seasonal = int.solve_ivp(func,[0,10000], [S,1-S,0.0, 0, 0],max_step=10, events=[event1])#,event3_thres,event4_thres])
                t = solve_dict_seasonal.t
                events = solve_dict_seasonal.y_events[0]
                t_events = solve_dict_seasonal.t_events[0]
                one_s_list = [s for _ in range(events.shape[0])]
                plt.scatter(one_s_list,events[:,1],marker=',',lw=0, s=1,c = t_events)
            gm.progress_featback.printProgressBar(i,precision)

        plt.ylabel('I')
        plt.xlabel('s')
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()   
  
"""
            peaks_list.append(events)
            time_list.append(t_events)

        s_list = np.linspace(0,0.3,precision)
        plt.figure(figsize=(10,10))

        for peaks,s  in zip(peaks_list,s_list):
            one_s_list = [s for _ in range(peaks.shape[0])]
            plt.scatter(one_s_list,peaks[:,1],marker=',',lw=0, s=1,c = time_list)


"""

    
if __name__ == "__main__":
    # Model=SIRModels()
    # plots.PlotBifurcation(model = Model, save = 'Plots/Bifurcation')



    Model = SIRModels(s=0.5)
    plots.PlotSRITime(Model.AdaptivePIncrement, y0=[0.99,0.01,0,0,0,0,0])

    
"""
    Model=SIRModels()
    Fix = Model.FindFixpoint(Model.MemoryIncrement)
    print(Fix)
    """
    
"""
    Model=SIRModels(s=0.5)
    SolveDict = int.solve_ivp(Model.MemoryIncrement,[0,10000],[0.99,0.01,0,0,0], max_step = 1)
    #int(np.shape(X) , X)
    t = SolveDict.t
    state = SolveDict.y
    #f = Model.P(state)
    #plt.plot(t,f)
    #plt.plot(t,state[0,:], label = 'S')
    #plt.plot(t,state[1,:], label = 'I')
    #plt.plot(t,state[2,:], label = 'R')
    plt.plot(state[0,:],state[1,:],label = 'Trajectory')
    plt.scatter(Model.Fix[0],Model.Fix[1], label = 'Fixpoint')
    plt.xlabel('S')
    plt.ylabel('I')
    plt.legend()
    plt.show()


"""



"""
Model=SIRModels()
plots.PlotStability(Model=Model, save='Plots/StabilityDiagrammTp_base')

"""






"""
    for p_base in np.arange(0.1,0.3,0.05):
        Model=SIRModels(model = 'Memory', p_base = p_base)
        plots.PlotTrajectoryAndFix(Model)
 
 """   
"""
Model=SIRModels()
Jacobi = Model.JacobiMemory(Model.Fix)
print(Jacobi)

"""

   
    
    