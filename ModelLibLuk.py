import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.integrate as int


class SIRModels:
    def __init__(self, gamma=0.1, beta=0.5, mue=0, nue=1/100, p_base = 0.1):
        self.gamma = gamma
        self.beta = beta
        self.mue = mue
        self.nue = nue
        self.p_base = p_base # Strongest possible factor that reduces the transmission rate
        self.p_cap = 1e-3 # percieved risk beyond which no further reduction of the reansmission rate takes palce
        self.epsilon = 1e-4 # Curvature parameter
        self.s = 0.3 # Amplitude of seasonal forcing
        self.omega = 2*np.pi/360 #Frequency of yearly seasonsl variation
        self.T = 60 # mittlerer Erinnerungszeit
        self.P = lambda H: self.p_base+(1-self.p_base)/self.p_cap*self.epsilon*np.log(1+np.exp(1/self.epsilon*(self.p_cap-H)))
        self.Gamma = lambda t : 1+self.s*np.cos(self.omega*t)
        self.dP = lambda H : -(((1-self.p_base)/self.p_cap)*np.exp((self.p_cap-H)/self.epsilon))/(1+np.exp((self.p_cap-H)/self.epsilon))
        self.Fix = self.FindFixpoint(self.MemoryIncrementForStability)
        self.direction= True
        self.threshold = 0.03


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

    def PlotSRITime(Model,save=None):
        SolveDict = int.solve_ivp(Model.MemoryIncrement,[0,10000],[0.99,0.01,0,0,0], max_step = 1)
        t = SolveDict.t
        state = SolveDict.y
        plt.plot(t,state[0,:], label = 'S')
        plt.plot(t,state[1,:], label = 'I')
        plt.plot(t,state[2,:], label = 'R')
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

        im = plt.imshow(Stable[-1:0:-1,:],vmin=0,vmax=1,extent=[p_bases.min(),p_bases.max(),Ts.min(),Ts.max()],aspect='auto')
        plt.colorbar(im)
        plt.title('Stability Diagramm', fontweight ="bold")
        if save is not None:
            plt.savefig(save+'.pdf')
        else:
            plt.show()       

    def PlotBifurcation(self,model, save=None):
        peaks_list = []
        precision = 200

        event1 = lambda t,x: model.i_peaks(t,x)  
        event1.direction = -1

        event3_thres = lambda t,x: model.event_threshold(t,x)
        event3_thres.direction = 1
        
        event4_thres = lambda t,x: model.event_threshold(t,x)
        event4_thres.direction = -1

        for s in np.linspace(0,0.3,precision):
            model.s = s
            S = np.random.random()
            solve_dict_seasonal = int.solve_ivp(model.SeasonalyIncrement,[500,3000], [S,1-S,0.0, 0, 0],max_step=10, events=[event1,event3_thres,event4_thres])
            I = solve_dict_seasonal.y[1,:]
            t = solve_dict_seasonal.t
            events = solve_dict_seasonal.y_events
            t_events = solve_dict_seasonal.t_events
            peaks_list.append(events)

        s_list = np.linspace(0,0.3,precision)
        plt.figure(figsize=(10,10))

        for peaks,s  in zip(peaks_list,s_list):
            one_s_list = [s for _ in range(peaks[0].shape[0])]
            plt.scatter(one_s_list,peaks[0][:,1],color='black',marker=',',lw=0, s=1)

        plt.ylabel('I')
        plt.xlabel('s')
        plt.show()

    
if __name__ == "__main__":

    Model = SIRModels()
    plotting = plots()

    plotting.PlotBifurcation(Model)

    """
    Model=SIRModels()
    Fix = Model.FindFixpoint(Model.MemoryIncrement)
    print(Fix)
    """
    
    """
    Model=SIRModels()
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

# Model=SIRModels()
# plots.PlotStability(Model=Model, save='/Plots/StabilityDiagrammTp_base')


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

   
    
    