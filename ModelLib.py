import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class SIRModels:
    def __init__(self, gamma=0.1, beta=0.5, mue=0, nue=1/100):
        self.gamma = gamma
        self.beta = beta
        self.mue = mue
        self.nue = nue
        self.p_base = 0.1 # Strongest possible factor that reduces the transmission rate
        self.p_cap = 1e-3 # percieved risk beyond which no further reduction of the reansmission rate takes palce
        self.epsilon = 1e-4 # Curvature parameter
        self.s = 0.3 # Amplitude of seasonal forcing
        self.omega = 2*np.pi/360 #Frequency of yearly seasonsl variation
        self.T = 60 # mittlerer Erinnerungszeit
        self.P = lambda state: self.p_base+(1-self.p_base)/self.p_cap*self.epsilon*np.log(1+np.exp(1/self.epsilon*(self.p_cap-state[4])))
        self.Gamma = lambda t : 1+ self.s*np.cos(self.omega*t)
        self.dP = lambda state : (((1-self.p_base)/self.p_cap)*self.epsilon*np.exp((self.p_cap-state[4])/self.epsilon))/(1+np.exp(self.p_cap-state[4]/self.epsilon))


    def ClassicIncrement(self,t, state):
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
        return [dS,dI,dR, 0 ,0]

    def VitalIncrement(self,t, state):
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
        return [dS,dI,dR, 0 ,0]


    def SeasonalyIncrement(self, t, state):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R,H1,H) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
            - t    - Required: transmission rate

        Return: 
            - state after one step
        """
        P = self.P(state)
        Gamma = self.Gamma(t)
        S , I, R, H1, H = state

        dS = -self.beta*P*Gamma*I*S + self.nue*R
        dI = self.beta*P*Gamma*I*S - self.gamma*I
        dR = self.gamma*I - self.nue*R
        dH1 = 2/self.T*(I-H1)
        dH = 2/self.T*(H1-H)

        return [dS,dI,dR,dH1,dH]

    def MemoryIncrement(self, t, state):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R,H1,H) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
            - t    - Required: transmission rate

        Return: 
            - state after one step
        """
        P = self.P(state)
        S , I, R, H1, H = state

        dS = -self.beta*P*1*I*S + self.nue*R
        dI = self.beta*P*1*I*S - self.gamma*I
        dR = self.gamma*I - self.nue*R
        dH1 = 2/self.T*(I-H1)
        dH = 2/self.T*(H1-H)

        return [dS,dI,dR,dH1,dH]

    def JacobiMemory(self,state):
        """
        The function returns the Jacobian.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
        Return: 
            - JacobiMatrix
        """
        Jac = np.zeros((4,4))
        Jac[0,0] = - self.beta*self.P(state)*state[1]-self.nue
        Jac[0,1] = - self.beta*self.P(state)*state[0]-self.nue

        Jac[0,3] = - self.beta*self.dP(state)*state[0]*state[1]
        Jac[1,0] = self.beta*self.P(state)*state[1]
        Jac[1,1] = self.beta*self.P(state)*state[0]-self.gamma

        Jac[1,3] = self.beta*self.dP(state)*state[0]*state[1]

        Jac[2,1] = 2/self.T
        Jac[2,2] = - 2/self.T


        Jac[3,2] = 2/self.T
        Jac[3,3] = - 2/self.T
        return Jac
