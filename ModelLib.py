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
        self.s = 0.5 # Amplitude of seasonal forcing
        self.omega = 2*np.pi/360 #Frequency of yearly seasonsl variation
        self.T = 100 # mittlerer Erinnerungszeit
        self.P = lambda state: self.p_base+(1-self.p_base)/self.p_cap*self.epsilon*np.log(1+np.exp(1/self.epsilon*(self.p_cap-state[4])))
        self.Gamma = lambda t : 1+self.s*np.cos(self.omega*t)

    
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
        return [dS,dI,dR]

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
        return [dS,dI,dR]


    def SeasonalyIncrement(self, t, state):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R,H1,H) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R,H1,H) vector
            - t    - Required: transmission rate

        Return: 
            - state after one step
        """
        dS = -self.beta*self.P(state)*self.Gamma(t)*state[1]*state[0]+ self.nue*state[2]
        dI = self.beta*self.P(state)*self.Gamma(t)+state[1]*state[0]- self.gamma+state[1]
        dR = self.gamma*state[1]-self.nue+state[2]
        dH1 = 2/self.T*(state[1]-state[3])
        dH = 2/self.T*(state[3]-state[4])

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
        dS = -self.beta*self.P(state)*1*state[1]*state[0]+ self.nue*state[2]
        dI = self.beta*self.P(state)*1+state[1]*state[0]- self.gamma+state[1]
        dR = self.gamma*state[1]-self.nue+state[2]
        dH1 = 2/self.T*(state[1]-state[3])
        dH = 2/self.T*(state[3]-state[4])

        return [dS,dI,dR,dH1,dH]
