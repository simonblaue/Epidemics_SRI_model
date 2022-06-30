import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

class SIRModels:
    def __init__(self) -> None:
        
        pass

    
    def ClassicIncrement (state, beta=0.1, gamma=1):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R) vector
            - beta    - Required: transmission rate
            - gamma   - Required: recovery rate
        Return: 
            - temporal derivative of the sate vector
        """
        dS = -beta*state[1]*state[0]
        dI = beta*state[1]*state[0] - gamma*state[1]
        dR = gamma*state[1]
        return (dS,dI,dR)

    def VitalIncrement(state,beta,gamma,mue,nue):
        """
        The function returns the temporal derivative of the state vector 'state'=(S,I,R) for the classical SIR Model.

        Args:
            - state   - Required: state=(S,I,R) vector
            - beta    - Required: transmission rate
            - gamma   - Required: recovery rate
        Return: 
            - temporal derivative of the sate vector
        """
        dS = -beta*state[1]*state[0] + mue - mue*state[0] + nue*state[2]
        dI = beta*state[1]*state[0] - gamma*state[1] - mue*state[1]
        dR = gamma*state[1] - mue*state[2] - nue*state[2]
        return (dS,dI,dR)

