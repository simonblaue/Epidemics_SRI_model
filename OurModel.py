
from ModelLib import SIRModels
import numpy as np

class OurModel(SIRModels):

    def __init__(self,gamma=0.1, beta=0.5, mue=0, nue=1/100, p_base = 0.1, s=0.3):
        super().__init__(gamma=0.1, beta=0.5, mue=0, nue=1/100, p_base = 0.1, s=0.3)

        self.Gamma = super().Gamma



