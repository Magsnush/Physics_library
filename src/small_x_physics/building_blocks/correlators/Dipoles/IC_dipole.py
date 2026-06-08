# This file defines the initial condition for the dipole amplitude, which is used as input for the BK evolution.
# It has been tested against the BK dipole amplitude at Y = 0.0 and it works

import numpy as np

class ICDipole:
    def __init__(self, Qs0, gamma, ec,LambdaQCD):
        self.Qs0 = Qs0
        self.gamma = gamma
        self.ec = ec
        self.LambdaQCD = LambdaQCD
        
    def radius(self, x, y):
        return np.linalg.norm(np.asarray(x) - np.asarray(y), axis=-1)
    
    def MV_model_S2(self, x, y):
        r = self.radius(x, y) + 1e-12
        return np.exp(-(r**2 * self.Qs0**2)**self.gamma / 4 * np.log(1/(r * self.LambdaQCD) + self.ec * np.exp(1)))

 
    

    


