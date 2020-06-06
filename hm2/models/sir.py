import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .abstract_model import ModelBase

class SIR(ModelBase):
    """A stochastic SIR model TODO"""

    def __init__(self, sir0=[190,10,0], Tmax=100, beta=0.05, gamma=0.1, seed=None):
        """Initialize the SIR model

        TODO: Where is this model from?

        Args:
            sir0 - Array specifying the initial values: [Susceptible, Infected, Recovered]
            TMax - TODO
            beta - TODO
            gama - TODO
            seed - Seed for the PRNG: used for reproducibility
        """

        assert len(sir0) == 3 # S, I, R
        assert beta > 0
        assert gamma > 0

        self.sir0  = np.array(sir0)
        self.Tmax  = Tmax
        self.beta  = beta
        self.gamma = gamma
        self.seed  = np.uint32(seed) if seed else None

        self.M = np.array( [[-1, 1, 0], [0, -1, 1]] ) # S-->I, I-->R

    def print_parameters(self):
        """Display input arguments"""
        print(f"sir0  = {self.sir0}")
        print(f"Tmax  = {self.Tmax}")
        print(f"beta  = {self.beta}")
        print(f"gamma = {self.gamma}")
        print(f"seed  = {self.seed}")

    def run(self):
        """Run the simulation, given the parameters specified in the constructor
        """

        t       = 0
        times   = [t]
        sir     = [self.sir0.copy()]
        S, I, R = sir[-1]

        if self.seed:
            np.random.seed(seed=self.seed)

        while t <= self.Tmax and I > 0:
            propensity = [self.beta * S * I, self.gamma * I]

            dt = np.random.exponential(scale = 1/sum(propensity))
            times.append( t+dt )
            t = times[-1]

            state_change = self.M[np.random.choice([0,1], 1, propensity)[0]]

            new_observation = np.clip(sir[-1]+state_change, a_min=0, a_max=None)
            sir.append(new_observation)
            S, I, R = new_observation

        P = [x/np.sum(x)*100 for x in sir]

        P   = np.vstack(P)
        sir = np.vstack(sir)

        return pd.DataFrame({
            'time':            times,
            'susceptible':     sir[:,0],
            'infected':        sir[:,1],
            'recovered':       sir[:,2],
            'per_susceptible': P  [:,0],
            'per_infected':    P  [:,1],
            'per_recovered':   P  [:,2],
        })
