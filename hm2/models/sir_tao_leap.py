import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .abstract_model import ModelBase

class SIRTaoLeap(ModelBase):
    """
    An event-driven simulation of an SIR model based on Gillespie's tao-leap method.
    A description of the algorithm can be found in:

        Keeling, M. and Rohani, P. Modeling Infectious Diseases in Animals and
        Humans. p.204. Princeton University Press, 2008.
    """

    def __init__( self,
                  beta=1.2,  gamma=0.3,        # model parameters
                  s0=990,  i0=10,  r0=0,       # initial conditions
                  nDays=20,                    # number of days to simulate
                  stepSize=0.1,                # step size in days (<=1)
                  seed=None,                   # random number seed
                 ):
        """Create new sirTaoLeap object

        Arguments:
            beta:     Transmission rate (>0)
            gamma:    Inverse of the average transmission period (>0)
            s0:       Initial number of succeptible individuals (non-negative
                      integer)
            i0:       Initial number of infected individuals (non-negative int)
            r0:       Initial number of recovered individuals (non-negative int)
            nDays:    Number of days to simulate (positive integer)
            stepSize: Step size in days (value between 0 and 1)
            seed:     Seed for random number generator
        """

        # Verify inputs
        assert beta > 0
        assert gamma > 0
        assert s0 >= 0
        assert i0 >= 0
        assert r0 >= 0
        assert nDays > 0
        assert stepSize > 0
        assert stepSize <= 1

        # Initialize variables
        self.beta = beta
        self.gamma = gamma
        self.s0 = int( s0 )
        self.i0 = int( i0 )
        self.r0 = int( r0 )
        self.nDays = int( nDays )
        self.stepSize = stepSize
        self.seed = seed

    def print_parameters(self):
        """Display input arguments"""
        print(f"beta     = {self.beta}")
        print(f"gamma    = {self.gamma}")
        print(f"s0       = {self.s0}")
        print(f"i0       = {self.i0}")
        print(f"r0       = {self.r0}")
        print(f"nDays    = {self.nDays}")
        print(f"stepSize = {self.stepSize}")
        print(f"seed     = {self.seed}")

    def run(self):
        """Return S, I, and R arrays obtained with input arguments"""

        # Initialization
        beta        = self.beta
        gamma       = self.gamma
        stepSize    = self.stepSize
        nSteps      = math.ceil( (self.nDays-1)/stepSize )
        stepsPerDay = int(1/self.stepSize)
        nDays       = int( self.nDays )
        N           = self.s0 + self.i0 + self.r0    # population size

        s = np.zeros( nSteps+1, dtype = int )
        i = np.zeros( nSteps+1, dtype = int )
        r = np.zeros( nSteps+1, dtype = int )
        s[0] = self.s0
        i[0] = self.i0
        r[0] = self.r0

        s_daily = np.zeros( self.nDays, dtype = np.int )
        i_daily = np.zeros( self.nDays, dtype = np.int )
        r_daily = np.zeros( self.nDays, dtype = np.int )
        s_daily[0] = s[0]
        i_daily[0] = i[0]
        r_daily[0] = r[0]

        if self.seed:
            np.random.seed(seed=self.seed)


        # Compute events per time step
        for j in range(1, nSteps+1):
            lambdaTransmission = ( beta * s[j-1] * i[j-1] / N ) * stepSize
            lambdaRecovery = gamma * i[j-1] * stepSize

            deltaMt = np.random.poisson( lambdaTransmission )
            deltaMr = np.random.poisson( lambdaRecovery )

            s[j] = max(  s[j-1] - deltaMt,            0 )
            i[j] = max(  i[j-1] + deltaMt - deltaMr,  0 )
            r[j] = r[j-1] + deltaMr


        # Summarize events per day
        for j in range(1, nDays):
            k = j*stepsPerDay

            s_daily[j] = s[k]
            i_daily[j] = i[k]
            r_daily[j] = r[k]

        # Finalize and return
        return pd.DataFrame({
            'time':        list(range(0, nDays)),
            'susceptible': s_daily,
            'infected':    i_daily,
            'recovered':   r_daily,
            'incidence':   [s_daily[j-1] - s_daily[j] for j in range(1,nDays)] + [np.nan]
        })

    def plot( self ): # pragma: no cover
        """Display figure with S, I, and R curves obtained with the input args"""

        results = self.simulate()
        print(results)
        x = range(0, self.nDays)
        title = "beta = " + str(self.beta)           \
                 + ";  gamma = " + str(self.gamma)
        plt.figure()
        plt.plot( x, results["susceptible"], 'g')
        plt.plot( x, results["infected"], 'r')
        plt.plot( x, results["recovered"], 'b')
        plt.legend(['S', 'I', 'R'])
        plt.ylabel('cases')
        plt.xlabel('days')
        plt.title( title )
        plt.grid(linestyle=':')
        plt.show(block=False)

    def plotIncidence( self ): # pragma: no cover
        """Display figure with the incidence obtained with the input args"""

        incidence = self.simulate()["incidence"]
        x = range(0, self.nDays)
        title = "Incidence (beta = " + str(self.beta)           \
                 + ";  gama = " + str(self.gamma) + ")"
        plt.figure()
        plt.plot( x, incidence, 'b' )
        plt.ylabel('new cases')
        plt.xlabel('days')
        plt.title(title)
        plt.grid(linestyle=':')
        plt.show(block=False)
