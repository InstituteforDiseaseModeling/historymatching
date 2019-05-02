import numpy as np
import matplotlib.pyplot as plt

class SIR:
    def __init__(self, x0=[190,10,0], Tmax=100, beta=0.05, gamma=0.1, seed=None):
        assert(len(x0) == 3) # S, I, R
        assert(beta > 0)
        assert(gamma > 0)

        self.x0 = np.array(x0)
        self.Tmax = Tmax
        self.beta = beta
        self.gamma = gamma
        self.seed = np.uint32(seed) if seed else None

        self.M = np.array( [[-1, 1, 0], [0, -1, 1]] ) # S-->I, I-->R


    def sim(self):
        t = 0
        T = [t]
        X = [self.x0]
        S, I, R = X[-1]

        if self.seed:
             np.random.seed(seed=self.seed)

        while t < self.Tmax and I > 0:
            propensity = [self.beta * S * I, self.gamma * I]
            total_propensity = sum(propensity)

            dt = np.random.exponential(scale = 1/total_propensity)
            T.append( t+dt )
            t = T[-1]

            r = np.random.uniform()
            cum_propensity = np.cumsum(propensity) / total_propensity

            reaction = next(i for (i,x) in enumerate(cum_propensity) if x >= r)
            state_change = self.M[reaction]

            X.append( X[-1]+state_change )
            S, I, R = X[-1]

        P = [x/np.sum(x)*100 for x in X]
        return T,X,P


    def plot(self, figsize=(16,10)):
        fig, ax = plt.subplots(figsize=figsize)
        T, X, P = self.sim()

        plt.plot(T, P)
        plt.legend(['S', 'I', 'R'])

        return fig, ax
