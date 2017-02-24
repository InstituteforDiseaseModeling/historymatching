class HistoryMatching():

    def __init__(self,
        params,         # Parameter definitions
        iteration = 0,  # Current iteration ?
        iterdir = 'Iterations'
    ):

        print('Welcome to IDM History Matching!')

    @classmethod
    def from_file():
        pass

    def save(self):
        pass

    def get_initial_samples(self, Nsamples):
        pass

    def step(self, samples, results,
            training_fraction = 0.75,
            force_optimize_glm = False,
            force_optimize_gpr = False,
            implausibility_threshold = 3,
            discrepancy_var = 30**2
        ):

        pass

    def glm(self,
            glm_fit_maxiter = 100000
            second_order_basis_terms = True
            third_order_basis_terms = False
            fourth_order_basis_terms = False
            fifth_order_basis_terms = False
            higher_order_basis_terms = False
        ):

        pass

    def gpr(self,
        method = 'CrossValidation'
    ):
        pass

    def joint(self):
        pass

    def implausibility(self):
        pass
