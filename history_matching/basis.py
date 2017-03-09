import patsy # TODO: Cleanup
from patsy import ModelDesc, Term, LookupFactor, EvalFactor, dmatrices
import itertools

class Basis():
    def __init__(self, model_terms, param_dict):
        self.model_terms = model_terms
        self.param_dict = param_dict


    @staticmethod
    def make_param_dict(param_names):
        # Return mapping from original parameter name to patsy-safe name
        return {p:p.replace(':','').replace('&',' ').replace(' ', '_') for p in param_names}


    @classmethod
    def identity_basis(cls, params):
        param_dict = Basis.make_param_dict(params)
        params_patsy = param_dict.values()
        model_terms += [Term([LookupFactor(x)]) for x in params_patsy] # X
        return cls(model_terms, param_dict)


    @classmethod
    def polynomial_basis(cls,
            params,
            intercept = True,
            first_order = True,
            second_order = False,
            third_order = False,
            fourth_order = False,
            fifth_order = False,
            higher_order = False
    ):
        param_dict = Basis.make_param_dict(params)
        params_patsy = param_dict.values()

        # Intercept
        if intercept:
            model_terms = [Term([])]
        else:
            model_terms = []

        # First order
        if first_order:
            model_terms += [Term([LookupFactor(x)]) for x in params_patsy] # X

        # Second order
        if second_order:
            model_terms += [Term([EvalFactor('%s**2'%x)]) for x in params_patsy] # X^2
            model_terms += [Term([EvalFactor('%s*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y

        # Third order
        if third_order:
            model_terms += [Term([EvalFactor('%s**3'%x)]) for x in params_patsy] # X^3

            model_terms += [Term([EvalFactor('%s*%s**2'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^2
            model_terms += [Term([EvalFactor('%s**2*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^2*Y

            model_terms += [Term([EvalFactor('%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y*Z

        # Fourth order
        if fourth_order:
            model_terms += [Term([EvalFactor('%s**4'%x)]) for x in params_patsy] # X^4
            model_terms += [Term([EvalFactor('%s**3*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^3*Y
            model_terms += [Term([EvalFactor('%s*%s**3'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^3

            model_terms += [Term([EvalFactor('%s**2*%s**2'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^2*Y^2

            model_terms += [Term([EvalFactor('%s**2*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X^2*Y*Z
            model_terms += [Term([EvalFactor('%s*%s**2*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y^2*Z
            model_terms += [Term([EvalFactor('%s*%s*%s**2'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y*Z^2

            model_terms += [Term([EvalFactor('%s*%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y*Z

        # Fifth order
        if fifth_order:
            model_terms += [Term([EvalFactor('%s**5'%x)]) for x in params_patsy] # X^5
            model_terms += [Term([EvalFactor('%s**4*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^4*Y
            model_terms += [Term([EvalFactor('%s*%s**4'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^4

            model_terms += [Term([EvalFactor('%s**3*%s**2'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^3*Y^2
            model_terms += [Term([EvalFactor('%s**2*%s**3'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^2*Y^3

            model_terms += [Term([EvalFactor('%s**3*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X^3*Y*Z
            model_terms += [Term([EvalFactor('%s*%s**3*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y^3*Z
            model_terms += [Term([EvalFactor('%s*%s*%s**3'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y*Z^3

            model_terms += [Term([EvalFactor('%s**2*%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W^2*X*Y*Z
            model_terms += [Term([EvalFactor('%s*%s**2*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X^2*Y*Z
            model_terms += [Term([EvalFactor('%s*%s*%s**2*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y^2*Z
            model_terms += [Term([EvalFactor('%s*%s*%s*%s**2'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y*Z^2

            model_terms += [Term([EvalFactor('%s**2*%s**2*%s'%x)]) for x in itertools.combinations(params_patsy, 3)] # X^2*Y^2*Z
            model_terms += [Term([EvalFactor('%s**2*%s*%s**2'%x)]) for x in itertools.combinations(params_patsy, 3)] # X^2*Y*Z^2
            model_terms += [Term([EvalFactor('%s*%s**2*%s**2'%x)]) for x in itertools.combinations(params_patsy, 3)] # X*Y^2*Z^2

            model_terms += [Term([EvalFactor('%s*%s*%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 5)] # V*W*X*Y*Z

        if higher_order:
            # Some sixth order
            model_terms += [Term([EvalFactor('%s**6'%x)]) for x in params_patsy] # X^6

            model_terms += [Term([EvalFactor('%s**5*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^5*Y
            model_terms += [Term([EvalFactor('%s*%s**5'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^5

            model_terms += [Term([EvalFactor('%s**3*%s*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W^3*X*Y*Z
            model_terms += [Term([EvalFactor('%s*%s**3*%s*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X^3*Y*Z
            model_terms += [Term([EvalFactor('%s*%s*%s**3*%s'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y^3*Z
            model_terms += [Term([EvalFactor('%s*%s*%s*%s**3'%x)]) for x in itertools.combinations(params_patsy, 4)] # W*X*Y*Z^3

            # Some seventh?! order
            model_terms += [Term([EvalFactor('%s**7'%x)]) for x in params_patsy] # X^7

            model_terms += [Term([EvalFactor('%s**6*%s'%x)]) for x in itertools.combinations(params_patsy, 2)] # X^6*Y
            model_terms += [Term([EvalFactor('%s*%s**6'%x)]) for x in itertools.combinations(params_patsy, 2)] # X*Y^6

        return cls(model_terms, param_dict)


    def apply(self, data):
        # Return data matrix as numpy array
        data = data.copy().rename(columns=self.param_dict)
        md = ModelDesc([], self.model_terms)
        dmat = patsy.dmatrix(md, data = data, return_type = 'dataframe', NA_action="raise")
        return dmat

    def get_terms(self):
        md = ModelDesc([], self.model_terms)
        terms = [c.strip() for c in md.describe().split('+')]
        terms[0] = terms[0][2:] # Remove '~ ' from beginning of first column

        if '0' in terms: # NO INTERCEPT
            terms.remove('0')
        else:
            terms = ['Intercept'] + terms

        return terms
