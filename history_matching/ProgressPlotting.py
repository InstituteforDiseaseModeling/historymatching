import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from history_matching import HistoryMatching
from history_matching.glm import GLM
from history_matching.gpr import GPR


class ProgressPlotting():

    def __init__(self, cut_dir, samples, iteration):
        self.cut_dir = cut_dir
        self.samples = samples
        self.iteration = iteration

        self.param_info = None
        self.Xcols_all_orig = None

        self.verbose = False

        self.hm_params = {}
        self.glm_all = {}
        self.gpr_all = {}
        self.cuts = []

        for it in reversed(range(self.iteration + 1)): # Loop over previous iterations
            cuts_dir = os.path.join('..', 'iter%d'%it, self.cut_dir)

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                print(f'Reading iteration {it}. cut {cut_name}')
                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                print('\t Desired Result:', hm.desired_result)
                print('\t Desired Result Var:', hm.desired_result_var)
                print('\t Discrepancy Var:', hm.discrepancy_var)
                print('\t Imp Thresh:', hm.implausibility_threshold)

                if self.param_info is None:
                    self.param_info = hm.param_info

                    self.Xcols_all_orig = self.param_info.index.get_level_values('Name').unique().tolist()
                    candidates = pd.DataFrame(columns=self.Xcols_all_orig) #TODO(dklein): This variable isn't used. Do you want it?

                self.hm_params[(it, cut_name)] = {
                    'desired_result':hm.desired_result,
                    'desired_result_var':hm.desired_result_var,
                    'discrepancy_var':hm.discrepancy_var,
                    'implausibility_threshold':hm.implausibility_threshold,
                }

                self.glm_all[(it, cut_name)] = GLM.from_config(os.path.join(cuts_dir, cut_name, 'GLM', 'model.json'), os.path.join(cuts_dir, cut_name, 'GLM', 'params.p'))
                self.gpr_all[(it, cut_name)] = GPR.from_config(os.path.join(cuts_dir, cut_name, 'GPR', 'model_with_test_data.json'))
                self.cuts.append((it, cut_name))


    #TODO(dklein): The constraint argument isn't used. Is it needed?
    def test_plausibility(self, points, constraint=None):
        points = points.copy()
        result = pd.DataFrame({
            'Implausible': np.zeros(points.shape[0], dtype=bool),
            'Min Implausibility': np.inf * np.ones(points.shape[0])
        })
        result.index.name = 'Sample'

        cols = []
        for cut in self.cuts:
            (it, cut_name) = cut

            print(f'Testing implausibility: iteration {it}, cut {cut_name}')

            t = time.time()
            points['Yglm'] = self.glm_all[cut].evaluate(points)
            if self.verbose:
                print('GLM:', time.time()-t)

            t = time.time()
            ret = self.gpr_all[cut].evaluate(points)
            if self.verbose:
                print('GPR:', time.time()-t)
            points['Mean_Estimate'] = points['Yglm'] + ret['Mean']
            points['Var_Predictive'] = ret['Var_Predictive']

            points[f'Implausibility_{it}_{cut_name}'] = \
                abs(points['Mean_Estimate'] - self.hm_params[cut]['desired_result']) / \
                np.sqrt(points['Var_Predictive'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'])


            points[f'Implausible_{it}_{cut_name}'] = points[f'Implausibility_{it}_{cut_name}'] > self.hm_params[cut]['implausibility_threshold']
            cols += [f'Implausibility_{it}_{cut_name}', f'Implausible_{it}_{cut_name}']

            result['Implausible'] |= points[f'Implausible_{it}_{cut_name}']
            result['Min Implausibility'] = pd.concat([
                result['Min Implausibility'],
                points[f'Implausibility_{it}_{cut_name}']
                ], axis=1) \
                .min(axis=1)

        return result

    def plot_implausibility(self, x, y, **kwargs):
        res = kwargs.get('resolution', 100)

        implausibility = kwargs['data']
        implausible = implausibility['Implausible']

        #TODO(dklein): Is there a reason not to put these imports at the top?
        #TODO(dklein): Should sklearn be a dependency listed in setup.py?
        from sklearn.kernel_ridge import KernelRidge
        #clf = KernelRidge(alpha=1, kernel='gaussian')
        from sklearn.gaussian_process.kernels import ConstantKernel, RBF
        kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
        clf = KernelRidge(alpha=1, kernel=kernel) # gaussian
        X = pd.concat([x, y], axis=1)
        clf.fit(X, implausible)

        xx = np.linspace(x.min(), x.max(), res)
        yy = np.linspace(y.min(), y.max(), res)
        [x1, x2] = np.meshgrid(xx, yy)
        x1f = x1.flatten()
        x2f = x2.flatten()
        test_grid = pd.DataFrame(np.column_stack((x1f, x2f)), columns=[x.name, y.name])
        test_grid['Pred'] = clf.predict(test_grid)

        plt.contourf(
            np.reshape(test_grid[x.name], (res, res)),
            np.reshape(test_grid[y.name], (res, res)),
            np.reshape(test_grid['Pred'], (res, res)),
            #cmap = plt.cm.jet, #TODO(dklein): Do you want these?
            #vmin=0,
            #vmax=1
        )
        plt.colorbar()

        #TODO(dklein): Do we still need the following?
        #sns.kdeplot(x, y)
        '''
        points = pd.concat([x,y], axis=1)
        points['Intercept'] = 1
        points['x2'] = x.multiply(x)
        points['y2'] = y.multiply(y)
        points['xy'] = x.multiply(y)
        logit = sm.Logit(~implausibility['Implausible'], points)
        result = logit.fit()

        print result.conf_int()

        xx = np.linspace(x.min(), x.max(), res)
        yy = np.linspace(y.min(), y.max(), res)
        [x1, x2] = np.meshgrid(xx, yy)
        x1f = x1.flatten()
        x2f = x2.flatten()
        test_grid = pd.DataFrame( np.column_stack((x1f, x2f)), columns = [x.name, y.name] )
        test_grid['Intercept'] = 1
        test_grid['x2'] = np.multiply(x1f, x1f)
        test_grid['y2'] = np.multiply(x2f, x2f)
        test_grid['xy'] = np.multiply(x1f, x2f)

        test_grid['Pred'] = result.predict( test_grid )

        plt.contourf(
            np.reshape(test_grid[x.name], (res,res)),
            np.reshape(test_grid[y.name], (res,res)),
            np.reshape(test_grid['Pred'], (res,res)),
            #cmap = plt.cm.jet,
            #vmin=0,
            #vmax=1
        )
        plt.colorbar()
        '''

        plt.scatter(x.loc[implausible], y.loc[implausible], 5, color='r')
        plt.scatter(x.loc[~implausible], y.loc[~implausible], 5, color='g')

        plt.show()

    #TODO(dklein): Variables is unused. Can it be removed?
    def plot(self, variables=None):

        implausibility = self.test_plausibility(self.samples, constraint=None)

        g = sns.PairGrid(self.samples)
        g.map_upper(self.plot_implausibility, data=implausibility)
        #g.map_upper(plt.scatter)
        #g.map_lower(sns.kdeplot, cmap="Blues_d", clip=(-50,50))
        #g.map_diag(sns.kdeplot, lw=3, legend=False);
        #g.set(xlim=(-50, 50), ylim=(-100, 100))

        plt.show()
        return g
