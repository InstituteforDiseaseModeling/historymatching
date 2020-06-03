import json
import os
import time
from pyDOE import lhs
import pandas as pd
import numpy as np
from history_matching.HistoryMatching import HistoryMatching
from history_matching.glm import GLM
from history_matching.gpr import GPR

class HistoryMatchingCut():

    def __init__(self, cut_dir, iteration, iterdir_parent=None, saveto_hd5 = None):
        self.cut_dir = cut_dir
        self.iteration = iteration

        self.param_info = None
        self.Xcols_all_orig = None

        if iterdir_parent == None:
            self.iterdir_parent = '..' # Folder containing iter0, iter1, ...
        else:
            self.iterdir_parent = iterdir_parent

        self.debug = False

        self.hm_params = {}
        self.glm_all = {}
        self.gpr_all = {}
        self.cuts = []

        if saveto_hd5 == None:
            self.saveto_hd5 = 'Candidates_for_iter%d.hd5'%(self.iteration+1)
        else:
            assert( os.path.splitext(saveto_hd5)[1].lower() in ['.hd5', '.hdf'] )
            self.saveto_hd5 = saveto_hd5

        for it in reversed(range(self.iteration + 1)): # Loop over previous iterations
            cuts_dir = os.path.join(self.iterdir_parent, 'iter%d'%it, self.cut_dir)

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                print('Reading iteration', it, 'cut', cut_name)
                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                print('\t Desired Result:', hm.desired_result)
                print('\t Desired Result Var:', hm.desired_result_var)
                print('\t Discrepancy Var:', hm.discrepancy_var)
                print('\t Imp Thresh:', hm.implausibility_threshold)

                if self.param_info is None:
                    self.param_info = hm.param_info

                    #self.Xcols_all_orig = self.param_info.index.unique().values.tolist()
                    self.Xcols_all_orig = self.param_info.index.get_level_values('Name').unique().tolist()
                    candidates = pd.DataFrame( columns=self.Xcols_all_orig )

                self.hm_params[(it, cut_name)] = {
                    'desired_result':hm.desired_result,
                    'desired_result_var':hm.desired_result_var,
                    'discrepancy_var':hm.discrepancy_var,
                    'implausibility_threshold':hm.implausibility_threshold,
                }

                self.glm_all[(it, cut_name)] = GLM.from_config(os.path.join(cuts_dir, cut_name, 'GLM', 'model.json'), os.path.join(cuts_dir, cut_name, 'GLM', 'params.p'))
                self.gpr_all[(it, cut_name)] = GPR.from_config(os.path.join(cuts_dir, cut_name, 'GPR', 'model_with_test_data.json'))
                self.cuts.append((it, cut_name))


    def test_plausibility(self, points, constraint = None):
        new_candidates = points.copy()
        new_candidates['Implausible'] = False

        cols = []
        for cut in self.cuts:
            (it, cut_name) = cut

            plausible_candidates = new_candidates.loc[new_candidates['Implausible']==False,:]

            print(plausible_candidates.shape)
            if plausible_candidates.shape[0] == 0:
                print('Returning early because none of the candidates are plausible.')
                return new_candidates['Implausible']

            print('Performing cut: iteration', it, ', cut', cut_name)
            t = time.time()
            plausible_candidates.loc[:,'Yglm'] = self.glm_all[cut].evaluate(plausible_candidates)
            if self.debug:
                print('GLM:', time.time()-t); t=time.time()
            ret = self.gpr_all[cut].evaluate(plausible_candidates)
            if self.debug:
                print('GPR:', time.time()-t); t=time.time()
            plausible_candidates.loc[:,'Mean_Estimate'] = plausible_candidates['Yglm'] + ret['Mean']
            plausible_candidates.loc[:,'Var_Predictive'] = ret['Var_Predictive']

            plausible_candidates.loc[:,'Implausibility_%d_%s'%(it, cut_name) ] = \
                abs( plausible_candidates['Mean_Estimate'] - self.hm_params[cut]['desired_result'] ) / \
                np.sqrt(plausible_candidates['Var_Predictive'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'] )

            plausible_candidates.loc[:,'Implausible_%d_%s'%(it, cut_name) ] = plausible_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] > self.hm_params[cut]['implausibility_threshold']
            cols += ['Implausibility_%d_%s'%(it, cut_name), 'Implausible_%d_%s'%(it, cut_name)]

            new_candidates['Implausible'] |= plausible_candidates[ 'Implausible_%d_%s'%(it, cut_name) ]

        return new_candidates['Implausible']


    def cut(self, num_desired_candidates = 5000, constraint = None):
        non_implausible_candidates = pd.DataFrame()
        num_trials = 0

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        while stats['num_plausible_candidates'] < num_desired_candidates:
            print('-'*80)
            max_nSamples = 10000 # TODO: make a parameter or determine from GPU info
            # Min here to avoid running out of GPU ram!
            if stats['num_candidates'] == 0:# or stats['num_plausible_candidates'] == 0:
                nSamples = min(max_nSamples, num_desired_candidates)
            else:
                nSamples = min(max_nSamples, int(round(1.25 * (num_desired_candidates-stats['num_plausible_candidates']) / ((1+stats['num_plausible_candidates'])/float(stats['num_candidates'])))))

            t = time.time()
            lhs_sample = lhs( len(self.Xcols_all_orig), samples=nSamples)
            print('LHS Sampling (', nSamples,'):', time.time() - t)

            t = time.time()
            for i, xc in enumerate(self.Xcols_all_orig):
                v = self.param_info.loc[xc]
                lhs_sample[:, i] = (v['Max'] - v['Min']) * lhs_sample[:, i] + (v['Min'])
            print('LHS Scaling:', time.time() - t)

            t = time.time()
            new_candidates = pd.DataFrame( lhs_sample, columns=self.Xcols_all_orig)
            print('DataFrame:', time.time() - t)
            t = time.time()
            if constraint is not None:
                #new_candidates = new_candidates.loc[new_candidates.apply(constraint, axis=1),:]
                #new_candidates = new_candidates.query(constraint)
                new_candidates = new_candidates.loc[constraint(new_candidates),:]
            print('Constraint:', time.time() - t)

            t = time.time()
            plausibility = self.test_plausibility(new_candidates, constraint)
            print('Test plausibility:', time.time() - t)

            t = time.time()
            new_candidates = new_candidates.merge(plausibility.to_frame(), left_index=True, right_index=True)
            print('Merge plausibility (needed?):', time.time() - t)
            #new_candidates['Implausible'] = False


            num_trials += new_candidates.shape[0]
            new_non_implausible_candidates = new_candidates.loc[ new_candidates['Implausible'] == False, :]
            non_implausible_candidates = non_implausible_candidates.append(new_non_implausible_candidates)

            stats['num_new_plausible_candidates'] = new_non_implausible_candidates.shape[0] # sum(new_candidates['Implausible'] == False)
            stats['num_plausible_candidates'] = non_implausible_candidates.shape[0]
            stats['num_candidates'] += num_trials

            del new_candidates

            print('Plausible candidates: New =', stats['num_new_plausible_candidates'], ', Tot =', stats['num_plausible_candidates'])

        #non_implausible_candidates = candidates.loc[ candidates['Implausible'] == False, :]

        print('Saving to:', self.saveto_hd5)
        hdf = pd.HDFStore(self.saveto_hd5)
        hdf.put('values', non_implausible_candidates[self.Xcols_all_orig].reset_index(drop=True))
        #hdf.put('non_implausible', non_implausible_candidates.set_index(self.Xcols_all_orig))
        #hdf.put('all', candidates.set_index(self.Xcols_all_orig))
        hdf.close()

        rejected_percent = 100 * (num_trials-non_implausible_candidates.shape[0]) / float(num_trials)
        stats = {
            'Rejected Percent': rejected_percent,
            'Num Trials': num_trials,
            'Num Implausible': num_trials-non_implausible_candidates.shape[0]
        }

        (d, filename) = os.path.split(self.saveto_hd5)
        (name, ext) = os.path.splitext(filename)
        stats_fn = os.path.join(d, name + '_stats.json')
        with open(stats_fn, 'w') as f:
            json.dump(stats, f)

        csv_fn = os.path.join(d, name + '.csv')
        non_implausible_candidates[self.Xcols_all_orig].to_csv(csv_fn, index=False)

        '''
        writer = pd.ExcelWriter('Candidates_for_iter%d.xlsx'%(self.iteration+1))
        non_implausible_candidates[self.Xcols_all_orig].to_excel(writer, sheet_name='Values', index=False)
        non_implausible_candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='NonImplausible')
        candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='All')
        writer.save()
        '''

        print('Rejected {0:.1f}% [{1:d} / {2:d}]'.format(rejected_percent, (num_trials-non_implausible_candidates.shape[0]), num_trials))

        return (non_implausible_candidates, stats)
