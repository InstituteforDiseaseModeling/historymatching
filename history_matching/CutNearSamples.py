import json
import os
from pyDOE import lhs
import pandas as pd
import numpy as np
from history_matching import HistoryMatching
from glm import GLM
from gpr import GPR

class CutNearSamples():

    def __init__(self, cut_dir, iteration, seeds, blur_fraction_of_range = 0.1):
        self.cut_dir = cut_dir
        self.iteration = iteration
        self.seeds = seeds # Center points for MVNs
        self.blur_fraction_of_range = blur_fraction_of_range # Points are displaced before kernel density estimation, which is then resamples.  This parameter determines the fration of the parameter range from which a U( -blur_fraction_of_range * RANGE, blur_fraction_of_range * RANGE) random perturbation is selected.  Bigger numbers mean a higher rejection rate because perturbed samples will be farther from their seeds.
        assert(blur_fraction_of_range > 0)
        assert(blur_fraction_of_range < 1)

        self.param_info = None
        self.Xcols_all_orig = None

        self.hm_params = {}
        self.glm_all = {}
        self.gpr_all = {}
        self.cuts = []

        for it in reversed(range(self.iteration + 1)): # Loop over previous iterations
            cuts_dir = os.path.join('..', 'iter%d'%it, self.cut_dir)

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                print('Reading iteration %d. cut %s' % (it,cut_name) )
                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                print '\t Desired Result:', hm.desired_result
                print '\t Desired Result Var:', hm.desired_result_var
                print '\t Discrepancy Var:', hm.discrepancy_var
                print '\t Imp Thresh:', hm.implausibility_threshold

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

            if plausible_candidates.shape[0] == 0:
                print 'Returning early because none of the candidates are plausible.'
                return new_candidates['Implausible']

            print('Performing cut: iteration %d, cut %s' % (it,cut_name) )
            plausible_candidates['Yglm'] = self.glm_all[cut].evaluate(plausible_candidates)
            ret = self.gpr_all[cut].evaluate(plausible_candidates)
            plausible_candidates['Mean_Estimate'] = plausible_candidates['Yglm'] + ret['Mean']
            plausible_candidates['Var_Predictive'] = ret['Var_Predictive']

            plausible_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] = \
                abs( plausible_candidates['Mean_Estimate'] - self.hm_params[cut]['desired_result'] ) / \
                np.sqrt(plausible_candidates['Var_Predictive'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'] )

            plausible_candidates[ 'Implausible_%d_%s'%(it, cut_name) ] = plausible_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] > self.hm_params[cut]['implausibility_threshold']
            cols += ['Implausibility_%d_%s'%(it, cut_name), 'Implausible_%d_%s'%(it, cut_name)]

            new_candidates['Implausible'] |= plausible_candidates[ 'Implausible_%d_%s'%(it, cut_name) ]

        return new_candidates['Implausible']


    def cut(self, num_desired_candidates = 5000, constraint = None):
        non_implausible_candidates = pd.DataFrame()
        num_trials = 0

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        while stats['num_plausible_candidates'] < num_desired_candidates:
            print '-'*80
            max_nSamples = 25000 #5000 # TODO: Make parameter
            # Min here to avoid running out of GPU ram!
            if stats['num_candidates'] == 0:# or stats['num_plausible_candidates'] == 0:
                nSamples = min(max_nSamples, num_desired_candidates)
            else:
                nSamples = min(max_nSamples, int(round(1.25 * (num_desired_candidates-stats['num_plausible_candidates']) / ((1+stats['num_plausible_candidates'])/float(stats['num_candidates'])))))

            print 'Testing (%d):'%nSamples

            #lhs_sample = lhs(len(self.Xcols_all_orig), samples=nSamples)

            #from sklearn.neighbors.kde import KernelDensity
            #kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
            #print kde.get_params()

            # BLUR THE SEEDS TO GET GOOD COVERAGE
            #sample = self.seeds.copy()
            #while sample.shape[0] < nSamples:
            #    sample = sample.append(self.seeds.copy()
            sample_seeds = self.seeds.sample(n=nSamples, replace=True).reset_index(drop=True)
            sample = sample_seeds.copy()

            if sample.isnull().any().any():
                print 'HAVE NULL BEFORE\n'*80

            #N = self.seeds.shape[0]
            for i, xc in enumerate(self.Xcols_all_orig):
                v = self.param_info.loc[xc]
                sample[xc] += \
                    np.random.uniform(
                        low=-self.blur_fraction_of_range*(v['Max']-v['Min']),
                        high=self.blur_fraction_of_range*(v['Max']-v['Min']),
                        size=sample.shape[0] )
                #sample[xc] = np.clip(sample[xc], v['Min'], v['Max'])

                # Resample points that are outside of Min-Max
                bad_inds = sample[ (sample[xc] < v['Min']) |  (sample[xc] > v['Max'])].index
                #print 'Starting with %d bad rows' % bad_inds.size
                while bad_inds.size > 0:
                    sample.loc[bad_inds, xc] = sample_seeds.loc[bad_inds, xc] + \
                        np.random.uniform(
                            low=-self.blur_fraction_of_range*(v['Max']-v['Min']),
                            high=self.blur_fraction_of_range*(v['Max']-v['Min']),
                            size=bad_inds.size )
                    bad_inds = sample[ (sample[xc] < v['Min']) |  (sample[xc] > v['Max'])].index
                    #print ' --> Now have %d bad rows' % bad_inds.size

            if sample.isnull().any().any():
                print 'HAVE NULL AFTER\n'*80
                if pd.isnull(sample).any().any():
                    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                        print sample[sample.isnull().any(axis=1)]
                    print 'Data contains Null/None/NaN, see data above.'
                    exit()

            #if sample.shape[0] > nSamples:
            #    sample = sample[:nSamples]

            #kde.fit(sample)
            #sample = kde.sample(n_samples = nSamples)

            new_candidates = pd.DataFrame( sample, columns=self.Xcols_all_orig)
            if constraint is not None:
                #new_candidates = new_candidates.loc[new_candidates.apply(constraint, axis=1),:]
                #new_candidates = new_candidates.query(constraint)
                new_candidates = new_candidates.loc[constraint(new_candidates),:]

            plausibility = self.test_plausibility(new_candidates, constraint)

            new_candidates = new_candidates.merge(plausibility.to_frame(), left_index=True, right_index=True)
            #new_candidates['Implausible'] = False


            num_trials += new_candidates.shape[0]
            new_non_implausible_candidates = new_candidates.loc[ new_candidates['Implausible'] == False, :]
            non_implausible_candidates = non_implausible_candidates.append(new_non_implausible_candidates)

            stats['num_new_plausible_candidates'] = new_non_implausible_candidates.shape[0] # sum(new_candidates['Implausible'] == False)
            stats['num_plausible_candidates'] = non_implausible_candidates.shape[0]
            stats['num_candidates'] += num_trials

            del new_candidates

            print 'Plausible candidates: New = %d, Tot = %d' % (stats['num_new_plausible_candidates'], stats['num_plausible_candidates'])

        #non_implausible_candidates = candidates.loc[ candidates['Implausible'] == False, :]

        fn = 'Candidates_NS_for_iter%d.hd5'%(self.iteration+1)
        print 'Saving to:', fn
        hdf = pd.HDFStore(fn)
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

        with open('cut_stats_ns.json', 'w') as f:
            json.dump(stats, f)

        print 'Rejected %.1f%% [%d / %d]' % (rejected_percent, (num_trials-non_implausible_candidates.shape[0]), num_trials)


        '''
        writer = pd.ExcelWriter('Candidates_for_iter%d.xlsx'%(self.iteration+1))
        non_implausible_candidates[self.Xcols_all_orig].to_excel(writer, sheet_name='Values', index=False)
        non_implausible_candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='NonImplausible')
        candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='All')
        writer.save()
        '''

        return (non_implausible_candidates, rejected_percent)
