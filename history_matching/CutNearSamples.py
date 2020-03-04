import json
import os
import pandas as pd
import numpy as np
from history_matching import HistoryMatching
from history_matching.glm import GLM
from history_matching.gpr import GPR

from history_matching.error import *

class CutNearSamples():

    def __init__(self, cut_dir, iteration, seeds, blur_fraction_of_range = 0.1, saveto_hd5 = None):
        self.cut_dir = cut_dir
        self.iteration = iteration
        self.seeds = seeds # Center points for MVNs
        self.blur_fraction_of_range = blur_fraction_of_range # Points are displaced before kernel density estimation, which is then resamples.  This parameter determines the fration of the parameter range from which a U( -blur_fraction_of_range * RANGE, blur_fraction_of_range * RANGE) random perturbation is selected.  Bigger numbers mean a higher rejection rate because perturbed samples will be farther from their seeds.
        
        #TODO: Can these be <= ?
        if not 0<blur_fraction_of_range<1:
            raise HistoryMatchingError("blur_fraction_of_range must be in the range (0,1)")

        self.param_info = None
        self.Xcols_all_orig = None

        self.hm_params = {}
        self.glm_all = {}
        self.gpr_all = {}
        self.cuts = []

        if saveto_hd5 == None:
            self.saveto_hd5 = 'Candidates_NS_for_iter%d.hd5'%(self.iteration+1)
        elif os.path.splitext(saveto_hd5)[1].lower() not in ['hd5', 'hdf']:
            raise HistoryMatchingError("saveto_hd5 must end with 'hd5' or 'hdf'!")
        else:
            self.saveto_hd5 = saveto_hd5

        for it in reversed(range(self.iteration + 1)): # Loop over previous iterations
            cuts_dir = os.path.join('..', 'iter%d'%it, self.cut_dir)

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                print('Reading iteration {it}. cut {cut_name}')
                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                print('\t Desired Result:    ', hm.desired_result)
                print('\t Desired Result Var:', hm.desired_result_var)
                print('\t Discrepancy Var:   ', hm.discrepancy_var)
                print('\t Imp Thresh:        ', hm.implausibility_threshold)

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
            it, cut_name = cut

            plausible_candidates = new_candidates.loc[new_candidates['Implausible']==False,:]

            if plausible_candidates.shape[0] == 0:
                print('Returning early because none of the candidates are plausible.')
                return new_candidates['Implausible']

            print(f'Performing cut: iteration {it}, cut {cut_name}')
            plausible_candidates['Yglm'] = self.glm_all[cut].evaluate(plausible_candidates)
            ret = self.gpr_all[cut].evaluate(plausible_candidates)
            plausible_candidates['Mean_Estimate'] = plausible_candidates['Yglm'] + ret['Mean']
            plausible_candidates['Var_Predictive'] = ret['Var_Predictive']

            plausible_candidates[f'Implausibility_{it}_{cut_name}'] = \
                abs( plausible_candidates['Mean_Estimate'] - self.hm_params[cut]['desired_result'] ) / \
                np.sqrt(plausible_candidates['Var_Predictive'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'] )

            plausible_candidates[f'Implausible_{it}_{cut_name}'] = plausible_candidates[f'Implausibility_{it}_{cut_name}'] > self.hm_params[cut]['implausibility_threshold']
            cols += [f'Implausibility_{it}_{cut_name}', f'Implausible_{it}_{cut_name}']

            new_candidates['Implausible'] |= plausible_candidates[f'Implausible_{it}_{cut_name}']

        return new_candidates['Implausible']

    def calc_seed_prob(self):
        # returns p: the volume of the symmetric space around each seed
        p = np.ones(self.seeds.shape[0])
        for col_name, col_series in self.seeds.iteritems():
            v = self.param_info.loc[col_name]
            v_range = self.blur_fraction_of_range * (v['Max']-v['Min'])

            right_range = (v['Max'] - col_series.values) / v_range
            left_range = 1.0/self.blur_fraction_of_range - right_range # (values - v['Min']) / v_range   #TODO(dklein): Should this be commented out?

            blur_range = np.minimum(np.minimum(1.0, right_range), left_range)

            p = np.multiply(p, blur_range)

        p /= sum(p)

        return p


    def draw_samples(self, nSamples, p):
        seed_inds = np.random.choice(self.seeds.shape[0], size=nSamples, replace=True, p=p)
        seeds = self.seeds.loc[seed_inds]

        #seeds = self.seeds.sample(n=nSamples, replace=True).reset_index(drop=True)
        sample = seeds.copy()

        #TODO: Needs unit test to show that it does indeed sample near the seeds and does something logical when the seeds are near the parameter boundaries (see `misc/misc.md`)
        for col_name, col_series in seeds.iteritems():
            v = self.param_info.loc[col_name]
            v_range = (v['Max']-v['Min'])

            right_range = v['Max'] - col_series.values
            left_range  = col_series.values - v['Min']

            blur_range = np.minimum(np.minimum(self.blur_fraction_of_range*v_range, right_range), left_range)

            sample[col_name] += -blur_range + np.multiply(2*blur_range, np.random.rand(nSamples))

        return sample

    def cut(self, num_desired_candidates = 5000, constraint = None):
        non_implausible_candidates = pd.DataFrame()
        num_trials = 0

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        p = self.calc_seed_prob()

        while stats['num_plausible_candidates'] < num_desired_candidates:
            print('-'*80)
            max_nSamples = 25000 #5000 # TODO: Make parameter
            # Min here to avoid running out of GPU ram!
            if stats['num_candidates'] == 0:# or stats['num_plausible_candidates'] == 0:
                nSamples = min(max_nSamples, num_desired_candidates)
            else:
                nSamples = min(max_nSamples, int(round(1.25 * (num_desired_candidates-stats['num_plausible_candidates']) / ((1+stats['num_plausible_candidates'])/float(stats['num_candidates'])))))

            print('Starting with (%d):'%nSamples)

            print('initialy draw_samples:')
            sample = self.draw_samples(nSamples, p)
            print('data frame and constraint:')
            new_candidates = pd.DataFrame( sample, columns=self.Xcols_all_orig)
            if constraint is not None:
                #new_candidates = new_candidates.loc[new_candidates.apply(constraint, axis=1),:]
                #new_candidates = new_candidates.query(constraint)
                new_candidates = new_candidates.loc[constraint(new_candidates),:]

            print('entering while loop:')
            while new_candidates.shape[0] < nSamples:
                print('draw_samples in while loop (%d):'%new_candidates.shape[0])
                samples = self.draw_samples(nSamples, p)
                print('data frame in while loop:')
                sample_df = pd.DataFrame( samples, columns=self.Xcols_all_orig )
                print('sample_df has rows numbering %d:'%sample_df.shape[0])
                if constraint is not None:
                    print('constraint evaluation in while loop:')
                    sample_df = sample_df.loc[constraint(sample_df),:]
                print('appending sample_df to new_candidates.  was %d:' % new_candidates.shape[0])
                new_candidates = new_candidates.append( sample_df, ignore_index=True )
                print('new_candidates.  now %d:' % new_candidates.shape[0])


            print(f'Testing ({nSamples}):')

            plausibility = self.test_plausibility(new_candidates, constraint)

            new_candidates = new_candidates.merge(plausibility.to_frame(), left_index=True, right_index=True)

            num_trials += new_candidates.shape[0]
            new_non_implausible_candidates = new_candidates.loc[ new_candidates['Implausible'] == False, :]
            non_implausible_candidates = non_implausible_candidates.append(new_non_implausible_candidates)

            stats['num_new_plausible_candidates'] = new_non_implausible_candidates.shape[0] # sum(new_candidates['Implausible'] == False)
            stats['num_plausible_candidates'] = non_implausible_candidates.shape[0]
            stats['num_candidates'] += num_trials

            del new_candidates

            print('Plausible candidates: New = %d, Tot = %d' % (stats['num_new_plausible_candidates'], stats['num_plausible_candidates']))

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

        d, filename = os.path.split(self.saveto_hd5)
        name, _ = os.path.splitext(filename)
        stats_fn = os.path.join(d, name + '_stats.json')
        with open(stats_fn, 'w') as f:
            json.dump(stats, f)

        csv_fn = os.path.join(d, name + '.xlsx')
        non_implausible_candidates[self.Xcols_all_orig].to_csv(csv_fn)

        '''
        writer = pd.ExcelWriter('Candidates_for_iter%d.xlsx'%(self.iteration+1))
        non_implausible_candidates[self.Xcols_all_orig].to_excel(writer, sheet_name='Values', index=False)
        non_implausible_candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='NonImplausible')
        candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='All')
        writer.save()
        '''

        print('Rejected %.1f%% [%d / %d]' % (rejected_percent, (num_trials-non_implausible_candidates.shape[0]), num_trials))

        return (non_implausible_candidates, stats)
