import json
import os
import time
from pyDOE import lhs
import pandas as pd
import numpy as np
from history_matching import HistoryMatching
from history_matching.glm import GLM
from history_matching.gpr import GPR
import logging

logger = logging.getLogger(__name__)

class CutNearSamples():

    def __init__(self, cut_dir, iteration, seeds, blur_fraction_of_range = 0.1, iterdir_parent=None, saveto_hd5 = None):
        self.cut_dir = cut_dir
        self.iteration = iteration
        self.seeds = seeds # Center points for MVNs
        self.blur_fraction_of_range = blur_fraction_of_range # Points are displaced before kernel density estimation, which is then resamples.  This parameter determines the fration of the parameter range from which a U( -blur_fraction_of_range * RANGE, blur_fraction_of_range * RANGE) random perturbation is selected.  Bigger numbers mean a higher rejection rate because perturbed samples will be farther from their seeds.
        assert(blur_fraction_of_range > 0)
        assert(blur_fraction_of_range < 1)

        if iterdir_parent == None:
            self.iterdir_parent = '..' # Folder containing iter0, iter1, ...
        else:
            self.iterdir_parent = iterdir_parent

        self.param_info = None
        self.Xcols_all_orig = None

        self.debug = False

        self.hm_params = {}
        self.glm_all = {}
        self.gpr_all = {}
        self.cuts = []

        if saveto_hd5 == None:
            self.saveto_hd5 = 'Candidates_NS_for_iter%d.hd5'%(self.iteration+1)
        else:
            assert( os.path.splitext(saveto_hd5)[1].lower() in ['.hd5', '.hdf'] )
            self.saveto_hd5 = saveto_hd5

        for it in reversed(range(self.iteration + 1)): # Loop over previous iterations
            cuts_dir = os.path.join(self.iterdir_parent, 'iter%d'%it, self.cut_dir)

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                logger.info(f'Reading iteration {it}. cut {cut_name}')

                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                logger.info(f'\t Desired Result: {hm.desired_result}')
                logger.info(f'\t Desired Result Var: {hm.desired_result_var}')
                logger.info(f'\t Discrepancy Var: {hm.discrepancy_var}')
                logger.info(f'\t Imp Thresh: {hm.implausibility_threshold}')

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

            logger.debug(plausible_candidates.shape)
            if plausible_candidates.shape[0] == 0:
                logger.info('Returning early because none of the candidates are plausible.')
                return new_candidates['Implausible']

            logger.info(f'Performing cut: iteration {it}, cut {cut_name}')
            t = time.time()
            plausible_candidates.loc[:,'Yglm'] = self.glm_all[cut].evaluate(plausible_candidates)
            logger.debug(f'GLM: {time.time()-t}'); t=time.time()
            ret = self.gpr_all[cut].evaluate(plausible_candidates)
            logger.debug(f'GPR: {time.time()-t}'); t=time.time()
            plausible_candidates.loc[:,'Mean_Estimate'] = plausible_candidates['Yglm'] + ret['Mean']
            plausible_candidates.loc[:,'Var_Predictive'] = ret['Var_Predictive']

            plausible_candidates.loc[:, 'Implausibility_%d_%s'%(it, cut_name) ] = \
                abs( plausible_candidates['Mean_Estimate'] - self.hm_params[cut]['desired_result'] ) / \
                np.sqrt(plausible_candidates['Var_Predictive'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'] )

            plausible_candidates.loc[:, 'Implausible_%d_%s'%(it, cut_name) ] = plausible_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] > self.hm_params[cut]['implausibility_threshold']
            cols += ['Implausibility_%d_%s'%(it, cut_name), 'Implausible_%d_%s'%(it, cut_name)]

            new_candidates.loc[new_candidates['Implausible']==False,'Implausible'] |= plausible_candidates[ 'Implausible_%d_%s'%(it, cut_name) ]

        return new_candidates['Implausible']

    def calc_seed_prob(self):
        # returns p: the volume of the symmetric space around each seed
        p = np.ones(self.seeds.shape[0])
        for col_name, col_series in self.seeds.iteritems():
            v = self.param_info.loc[col_name]
            v_range = self.blur_fraction_of_range * (v['Max']-v['Min'])

            values = col_series.values
            right_range = (v['Max'] - values) / v_range
            left_range = 1.0/self.blur_fraction_of_range - right_range # (values - v['Min']) / v_range

            blur_range = np.minimum(np.minimum(1.0, right_range), left_range)

            p = np.multiply(p, blur_range)

        p /= sum(p)

        return p


    def draw_samples(self, nSamples, p):
        seed_inds = np.random.choice(self.seeds.shape[0], size=nSamples, replace=True, p=p)
        seeds = self.seeds.iloc[seed_inds]

        #seeds = self.seeds.sample(n=nSamples, replace=True).reset_index(drop=True)
        sample = seeds.copy()

        for col_name, col_series in seeds.iteritems():
            v = self.param_info.loc[col_name]
            v_range = (v['Max']-v['Min'])

            values = col_series.values
            right_range = (v['Max'] - values)
            left_range = (values - v['Min'])

            blur_range = np.minimum(np.minimum(self.blur_fraction_of_range*v_range, right_range), left_range)

            sample[col_name] += -blur_range + np.multiply(2*blur_range, np.random.rand(nSamples))

        return sample


        '''
        print('Drawing %d samples:'%nSamples)

        seeds = self.seeds.sample(n=nSamples, replace=True).reset_index(drop=True)
        sample = seeds.copy()

        for row_idx, row in seeds.iterrows():
            for (col_name, value) in row.iteritems():
                v = self.param_info.loc[col_name]
                v_range = (v['Max']-v['Min'])

                right_frac = (v['Max'] - value) / v_range
                left_frac = (value - v['Min']) / v_range

                blur_frac = np.min([self.blur_fraction_of_range, right_frac, left_frac])

                sample.loc[row_idx, col_name] = value + \
                        np.random.uniform(
                            low = -blur_frac * v_range,
                            high = blur_frac * v_range,
                            size = 1 )

        print('DONE drawing %d samples:'%nSamples)

        return sample
        '''

        '''
        print('in draw_samples, nSamples=%d' % nSamples)

        good = np.ones(nSamples, dtype=bool)
        sample = self.seeds.sample(n=nSamples, replace=True).reset_index(drop=True)
        print('in draw_samples, sample len is %d' % sample.shape[0])
        print('in draw_samples, here is sample head:\n', sample.head())

        for i, xc in enumerate(self.Xcols_all_orig):
            v = self.param_info.loc[xc]
            print('in draw_samples, here is v:\n', v)
            print('blur fraction of range is %f', self.blur_fraction_of_range)
            sample[xc] += \
                np.random.uniform(
                    low=-self.blur_fraction_of_range*(v['Max']-v['Min']),
                    high=self.blur_fraction_of_range*(v['Max']-v['Min']),
                    size=sample.shape[0] )

            # Resample points that are outside of Min-Max
            df = (sample[xc] > v['Min']) & (sample[xc] < v['Max'])
            good &= df.values

        print('in draw_samples, good len is %d, sum is %d' % (good.shape[0], sum(good)))
        return sample.loc[good]
        '''


    def cut(self, num_desired_candidates = 5000, constraint = None):
        non_implausible_candidates = pd.DataFrame()
        num_trials = 0

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        p = self.calc_seed_prob()

        while stats['num_plausible_candidates'] < num_desired_candidates:
            logger.info('-'*80)
            max_nSamples = 10000 # TODO: Make parameter
            # Min here to avoid running out of GPU ram!
            if stats['num_candidates'] == 0:# or stats['num_plausible_candidates'] == 0:
                nSamples = min(max_nSamples, num_desired_candidates)
            else:
                nSamples = min(max_nSamples, int(round(1.25 * (num_desired_candidates-stats['num_plausible_candidates']) / ((1+stats['num_plausible_candidates'])/float(stats['num_candidates'])))))

            logger.info(f'Starting with ({nSamples}):')
            sample = self.draw_samples(nSamples, p)
            logger.debug(f'initially draw_samples: {sample}')
            logger.debug(f'constraint: {constraint}')
            logger.debug(f'new_candidates data frame: {new_candidates}')
            new_candidates = pd.DataFrame( sample, columns=self.Xcols_all_orig)
            if constraint is not None:
                #new_candidates = new_candidates.loc[new_candidates.apply(constraint, axis=1),:]
                #new_candidates = new_candidates.query(constraint)
                new_candidates = new_candidates.loc[constraint(new_candidates),:]

            logger.debug('entering while loop')
            while new_candidates.shape[0] < nSamples:
                logger.debug(f'draw_samples in while loop ({new_candidates.shape[0]})')
                samples = self.draw_samples(nSamples, p)
                logger.debug('data frame in while loop')
                sample_df = pd.DataFrame( samples, columns=self.Xcols_all_orig )
                logger.debug(f'sample_df has rows numbering {sample_df.shape[0]}')
                if constraint is not None:
                    logger.debug('constraint evaluation in while loop:')
                    sample_df = sample_df.loc[constraint(sample_df),:]
                logger.debug(f'before appending sample_df to new_candidates.  was {new_candidates.shape[0]}')
                new_candidates = new_candidates.append( sample_df, ignore_index=True )
                logger.debug(f'after appending sample_df to new_candidates.  now {new_candidates.shape[0]}')
                logger.debug(f'new_candidates.  now {new_candidates.shape[0]}:')


            logger.info(f'Testing ({nSamples}):')

            t = time.time()
            plausibility = self.test_plausibility(new_candidates, constraint)
            logger.debug(f'Test plausibility:{time.time() - t}')

            #t = time.time()
            ###new_candidates = new_candidates.merge(plausibility.to_frame(), left_index=True, right_index=True)
            #print('Merge plausibility (needed?):', time.time() - t)
            #new_candidates['Implausible'] = False

            num_trials += new_candidates.shape[0]
            new_non_implausible_candidates = new_candidates.loc[ plausibility == False, :]
            non_implausible_candidates = non_implausible_candidates.append(new_non_implausible_candidates)

            stats['num_new_plausible_candidates'] = new_non_implausible_candidates.shape[0] # sum(new_candidates['Implausible'] == False)
            stats['num_plausible_candidates'] = non_implausible_candidates.shape[0]
            stats['num_candidates'] += num_trials

            del new_candidates

            logger.info(f"Plausible candidates: New ={stats['num_new_plausible_candidates']}, Tot ={stats['num_plausible_candidates']}")


        logger.info(f'Saving to: {self.saveto_hd5}')
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
