import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pyDOE import lhs

from history_matching.HistoryMatching import HistoryMatching
from history_matching.glm import GLM
from history_matching.gpr import GPR
#pd.set_option('mode.chained_assignment', 'raise')

logger = logging.getLogger(__name__)


class HistoryMatchingCut():

    def __init__(self, cut_folder: str, iteration: int, iterdir_parent: Path, hdf_file: Path):

        self.cut_folder = cut_folder
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

        self.iter_dir = iterdir_parent / f"iter{iteration}"

        if hdf_file == None:
            self.hdf_file = self.iter_dir / f"Candidates_for_iter{self.iteration+1}.hd5"
        else:
            assert( hdf_file.suffix.lower() in [".hd5", ".hdf"] )
            self.hdf_file = hdf_file

        for it in reversed(range(self.iteration + 1)): # Loop over previous iterations

            cuts_dir = self.iterdir_parent / f"iter{it}" / self.cut_folder

            for cut_name in [entry.name for entry in cuts_dir.iterdir() if entry.is_dir()]:

                logger.info(f'Reading iteration {it} cut {cut_name}')
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

                cut_dir = cuts_dir / cut_name
                glm_dir = cut_dir / "GLM"
                self.glm_all[(it, cut_name)] = GLM.from_config( glm_dir / "model.json", glm_dir / "params.p")
                gpr_dir = cut_dir / "GPR"
                self.gpr_all[(it, cut_name)] = GPR.from_config( gpr_dir / "model_with_test_data.json" )
                self.cuts.append((it, cut_name))

        return

    def assess_plausibility(self, points, constraint = None):

        points['Implausible'] = False

        for cut in reversed(self.cuts):
            (it, cut_name) = cut

            logger.info(f'Performing cut: iteration {it}, cut{cut_name}')
            points.loc[:,'Yglm'] = self.glm_all[cut].evaluate(points)
            ret = self.gpr_all[cut].evaluate(points)

            points.rename(columns={ 'Yglm': f'Yglm_{it}_{cut_name}', }, inplace=True)

            points.loc[:,f'Mean_Estimate_{it}_{cut_name}'] = points[f'Yglm_{it}_{cut_name}'] + ret['Mean']
            points.loc[:,f'Var_Predictive_{it}_{cut_name}'] = ret['Var_Predictive']

            points.loc[:,f'Implausibility_{it}_{cut_name}' ] = \
                abs( points[f'Mean_Estimate_{it}_{cut_name}'] - self.hm_params[cut]['desired_result'] ) / \
                np.sqrt(points[f'Var_Predictive_{it}_{cut_name}'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'] )

            points.loc[:,f'Implausible_{it}_{cut_name}' ] = points[ f'Implausibility_{it}_{cut_name}' ] > self.hm_params[cut]['implausibility_threshold']
            points['Implausible'] |= points.loc[:,f'Implausible_{it}_{cut_name}' ]

        return points

    def test_plausibility(self, points, constraint = None):

        new_candidates = points.copy()
        new_candidates['Implausible'] = False

        for cut in self.cuts:

            (it, cut_name) = cut

            plausible_candidates = new_candidates.loc[new_candidates['Implausible']==False,:]

            logger.debug(f'plausible_candidates.shape: {plausible_candidates.shape}')
            if plausible_candidates.shape[0] == 0:
                logger.info('Returning early because none of the candidates are plausible.')
                return new_candidates['Implausible']

            logger.info(f'Performing cut: iteration {it}, cut {cut_name}')
            t = time.time()
            plausible_candidates.loc[:,'Yglm'] = self.glm_all[cut].evaluate(plausible_candidates)
            logger.debug(f'GLM:{time.time()-t}'); t=time.time()
            ret = self.gpr_all[cut].evaluate(plausible_candidates)
            logger.debug(f'GPR:{time.time()-t}'); t=time.time()
            plausible_candidates.loc[:,'Mean_Estimate'] = plausible_candidates['Yglm'] + ret['Mean']
            plausible_candidates.loc[:,'Var_Predictive'] = ret['Var_Predictive']

            plausible_candidates.loc[:,'Implausibility_%d_%s'%(it, cut_name) ] = \
                abs( plausible_candidates['Mean_Estimate'] - self.hm_params[cut]['desired_result'] ) / \
                np.sqrt(plausible_candidates['Var_Predictive'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'] )

            plausible_candidates.loc[:,'Implausible_%d_%s'%(it, cut_name) ] = plausible_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] > self.hm_params[cut]['implausibility_threshold']

            new_candidates['Implausible'] |= plausible_candidates[ 'Implausible_%d_%s'%(it, cut_name) ]

        return new_candidates['Implausible']

    def cut(self, num_desired_candidates = 5000, constraint = None):

        non_implausible_candidates = pd.DataFrame()
        num_trials = 0

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        while stats['num_plausible_candidates'] < num_desired_candidates:

            logger.info("-"*80)
            max_nSamples = 10000 # TODO: make a parameter or determine from GPU info
            # Min here to avoid running out of GPU ram!
            if stats['num_candidates'] == 0:# or stats['num_plausible_candidates'] == 0:
                nSamples = min(max_nSamples, num_desired_candidates)
            else:
                nSamples = min(max_nSamples, int(round(1.25 * (num_desired_candidates-stats['num_plausible_candidates']) / ((1+stats['num_plausible_candidates'])/float(stats['num_candidates'])))))

            t = time.time()
            lhs_sample = lhs( len(self.Xcols_all_orig), samples=nSamples)
            logger.debug(f'LHS Sampling ({nSamples}):{time.time() - t}')

            t = time.time()
            for i, xc in enumerate(self.Xcols_all_orig):
                v = self.param_info.loc[xc]
                lhs_sample[:, i] = (v['Max'] - v['Min']) * lhs_sample[:, i] + (v['Min'])
            logger.debug(f'LHS Scaling:{time.time() - t}')

            t = time.time()
            new_candidates = pd.DataFrame( lhs_sample, columns=self.Xcols_all_orig)
            logger.debug(f'DataFrame:{time.time() - t}')
            t = time.time()
            if constraint is not None:
                #new_candidates = new_candidates.loc[new_candidates.apply(constraint, axis=1),:]
                #new_candidates = new_candidates.query(constraint)
                new_candidates = new_candidates.loc[constraint(new_candidates),:]
            logger.debug(f'Constraint:{time.time() - t}')

            t = time.time()
            plausibility = self.test_plausibility(new_candidates, constraint)
            logger.debug(f'Test plausibility:{time.time() - t}')

            t = time.time()
            new_candidates = new_candidates.merge(plausibility.to_frame(), left_index=True, right_index=True)
            logger.debug(f'Merge plausibility (needed?):{time.time() - t}')
            #new_candidates['Implausible'] = False

            num_trials += new_candidates.shape[0]
            new_non_implausible_candidates = new_candidates.loc[ new_candidates['Implausible'] == False, :]
            non_implausible_candidates = non_implausible_candidates.append(new_non_implausible_candidates)

            stats['num_new_plausible_candidates'] = new_non_implausible_candidates.shape[0] # sum(new_candidates['Implausible'] == False)
            stats['num_plausible_candidates'] = non_implausible_candidates.shape[0]
            stats['num_candidates'] += num_trials

            del new_candidates

            logger.info(f"Plausible candidates: New ={stats['num_new_plausible_candidates']}, Tot ={stats['num_plausible_candidates']}")

        #non_implausible_candidates = candidates.loc[ candidates['Implausible'] == False, :]

        logger.info(f'Saving to:{self.hdf_file}')
        hdf = pd.HDFStore(self.hdf_file)
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

        d = self.hdf_file.parent
        name = self.hdf_file.stem
        stats_fn = d / f"{name}_stats.json"
        with stats_fn.open("w") as f:
            json.dump(stats, f)

        csv_fn = d / f"{name}.csv"
        non_implausible_candidates[self.Xcols_all_orig].to_csv(csv_fn, index=False)

        '''
        writer = pd.ExcelWriter('Candidates_for_iter%d.xlsx'%(self.iteration+1))
        non_implausible_candidates[self.Xcols_all_orig].to_excel(writer, sheet_name='Values', index=False)
        non_implausible_candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='NonImplausible')
        candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='All')
        writer.save()
        '''

        logger.info(f"Rejected {rejected_percent:.1f}% [{(num_trials-non_implausible_candidates.shape[0]):d} / {num_trials:d}]")

        return (non_implausible_candidates, stats)
