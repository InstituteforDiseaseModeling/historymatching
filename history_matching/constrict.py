# constrict.py

import math
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .samplers import lhs
from .emulators import BaseEmulator


def npg(
    iteration: int,
    parameter_space: pd.DataFrame,
    observations: pd.DataFrame,
    emulator_bank: Dict[int, Dict[str, BaseEmulator]],
    config: Config,
    # consider a constraint function which imposes "business" rules on parameters sets, i.e., not all combinations may be valid, e.g. max_allowed_value < min_alowed_value
) -> Tuple[pd.DataFrame, float]:

    """Given observations, emulators, and a parameter space, generate a set of candidate points for the next iteration."""

    num_desired_candidates = config.candidates_per_iteration

    # While we need more candidates...
    count_considered = 0
    candidates = pd.DataFrame(columns=list(parameter_space["parameter"])+['implausible'])   # one column for each parameter + track implausibility
    count_plausible_candidates = 0
    while count_plausible_candidates < num_desired_candidates:

        # Generate a set of candidate points
        num_requested = 2 * (num_desired_candidates - count_plausible_candidates)
        sample_points = lhs(parameter_space, num_requested)
        count_considered += num_requested

        # Filter sample_points on constraint function, if/when implemented
        # apply constraint function to sample_points

        # Use each emulator to, potentially, weed out implausible points in parameter space
        sample_points['implausible'] = False
        for iteration in sorted(emulator_bank.keys(), reverse=True):    # start with most recent emulator(s) (iterations)
            emulators_for_iteration = emulator_bank[iteration]
            for feature, emulator in emulators_for_iteration.items():
                print(f"Testing sample points against emulator for'{feature}' (iteration {iteration})...")
                plausible_candidates = sample_points.loc[not sample_points['implausible'],:]
                emulated_values = emulator.evaluate(plausible_candidates)
                # Mean_Estimate = Yglm (from GLM) + Mean (from GPR)
                # desired_result øtarget from feature selection? (TBD)
                # Var_Predictive (from GPR)
                # desired_result_var øobservation variance? (TBD)
                # discrepancy_var ø? (TBD)
                # discrepancy = (desired_result - Mean_Estimate) / math.sqrt(desired_result_var + discrepancy_var)
                # implausible = abs(discrepancy) > config.discrepancy_threshold
                # sample_points.loc[implausible, 'implausible'] = True
                # count_plausible_candidates = len(sample_points.loc[not sample_points['implausible'],:])
                # print(f"  {count_plausible_candidates} plausible candidates found.")

    return lhs(parameter_space, config.candidates_per_iteration), 0.99  # TODO - implement


def next_point_generation(
    iteration: int,
    parameter_space: pd.DataFrame,
    observations: pd.DataFrame,
    emulator_bank: Dict[int, Dict[str, BaseEmulator]],
    config: Config,
    # consider a constraint function which imposes "business" rules on parameters sets, i.e., not all combinations may be valid, e.g. max_allowed_value < min_alowed_value
) -> Tuple[pd.DataFrame, float]:

    """Docstring TBD"""

    num_desired_candidates = config.candidates_per_iteration

    count_considered = 0
    candidates = pd.DataFrame(columns=list(parameter_space["parameter"])+['implausible'])   # one column for each parameter + track implausibility
    count_plausible_candidates = 0
    # while we need more candidates...
    while count_plausible_candidates < num_desired_candidates:

        num_requested = 2 * (num_desired_candidates - count_plausible_candidates)
        sample_points = lhs(parameter_space, num_requested)
        count_considered += num_requested

        # filter sample_points on constraint function, if/when implemented

        # use each emulator to, potentially, weed out implausible points in parameter space
        sample_points['implausible'] = False
        for iteration in sorted(emulator_bank.keys(), reverse=True):    # start with most recent emulator(s) (iterations)
            emulators_for_iteration = emulator_bank[iteration]
            for feature, emulator in emulators_for_iteration.items():
                print(f"Testing sample points against emulator for'{feature}' (iteration {iteration})...")
                plausible_candidates = sample_points.loc[not sample_points['implausible'],:]
                emulated_values = emulator.evaluate(plausible_candidates)
                # Mean_Estimate = Yglm (from GLM) + Mean (from GPR)
                # desired_result øtarget from feature selection? (TBD)
                # Var_Predictive (from GPR)
                # desired_result_var øobservation variance? (TBD)
                # discrepancy_var ø? (TBD)
                implausibility = abs( plausible_candidates['Mean_Estimate'] - self.hm_params[cut]['desired_result'] ) / np.sqrt(plausible_candidates['Var_Predictive'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'] )
                # implausibility_threshold - øuser configured?
                is_implausible = implausibility > self.hm_params[cut]['implausibility_threshold']
                plausible_candidates['implausible'] = is_implausible

        count_plausible_candidates = (not candidates['implausible']).sum()
        break

    return

"""
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
""" # pylint:disable=pointless-string-statement

def cut(self, num_desired_candidates = 5000, constraint = None):
    non_implausible_candidates = pd.DataFrame()
    num_trials = 0

    stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
    stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

    while stats['num_plausible_candidates'] < num_desired_candidates:
        # logger.info('-'*80)
        # max_nSamples = 10000 # TODO: make a parameter or determine from GPU info
        # # Min here to avoid running out of GPU ram!
        # if stats['num_candidates'] == 0:# or stats['num_plausible_candidates'] == 0:
        #     nSamples = min(max_nSamples, num_desired_candidates)
        # else:
        #     nSamples = min(max_nSamples, int(round(1.25 * (num_desired_candidates-stats['num_plausible_candidates']) / ((1+stats['num_plausible_candidates'])/float(stats['num_candidates'])))))

        # lhs_sample = lhs( len(self.Xcols_all_orig), samples=nSamples)
        # lhs sample returns values in range of 0-1, convert to param<sub>min</sub> - param<sub>max</sub>
        # for i, xc in enumerate(self.Xcols_all_orig):
        #     v = self.param_info.loc[xc]
        #     lhs_sample[:, i] = (v['Max'] - v['Min']) * lhs_sample[:, i] + (v['Min'])

        # filter candidates by constraint, if present
        # new_candidates = pd.DataFrame( lhs_sample, columns=self.Xcols_all_orig)
        # if constraint is not None:
        #     #new_candidates = new_candidates.loc[new_candidates.apply(constraint, axis=1),:]
        #     #new_candidates = new_candidates.query(constraint)
        #     new_candidates = new_candidates.loc[constraint(new_candidates),:]

        # test each candidate against existing emulators
        plausibility = self.test_plausibility(new_candidates, constraint)
        new_candidates = new_candidates.merge(plausibility.to_frame(), left_index=True, right_index=True)

        num_trials += new_candidates.shape[0]
        new_non_implausible_candidates = new_candidates.loc[ new_candidates['Implausible'] == False, :] # candidates that are not implausible
        non_implausible_candidates = non_implausible_candidates.append(new_non_implausible_candidates)  # append to existing non-implausible candidates

        stats['num_new_plausible_candidates'] = new_non_implausible_candidates.shape[0] # track _new_ non-implausible candidates (are we making progress?)
        stats['num_plausible_candidates'] = non_implausible_candidates.shape[0]         # track count of non-implausible candidates
        stats['num_candidates'] += num_trials                                           # track total number of candidates considered

        del new_candidates

    # hdf = pd.HDFStore(self.saveto_hd5)
    # hdf.put('values', non_implausible_candidates[self.Xcols_all_orig].reset_index(drop=True))
    # #hdf.put('non_implausible', non_implausible_candidates.set_index(self.Xcols_all_orig))
    # #hdf.put('all', candidates.set_index(self.Xcols_all_orig))
    # hdf.close()

    rejected_percent = 100 * (num_trials-non_implausible_candidates.shape[0]) / float(num_trials)
    stats = {
        'Rejected Percent': rejected_percent,
        'Num Trials': num_trials,
        'Num Implausible': num_trials-non_implausible_candidates.shape[0]
    }

    # (d, filename) = os.path.split(self.saveto_hd5)
    # (name, ext) = os.path.splitext(filename)
    # stats_fn = os.path.join(d, name + '_stats.json')
    # with open(stats_fn, 'w') as f:
    #     json.dump(stats, f)

    # csv_fn = os.path.join(d, name + '.csv')
    # non_implausible_candidates[self.Xcols_all_orig].to_csv(csv_fn, index=False)

    return (non_implausible_candidates, stats)
