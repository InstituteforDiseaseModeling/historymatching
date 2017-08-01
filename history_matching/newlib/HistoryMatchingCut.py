# ck4, should all of this exist as part of Case directly?

import os
from pyDOE import lhs
import pandas as pd
import numpy as np
from newlib.HistoryMatching import HistoryMatching
from glm import GLM
from gpr import GPR

class HistoryMatchingCut(object):

    def __init__(self, case, iteration_number): # cut_dir, iteration):
        """

        :param case: A Case object
        :param iteration_number: an integer (0, 1, 2, ...) of the iteration being considered
        """
        self.case = case

        self.param_info = None
        self.Xcols_all_orig = None

        self.hm_params = {}
        self.glm_all = {}
        self.gpr_all = {}
        self.cuts = []

        for iteration in reversed(case.iterations_up_to(iteration_number)):
#        for it in reversed(range(self.iteration.iteration_number + 1)): # Loop over previous iterations
#            other_iteration = Iteration.in_same_case(source_iteration = self.iteration, num = it)
            cuts_dir = iteration.cut_root_directory   # os.path.join('..', 'iter%d'%it, self.cut_dir)

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                print('Reading iteration %d. cut %s' % (iteration.iteration_number, cut_name))
                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                print '\t Desired Result:', hm.desired_result
                print '\t Desired Result Var:', hm.desired_result_var
                print '\t Discrepancy Var:', hm.discrepancy_var
                print '\t Imp Thresh:', hm.implausibility_threshold

                if self.param_info is None:
                    self.param_info = hm.param_info
                    self.Xcols_all_orig = self.param_info.index.unique().values.tolist()

                self.hm_params[(iteration.iteration_number, cut_name)] = {
                    'desired_result':hm.desired_result,
                    'desired_result_var':hm.desired_result_var,
                    'discrepancy_var':hm.discrepancy_var,
                    'implausibility_threshold':hm.implausibility_threshold,
                }

                self.glm_all[(iteration.iteration_number, cut_name)] = GLM.from_config(os.path.join(cuts_dir, cut_name, 'GLM', 'model.json'), os.path.join(cuts_dir, cut_name, 'GLM', 'params.p'))
                self.gpr_all[(iteration.iteration_number, cut_name)] = GPR.from_config(os.path.join(cuts_dir, cut_name, 'GPR', 'model_with_test_data.json'))
                self.cuts.append((iteration.iteration_number, cut_name))


    def test_plausibility(self, points, constraint = None):
        new_candidates = points.copy()
        new_candidates['Implausible'] = False

        cols = []
        for cut in self.cuts:
            (it, cut_name) = cut

            plausible_candidates = new_candidates.loc[new_candidates['Implausible']==False,:]

            print plausible_candidates.shape
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


    # ck4, decouple the writing to the xlsx from cutting algorithm
    def cut(self, output_filename, num_desired_candidates = 5000, constraint = None):
        candidates = pd.DataFrame()

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        while stats['num_plausible_candidates'] < num_desired_candidates:
            print '-'*80
            max_nSamples = 5000
            # Min here to avoid running out of GPU ram!
            if stats['num_candidates'] == 0 or stats['num_plausible_candidates'] == 0:
                nSamples = min(max_nSamples, num_desired_candidates)
            else:
                nSamples = min(max_nSamples, int(round(1.25 * (num_desired_candidates-stats['num_plausible_candidates']) / (stats['num_plausible_candidates']/float(stats['num_candidates'])))))
            lhs_sample = lhs( len(self.Xcols_all_orig), samples=nSamples)

            for i, xc in enumerate(self.Xcols_all_orig):
                v = self.param_info.loc[xc]
                lhs_sample[:, i] = (v['Max'] - v['Min']) * lhs_sample[:, i] + (v['Min'])

            new_candidates = pd.DataFrame( lhs_sample, columns=self.Xcols_all_orig)
            if constraint is not None:
                new_candidates = new_candidates.loc[new_candidates.apply(constraint, axis=1),:]

            plausibility = self.test_plausibility(new_candidates, constraint)
            new_candidates = new_candidates.merge(plausibility.to_frame(), left_index=True, right_index=True)
            #new_candidates['Implausible'] = False

            '''
            for cut in self.cuts:
                (it, cut_name) = cut

                stats[cut]['cut_implausible'] += new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ].sum()
                stats[cut]['newly_implausible'] += sum(new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ] & ~new_candidates['Implausible'])
                stats[cut]['num'] += new_candidates.shape[0]
                print('--> Iteration %d, cut %s: Implausible=%.1f%%, Newly_Implausible=%.1f%%'%(it, cut_name, 
                    100.*stats[cut]['cut_implausible']/float(stats[cut]['num']),
                    100.*stats[cut]['newly_implausible']/float(stats[cut]['num'])))

                #new_candidates['Implausible'] |= new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ]
            '''

            candidates = candidates.append(new_candidates)
            stats['num_new_plausible_candidates'] = sum(new_candidates['Implausible'] == False)
            stats['num_plausible_candidates'] += stats['num_new_plausible_candidates']
            stats['num_candidates'] += new_candidates.shape[0]

            del new_candidates

            print 'Plausible candidates: New = %d, Tot = %d' % (stats['num_new_plausible_candidates'], stats['num_plausible_candidates'])

        rejected_percent = (100 * sum(candidates['Implausible']) / float(candidates.shape[0]))
        print 'Rejected %.1f%% [%d / %d]' % (rejected_percent, sum(candidates['Implausible']), candidates.shape[0])

        non_implausible_candidates = candidates.loc[ candidates['Implausible'] == False, :]

        hdf_filename = os.path.splitext(output_filename)[0] + '.hd5'
        hdf = pd.HDFStore(hdf_filename)
        hdf.put('values', non_implausible_candidates[self.Xcols_all_orig].reset_index(drop=True))
        hdf.put('non_implausible', non_implausible_candidates.set_index(self.Xcols_all_orig))
        hdf.put('all', candidates.set_index(self.Xcols_all_orig))
        hdf.close()

        xlsx_filename = output_filename
        writer = pd.ExcelWriter(xlsx_filename)
        non_implausible_candidates[self.Xcols_all_orig].to_excel(writer, sheet_name='Values', index=False)
        non_implausible_candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='NonImplausible')
        candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='All')
        writer.save()

        return (candidates, rejected_percent)
