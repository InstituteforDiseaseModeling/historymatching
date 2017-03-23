import os
from pyDOE import lhs
import pandas as pd
import numpy as np
from history_matching import HistoryMatching
from glm import GLM
from gpr import GPR

class HistoryMatchingCut():

    def __init__(self, cut_dir, iteration):
        self.cut_dir = cut_dir
        self.iteration = iteration

        self.param_info = None
        #self.Xcols_all = None
        self.Xcols_all_orig = None

        self.hm_params = {}
        self.glm_all = {}
        self.gpr_all = {}
        self.cuts = []

        for it in range(self.iteration + 1): # Loop over previous iterations
            cuts_dir = os.path.join('..', 'iter%d'%it, 'Cuts')

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                print('Reading iteration %d. cut %s' % (it,cut_name) )
                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                print '\t Discrepancy Var:', hm.discrepancy_var
                print '\t Desired Result:', hm.desired_result
                print '\t Imp Thresh:', hm.implausibility_threshold

                if self.param_info is None:
                    self.param_info = hm.param_info
                    # TODO: Get both orig and modified parameter names
                    self.Xcols_all_orig = self.param_info.index.unique().values.tolist()
                    #self.Xcols_all_orig = hm.inputs.columns.values.tolist()
                    #print self.Xcols_all
                    #print type(self.Xcols_all)
                    print self.Xcols_all_orig
                    print type(self.Xcols_all_orig)
                    candidates = pd.DataFrame( columns=self.Xcols_all_orig )

                self.hm_params[(it, cut_name)] = {
                    'desired_result':hm.desired_result,
                    'discrepancy_var':hm.discrepancy_var,
                    'implausibility_threshold':hm.implausibility_threshold,
                }

                self.glm_all[(it, cut_name)] = GLM.from_config(os.path.join(cuts_dir, cut_name, 'GLM', 'model.json'), os.path.join(cuts_dir, cut_name, 'GLM', 'params.p'))
                self.gpr_all[(it, cut_name)] = GPR.from_config(os.path.join(cuts_dir, cut_name, 'GPR', 'model_with_test_data.json'))
                self.cuts.append((it, cut_name))

        self.cuts.sort()


    def test_plausibility(self, points, constraint = None):
        new_candidates = points.copy()
        new_candidates['Implausible'] = False

        cols = ['Implausible']
        for cut in self.cuts:
            (it, cut_name) = cut
            print('Performing cut: iteration %d, cut %s' % (it,cut_name) )
            new_candidates['Yglm'] = self.glm_all[cut].evaluate(new_candidates)
            ret = self.gpr_all[cut].evaluate(new_candidates)
            new_candidates['Mean_Estimate'] = new_candidates['Yglm'] + ret['Mean']
            new_candidates['Var_Predictive'] = ret['Var_Predictive']

            new_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] = \
                abs( new_candidates['Mean_Estimate'] - self.hm_params[cut]['desired_result'] ) / \
                np.sqrt(new_candidates['Var_Predictive'] + self.hm_params[cut]['discrepancy_var'] )

            new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ] = new_candidates[ 'Implausibility_%d_%s'%(it, cut_name) ] > self.hm_params[cut]['implausibility_threshold']
            cols += ['Implausibility_%d_%s'%(it, cut_name), 'Implausible_%d_%s'%(it, cut_name)]

            new_candidates['Implausible'] |= new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ]
        return new_candidates[cols]


    def cut(self, num_desired_candidates = 5000, constraint = None):

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        while stats['num_plausible_candidates'] < num_desired_candidates:
            print '-'*80
            # Min here to avoid running out of GPU ram!
            if stats['num_candidates'] == 0 or stats['num_plausible_candidates'] == 0:
                nSamples = min(2500, num_desired_candidates)
            else:
                nSamples = min(2500, int(round(1.25 * (num_desired_candidates-stats['num_plausible_candidates']) / (stats['num_plausible_candidates']/float(stats['num_candidates'])))))
            lhs_sample = lhs( len(self.Xcols_all_orig), samples=nSamples)

            for i, xc in enumerate(self.Xcols_all_orig):
                v = self.param_info.loc[xc]
                lhs_sample[:, i] = (v['Max'] - v['Min']) * lhs_sample[:, i] + (v['Min'])

            new_candidates = pd.DataFrame( lhs_sample, columns=self.Xcols_all_orig)
            if constraint is not None:
                new_candidates = new_candidates.loc[new_candidates.apply(constraint, axis=1),:]

            plausibility = self.test_plausibility(new_candidates, constraint)

            print plausibility.head()

            new_candidates = new_candidates.merge(plausibility, left_index=True, right_index=True)

            print new_candidates.head()
            exit()

            for cut in self.cuts:
                (it, cut_name) = cut

                stats[cut]['cut_implausible'] += new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ].sum()
                stats[cut]['newly_implausible'] += sum(new_candidates[ 'Implausible_%d_%s'%(it, cut_name) ] & ~new_candidates['Implausible'])
                stats[cut]['num'] += new_candidates.shape[0]
                print('--> Iteration %d, cut %s: Implausible=%.1f%%, Newly_Implausible=%.1f%%'%(it, cut_name, 
                    100.*stats[cut]['cut_implausible']/float(stats[cut]['num']),
                    100.*stats[cut]['newly_implausible']/float(stats[cut]['num'])))

            candidates = candidates.append(new_candidates)
            stats['num_new_plausible_candidates'] = sum(new_candidates['Implausible'] == False)
            stats['num_plausible_candidates'] += stats['num_new_plausible_candidates']
            stats['num_candidates'] += new_candidates.shape[0]

            del new_candidates

            print 'Plausible candidates: New = %d, Tot = %d' % (stats['num_new_plausible_candidates'], stats['num_plausible_candidates'])

        rejected_percent = (100 * sum(candidates['Implausible']) / float(candidates.shape[0]))
        print 'Rejected %.1f%%' % rejected_percent

        non_implausible_candidates = candidates.loc[ candidates['Implausible'] == False, :]
        writer = pd.ExcelWriter('Candidates_for_iter%d.xlsx'%(self.iteration+1))
        non_implausible_candidates[self.Xcols_all_orig].to_excel(writer, sheet_name='Values', index=False)

        non_implausible_candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='NonImplausible')
        candidates.set_index(self.Xcols_all_orig).to_excel(writer, sheet_name='All')
        writer.save()

        return (candidates, rejected_percent)
