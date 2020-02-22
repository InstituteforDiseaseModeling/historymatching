import multiprocessing as mp

from pyDOE import lhs
import pandas as pd



class MCMCCutWorker(mp.Process):

    logger = mp.get_logger()

    def __init__(self, x0, N, param_info, constraint):
        super(MCMCCutWorker, self).__init__()

        self.x0 = x0.copy()
        self.N = N
        self.constraint = constraint

        '''
        self.hm_params = copy.deepcopy(hm_params)
        self.Xcols_all_orig = copy.deepcopy(Xcols_all_orig)
        self.cuts = copy.deepcopy(cuts)

        self.param_info = param_info.copy(deep=True)

        # FAILS:
        #self.glm_all = {k: copy.deepcopy(v) for k,v in glm_all.items()}
        #self.gpr_all = {k: copy.deepcopy(v) for k,v in gpr_all.items()}

        self.glm_all = glm_all
        self.gpr_all = gpr_all
        '''

        import os
        from history_matching import HistoryMatching
        from glm import GLM
        from gpr import GPR
        self.iteration = 0
        self.cut_dir = 'Cuts'

        self.hm_params = {}
        self.glm_all = {}
        self.gpr_all = {}
        self.cuts = []

        self.param_info = param_info
        self.Xcols_all_orig = self.param_info.index.unique().values.tolist()
        candidates = pd.DataFrame( columns=self.Xcols_all_orig )

        for it in reversed(range(self.iteration + 1)): # Loop over previous iterations
            cuts_dir = os.path.join('..', 'iter%d'%it, self.cut_dir)

            for cut_name in [name for name in os.listdir(cuts_dir) if os.path.isdir(os.path.join(cuts_dir, name))]:
                print('Reading iteration %d. cut %s' % (it,cut_name) )
                hm = HistoryMatching.from_file(cuts_dir, cut_name)
                print('\t Desired Result:', hm.desired_result)
                print('\t Desired Result Var:', hm.desired_result_var)
                print('\t Discrepancy Var:', hm.discrepancy_var)
                print('\t Imp Thresh:', hm.implausibility_threshold)

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

            print('Performing cut: iteration %d, cut %s' % (it,cut_name) )
            plausible_candidates['Yglm'] = self.glm_all[cut].evaluate(plausible_candidates)
            ret = self.gpr_all[cut].evaluate(plausible_candidates)
            plausible_candidates['Mean_Estimate'] = plausible_candidates['Yglm'] + ret['Mean']
            plausible_candidates['Var_Predictive'] = ret['Var_Predictive']

            plausible_candidates[ f'Implausibility_{it}_{cut_name}' ] = \
                abs( plausible_candidates['Mean_Estimate'] - self.hm_params[cut]['desired_result'] ) / \
                np.sqrt(plausible_candidates['Var_Predictive'] + self.hm_params[cut]['desired_result_var'] + self.hm_params[cut]['discrepancy_var'] )

            plausible_candidates[ f'Implausible_{it}_{cut_name}' ] = plausible_candidates[ f'Implausibility_{it}_{cut_name}' ] > self.hm_params[cut]['implausibility_threshold']
            cols += [f'Implausibility_{it}_{cut_name}', f'Implausible_{it}_{cut_name}']

            new_candidates['Implausible'] |= plausible_candidates[ f'Implausible_{it}_{cut_name}' ]

        return new_candidates['Implausible']


    def run(self):
        self.logger.info( 'x0:\n%s' % self.x0.to_string() )

        candidates = pd.DataFrame()

        stats = {k:{'cut_implausible':0, 'newly_implausible':0, 'num':0} for k in self.cuts}
        stats.update({'num_plausible_candidates':0, 'num_candidates':0, 'num_new_plausible_candidates':0})

        while stats['num_plausible_candidates'] < self.N:
            print('-'*80)
            max_nSamples = 2
            # Min here to avoid running out of GPU ram!
            if stats['num_candidates'] == 0 or stats['num_plausible_candidates'] == 0:
                nSamples = min(max_nSamples, self.N)
            else:
                nSamples = min(max_nSamples, int(round(1.25 * (self.N-stats['num_plausible_candidates']) / (stats['num_plausible_candidates']/float(stats['num_candidates'])))))
            lhs_sample = lhs( len(self.Xcols_all_orig), samples=nSamples)

            for i, xc in enumerate(self.Xcols_all_orig):
                v = self.param_info.loc[xc]
                lhs_sample[:, i] = (v['Max'] - v['Min']) * lhs_sample[:, i] + (v['Min'])

            new_candidates = pd.DataFrame( lhs_sample, columns=self.Xcols_all_orig)
            if self.constraint is not None:
                self.logger.info('About to constrain:\n%s', new_candidates.to_string())
                print(type(new_candidates))
                new_candidates = new_candidates.loc[new_candidates.apply(self.constraint, axis=1),:]

            plausibility = self.test_plausibility(new_candidates, self.constraint)
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

            print('Plausible candidates: New = {0}, Tot = {1}'.format(stats['num_new_plausible_candidates'], stats['num_plausible_candidates']))

        rejected_percent = (100 * sum(candidates['Implausible']) / float(candidates.shape[0]))
        print('Rejected {0:.1f} [{1} / {2}]'.format(rejected_percent, sum(candidates['Implausible']), candidates.shape[0]))

