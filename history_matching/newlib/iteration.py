import os
import re
import pandas as pd
from pyDOE import lhs

from history_matching.newlib.cut import Cut
from history_matching.newlib.data_source import DataSource
from history_matching.newlib.sample_file import SampleFile

class Iteration(object):

    NONE = 'none'

    ITERATION_REGEX = re.compile('^iter(?P<num>\d+)$')
    ITERATION_DIR_PATTERN = 'iter%d'
    CANDIDATES_FILENAME = 'Candidates_for_iteration.xlsx'
    DATA_ROOT = 'Data'

    def __init__(self, directory, create=False, parameters=None):
        """
        :param directory: path of this iteration
        :param parameters: from ParameterFile().parameters
        """
        directory = os.path.abspath(directory)
        self._validate_directory_name(directory)
        self.directory = directory
        self.iteration_number = self._parse_iteration_number(directory)

        
        if not os.path.exists(self.directory):
            if create:
                os.makedirs(self.directory)
            else:
                raise Exception('Cannot open iteration directory %s as it does not exist.')
        
        self.samples = None

        # One route for gathering samples is via a candidates file from another Iteration.
        self.sample_candidates_filename = os.path.join(directory, self.CANDIDATES_FILENAME)

        # load up data directories
        self.data_root = os.path.join(self.directory, self.DATA_ROOT)
        self.data_sources = {}
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        items = os.listdir(self.data_root)
        print('loading dss from dir: %s items: %s' % (self.data_root, items))
        for item in items:
            data_dir = os.path.join(self.data_root, item)
            if os.path.isdir(data_dir):
                ds = DataSource(directory=data_dir)
                self.data_sources[ds.name] = ds

        # we keep it as an optional item here (needed for creating parameter space
        # samples, e.g. in the first iteration)
        self.parameters = parameters

        # load up existing cuts; all are equally valid until a user specifies one on execution
        self.cut_root_directory = os.path.join(self.directory, 'Cuts')
        self.cuts = {}
        if not os.path.exists(self.cut_root_directory):
            os.makedirs(self.cut_root_directory)
            
        items = os.listdir(self.cut_root_directory)
        for item in items:
            if os.path.isdir(item):
                cut_dir = os.path.join(self.cut_root_directory, item)
                cut = Cut(cut_dir)
                self.cuts[cut.name] = cut
    
    @property
    def sample_candidates(self):
        if os.path.exists(self.sample_candidates_filename):
            return SampleFile(self.sample_candidates_filename).samples
        else:
            return None
    
    def set_samples(self, previous_iteration=None, n_samples=None, samples_file = None):
        """
        Selects parameter space points to utilize, first reading a provided samples file, second by
        considering any previous iteration candidates, and lastly by generating the points.
        :param current_iteration: the current iteration being considered. Iteration object.
        :param previous_iteration: candidate sample points will be searched for here. Iteration object.
        :param samples_file: if not reading from candidates of previous_iteration, read this file if specified.
        :param n_samples: the number of parameter space points to generate (if generating)
        :return: Nothing.
        """
        if self.samples:
            raise Exception('Cannot re-set samples.')

        if samples_file:
            # try to read a specified sample file
            self.samples = SampleFile(samples_file).samples
        elif previous_iteration and previous_iteration.sample_candidates is not None:
            # read from the prior iteration's Candidate list
            self.samples = previous_iteration.sample_candidates
        else:
            # generate samples
            if not n_samples:
                raise Exception('Cannot generate samples as n_samples was not specified.')
            self.samples = self._generate_samples(n_samples)

    def write_samples(self, data_source_name):
        if self.samples is None:
            raise Exception('Cannot write samples; they have not been set yet.')
        else:
            if self.data_sources.get(data_source_name, None):
                ds = self.data_sources[data_source_name]
            else:
                data_dir = os.path.join(self.data_root, data_source_name)
                ds = DataSource(directory=data_dir)
                self.data_sources[ds.name] = ds
            SampleFile.write(samples=self.samples, filename=ds.samples_filename)

    def _generate_samples(self, num_samples):
        N_dim = self.parameters.shape[0]
        samples = pd.DataFrame(lhs(N_dim, samples = num_samples), columns=self.parameters.index.tolist())
        
        for param_name in samples.columns.values:
            pmin,pmax = (self.parameters.loc[param_name,'Min'], self.parameters.loc[param_name,'Max'])
            samples[param_name] = pmin + samples[param_name]*(pmax-pmin)
        samples.index.name = 'id'
        return samples

    def make_bases(self, cut_name, inputs, results, remake='none'):
        if not self.cuts.get(cut_name, None):
            cut_dir = os.path.join(self.cut_root_directory, cut_name)
            self.cuts[cut_name] = Cut(cut_dir)
        self.cuts[cut_name].make_bases(param_info=self.parameters, inputs=inputs, results=results, remake=remake)
        return self.cuts[cut_name]

    # This method sets up the inputs and results dicts(?) used as inputs to history matching.
    # This method depends on all data sources having the column specified by 'field'.
    def setup_inputs_and_results(self, data_sources, field):
        sim_inputs = []
        sim_results = []
        # ck4, should use SampleFile for reading
        #all_directories = [training_directory] + data_directories
        #for exp_id in all_directories:
        missing_field = []
        for ds in data_sources:
            print('Reading samples file: %s' % ds.samples_filename)
            read = SampleFile(ds.samples_filename).samples # this essentially returns a dict-like object (pandas.DataFrame)
            print('Read in a type: %s' % type(read))
            print('dict of read item: %s' % dir(read))
            read['Exp_Id'] = ds.name
            #            read['Sample_Id'] = read['Values'].apply(lambda x: '%s.%06d'%(exp_id,x))
            read['Sample_Id'] = read.apply(lambda x: '%s.%06d'%(ds.name,x['id']), axis='columns').values
#            item = read.apply(lambda x: '%s.%06d'%(exp_id,x['id']), axis='columns')
#            print('item type: %s' % type(item))
#            print(item)
#            print(dict(item))
#            print(item.values)

#            read.apply(lambda x: '%s.%06d'%(exp_id,x['id']), axis='columns')
            read = read.set_index('id').sort_index()

            # determine if this DataSource is to be used for GLM and/or GPR basis generation
            read['use_for_glm'] = ds.use_for_glm
            read['use_for_gpr'] = ds.use_for_gpr

            # Train/test split
            if ds.use_for_training:
                read['Train'] = False # sets the default value to False for all rows
                nSamp = len(read.index.get_level_values('id'))
                nTrain = int(round(ds.training_fraction * nSamp))

                # ck4, WEIRD, iloc seems to fail to actually write; seems to edit a transient copy, only
                #read.iloc[0:(nTrain-1)]['Train'] = True # row_indexer,col_indexer]
                # read['Train'][0:nTrain] = True # this yields a 'working on a dataframe copy' warning but does seem to work
                read.loc[0:nTrain, 'Train'] = True
            else:
                read['Train'] = False
            print('nSamp: %d nTrain: %d' % (nSamp, nTrain))
            print('>>>>> appending sim_input:\n%s' % read)
            sim_inputs.append(read)

            read = SampleFile(ds.results_filename).samples
            read['Exp_Id'] = ds.name
            # ck4, change 'Sample' to 'id' ... and update the result write to do so as well
            read['Sample_Id'] = read.apply(lambda x: '%s.%06d'%(ds.name,x['id']), axis='columns').values

            new_result = read.set_index('Sample_Id').sort_index()
            if new_result.get(field, None) is None: # missing!
                missing_field.append(ds.directory)
            sim_results.append(new_result)


        # ck4, this check should ideally be in bhm.py#fit, but there is no reading of Results.xlsx in DataSource currently
        # to check over there.
        if len(missing_field) > 0:
            raise Exception('The following specified data sources are missing the field: %s requested for comparision:\n%s'
                % (field, '\n'.join(missing_field)))

        inputs = pd.concat(sim_inputs)
        sim_results_all = pd.concat(sim_results)
        print('sim_result_all:\nlen: %d\ndata:\n%s' % (len(sim_results_all), sim_results_all))
        sim_results_all.set_index(['Exp_Id', 'id', 'Sim_Id'], append=True, inplace=True)
        results = sim_results_all[field]  # results is a Series

        return inputs, results
        
    # was originally bhm.py
    def fit(self, cut_name, data_sources, field, target, target_std,
            force_optimize_glm=True, force_optimize_gpr=True,
            implausibility_threshold=3, remake_basis='none'):
        
        from history_matching.newlib.HistoryMatching import HistoryMatching
        from history_matching.newlib.quick_read import quick_read
        from history_matching.newlib.basis import Basis
 
        desired_result = target
        discrepancy_std = target_std # 0.1 * desired_result
        print 'Desired result is: ', desired_result

        sim_inputs = []
        sim_results = []

        # ck4, I think use_for_gpr/glm flags should be set on the returned inputs in this method, to carry forward to
        # use in self.make_bases() call
        inputs, results = self.setup_inputs_and_results(data_sources=data_sources, field=field)

        cut = self.make_bases(cut_name=cut_name, inputs=inputs, results=results, remake=remake_basis)

        param_info = self.parameters
        param_names = param_info.index.tolist()
        print 'All available parameters:'
        print ' *','\n * '.join(param_names)

        # History Matching!
        hm = HistoryMatching(
            cut_name = cut_name,
            cut_directory = cut.directory,
            param_info = param_info,
            inputs = inputs,
            field = field,
            results = results,
            desired_result = desired_result,
            iteration = self.iteration_number,
            implausibility_threshold = implausibility_threshold,
            discrepancy_var = discrepancy_std**2,
            training_fraction = None
        )
        hm.save()

        # If desired, you can filter train/test/both data with lower and upper bounds on the result
        #hm.filter_data(source='Both', lower=0)

        ### GLM ###############################################################
        print "="*80, "\nGeneralized Linear Modeling\n", "="*80
        #######################################################################
        hm.glm(
            basis = cut.glm_basis,
            family = 'Gaussian',
            force_optimize_glm = force_optimize_glm,
            glm_fit_maxiter = 100000,
            plot = force_optimize_glm,
            plot_data = False
        )

        ### GPR ###############################################################
        print "="*80, "\nGaussian Process Regression\n", "="*80
        #######################################################################
        hm.gpr(
            basis = cut.gpr_basis,
            force_optimize_gpr = force_optimize_gpr,
            K_folds = 10,
            sigma2_f_guess = 1,
            sigma2_f_bounds = (0.1, 100),
            sigma2_n_guess = 1,
            sigma2_n_bounds = (0.001, 100),
            #lengthscale_guess = [0.04313128, 0.2, 0.14240553, 0.01418867, 0.2, 0.17683428],
            lengthscale_bounds = (0.001, 0.2),
            verbose = True,
            optimizer_options = {
                'eps': 5e-3,
                'disp': True,
                'maxiter': 15000,
                #'ftol': 1e-1,
                #'gtol': 1e-1,
                #'factr': 1e12 # <-- Not working?
            },
            plot = True, #force_optimize_gpr,
            plot_data = False
        )
        
        ### Implausibility ############################################################
        print "="*80, "\nImplausibility\n", "="*80
        ###############################################################################
        hm.calc_and_plot_implausibility(plot=True, do_plot_data=True, plot_data_highlight=pd.DataFrame()) # plot_data_highlight=hm.training_data.loc['8c7e4af7-1120-e711-9400-f0921c16849c.003328']

        directory = self.cuts[cut_name].directory
        hm.glm_training_data.to_excel(os.path.join(directory, 'glm_train_data.xlsx'))
        hm.glm_test_data.to_excel(os.path.join(directory, 'glm_test_data.xlsx'))
        hm.gpr_training_data.to_excel(os.path.join(directory, 'gpr_train_data.xlsx'))
        hm.gpr_test_data.to_excel(os.path.join(directory, 'gpr_test_data.xlsx'))

        print 'Good'

    def get_data_source(self, source):
        """

        :param source: a directory name in the Data directory of an iteration
        :return: a DataSource object
        """
        return self.data_sources[source]

    @classmethod
    def _validate_directory_name(cls, directory):
        # verifies the dir is named properly, e.g. ..../iterN where N >= 0 (int)
        if not cls.ITERATION_REGEX.match(os.path.split(directory)[-1]):
            raise Exception('Invalid iteration directory name: %s' % directory)

    @classmethod
    def _parse_iteration_number(cls, directory):
        cls._validate_directory_name(directory)
        return int(cls.ITERATION_REGEX.match(os.path.split(directory)[-1]).group('num'))

    @classmethod
    def directory_for_number(cls, case_dir, num):
        """
        Constructs a full path canonical iteration directory name.
        :param case_dir: The case to construct the iteration directory name in.
        :param num: The iteration number to construct the directory name with.
        :return: A full path directory name
        """
        return os.path.join(case_dir, cls.ITERATION_DIR_PATTERN % int(num))
    
