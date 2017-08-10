import os
import re
import pandas as pd
from pyDOE import lhs

from newlib.cut import Cut
from newlib.data_source import DataSource
from newlib.sample_file import SampleFile # ck4, all pathing for hm package needs to be fixed

class Iteration(object):

    NONE = 'none'

    ITERATION_REGEX = re.compile('^iter(?P<num>\d+)$')
    ITERATION_DIR_PATTERN = 'iter%d'
    CANDIDATES_FILENAME = 'Candidates_for_iteration.xlsx'
    DATA_ROOT = 'Data'
    SAMPLES_FILENAME = 'Samples.xlsx'

    def __init__(self, directory, parameters=None):
        """
        :param directory: path of this iteration
        :param parameters: from ParameterFile().parameters
        """
        directory = os.path.abspath(directory)
        self._validate_directory_name(directory)
        self.directory = directory
        self.iteration_number = self._parse_iteration_number(directory)

        self.samples = None

        # One route for gathering samples is via a candidates file from another Iteration.
        self.sample_candidates_filename = os.path.join(directory, self.CANDIDATES_FILENAME)

#        # The canonical path where samples will be stored for this iteration, once discovered.
        self.samples_file = os.path.join(self.directory, self.SAMPLES_FILENAME)

        # load up data directories
        self.data_root = os.path.join(self.directory, self.DATA_ROOT)
        self.data_sources = {}
        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)
        items = os.listdir(self.data_root)
        for item in items:
            if os.path.isdir(item):
                data_dir = os.path.join(self.data_root, item)
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

    def write_samples(self):
        if self.samples is None:
            raise Exception('Cannot write samples; they have not been set yet.')
        else:
            SampleFile.write(samples=self.samples, filename=self.samples_file)

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
    def setup_inputs_and_results(self, data_sources): #training_directory, data_directories, training_fraction):
        sim_inputs = []
        sim_results = []
        # ck4, exp_id and ds.name are the same; refactor...
        # ck4, should use SampleFile for reading
        #all_directories = [training_directory] + data_directories
        #for exp_id in all_directories:
        for ds in data_sources:
            print('Reading samples file: %s' % ds.samples_filename)
            read = SampleFile(ds.samples_filename).samples # ck4, this essentially returns a dict-like object (pandas.DataFrame)
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
            #if exp_id == training_directory:
                read['Train'] = False # sets the default value to False for all rows
#                nSamp = len(read.index.get_level_values('Sample_Id'))
                nSamp = len(read.index.get_level_values('id'))
                print('nsamp: %s' % nSamp)
                nTrain = int(round(training_fraction * nSamp))
                read.iloc[:nTrain-1]['Train'] = True
            else:
                # ck4, was originally '= true'. Ask Dan, is this right??? Shouldn't this be False?
                read['Train'] = False

            sim_inputs.append(read)

            read = SampleFile(ds.results_filename).samples
            read['Exp_Id'] = ds.name
            # ck4, change 'Sample' to 'id' ... and update the result write to do so as well
            read['Sample_Id'] = read.apply(lambda x: '%s.%06d'%(ds.name,x['id']), axis='columns').values
            sim_results.append(read.set_index('Sample_Id').sort_index())

        inputs = pd.concat(sim_inputs)
        sim_results_all = pd.concat(sim_results)
        sim_results_all.set_index(['Exp_Id', 'id', 'Sim_Id'], append=True, inplace=True)
        results = sim_results_all['Sim_Result']
        
        return inputs, results
        
    # was originally bhm.py
    # ck4, this method needs to be updated to detect/use info regarding which data_sources to use for GLM and which to
    # use for GPR
    def fit(self, cut_name, data_sources, target, target_std,
            force_optimize_glm=True, force_optimize_gpr=True,
            implausibility_threshold=3, remake_basis='none'):
        
        from newlib.HistoryMatching import HistoryMatching
        from newlib.quick_read import quick_read
        from newlib.basis import Basis
 
        desired_result = target
        discrepancy_std = target_std # 0.1 * desired_result
        print 'Desired result is: ', desired_result

        sim_inputs = []
        sim_results = []

        # ck4, I think use_for_gpr/glm flags should be set on the returned inputs in this method, to carry forward to
        # use in self.make_bases() call
        inputs, results = self.setup_inputs_and_results(data_sources=data_sources)

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
            results = results,
            desired_result = desired_result,
            iteration = self.iteration_number,
            implausibility_threshold = implausibility_threshold,
            discrepancy_var = discrepancy_std**2,
            training_fraction = None # ck4, ok? We are using a input csv to do line-by line specification
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
        hm.training_data.to_excel(os.path.join(directory, 'train_data.xlsx'))
        hm.test_data.to_excel(os.path.join(directory, 'test_data.xlsx'))
        
        print 'Good'

    def get_data_source(self, source):
        """

        :param source: a directory name in the Data directory of an iteration
        :return: a DataSource object
        """
        return self.data_sources[source]
        #
        # data_sources = [ds for ds in self.data_sources if ds.name == source] # ck4, define
        # if len(data_sources) != 1:
        #     raise Exception('Could not determine which data_source to use for %s. There are %d possibilities.' %
        #                     (source, len(data_sources)))
        # return data_sources[0]

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
    
