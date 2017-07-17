import os
import re
import pandas as pd
from pyDOE import lhs

from newlib.cut import Cut
from newlib.sample_file import SampleFile # ck4, all pathing for hm package needs to be fixed
from newlib.parameter_file import ParameterFile

class Iteration(object):

    NONE = 'none'
    SAMPLES_FILENAME = 'Samples.xlsx'
    RESULTS_FILENAME = 'Results.xlsx'
    ITERATION_REGEX = re.compile('^iter(?P<num>\d+)$')
    ITERATION_DIR_PATTERN = 'iter%d'
    CANDIDATES_FILENAME = 'Candidates_for_iteration.xlsx'

    def __init__(self, directory, parameter_filename=None):
        directory = os.path.abspath(directory)
        self._validate_directory_name(directory)
        self.directory = directory
        self.iteration_number = self._parse_iteration_number(directory)
        self.case_directory = os.path.split(directory)[0] # ck4, really should be in higher up object, but...
        
        self.samples = None

        # One route for gathering samples is via a candidates file from another Iteration.
        self.sample_candidates_filename = os.path.join(directory, self.CANDIDATES_FILENAME)
        
        # The canonical path where samples will be stored for this iteration, once discovered.
        self.samples_file = os.path.join(self.directory, self.SAMPLES_FILENAME)

        # Not ideal, but the parameter file exists a level above iterations. Until code
        # is written to represent that higher level of the history matching process,
        # we keep it as an optional item here (needed for creating parameter space
        # samples, e.g. in the first iteration)
        self.parameter_filename = parameter_filename
        if parameter_filename:
            self.parameters = ParameterFile(parameter_filename).parameters
        else:
            self.parameters = None

        self.cut_root_directory = os.path.join(directory, 'Cuts')

        # load up existing cuts; all are equally valid until a user specifies one on execution
        self.cuts = {}
        if not os.path.exists(self.cut_root_directory):
            os.makedirs(self.cut_root_directory)
            
        items = os.listdir(self.cut_root_directory)
        for item in items:
            if os.path.isdir(item):
                cut_dir = os.path.join(self.cut_root_directory, item)
                self.cuts[item] = Cut(cut_dir)
    
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

    def cut_param_space(self, n_desired_candidates, constraint=None):
        from newlib.HistoryMatchingCut import HistoryMatchingCut # ck4, fix all newlib import statements eventually
        hm = HistoryMatchingCut(iteration = self) # ck4, pretty funky passing only self; probably means HMC.cut needs to be a method on Iteration objects. Some day.
        
        print "="*80, "\nCut\n", "="*80
        (_, rejected_percent) = hm.cut(output_filename = self.sample_candidates_filename,
                                       num_desired_candidates = n_desired_candidates,
                                       constraint = constraint)
        # ck4, move printing from hm.cut to here (of cut result)
        # ck4, move writing of candidates xlsx file to a hm.write_... call here.

    def make_bases(self, cut_name, inputs, results, force = False):
        if not self.cuts.get(cut_name, None):
            cut_dir = os.path.join(self.cut_root_directory, cut_name)
            self.cuts[cut_name] = Cut(cut_dir)
        self.cuts[cut_name].make_bases(param_info=self.parameters, inputs=inputs, results=results, force=force)

    # This method sets up the inputs and results dicts(?) used as inputs to history matching.
    def setup_inputs_and_results(self, training_directory, data_directories, training_fraction):
        sim_inputs = []
        sim_results = []
        # ck4, should use SampleFile for reading
        all_directories = [training_directory] + data_directories
        for idx, exp_id in enumerate(all_directories):
            samples_filename = os.path.join(exp_id, self.SAMPLES_FILENAME)
            print('Reading samples file: %s' % samples_filename)
            read = SampleFile(samples_filename).samples # quick_read(os.path.join(exp_id, samples_fn), 'Values')
            print('Read in a type: %s' % type(read))
            print('dict of read item: %s' % dir(read))
            read['Exp_Id'] = exp_id
            #            read['Sample_Id'] = read['Values'].apply(lambda x: '%s.%06d'%(exp_id,x))
            read['Sample_Id'] = read.apply(lambda x: '%s.%06d'%(exp_id,x['id']), axis='columns').values
#            item = read.apply(lambda x: '%s.%06d'%(exp_id,x['id']), axis='columns')
#            print('item type: %s' % type(item))
#            print(item)
#            print(dict(item))
#            print(item.values)

#            read.apply(lambda x: '%s.%06d'%(exp_id,x['id']), axis='columns')
            read = read.set_index('id').sort_index()

            # Train/test split
            if exp_id == training_directory:
                read['Train'] = False
#                nSamp = len(read.index.get_level_values('Sample_Id'))
                nSamp = len(read.index.get_level_values('id'))
                print('nsamp: %s' % nSamp)
                nTrain = int(round(training_fraction * nSamp))
                read.iloc[:nTrain-1]['Train'] = True
            else:
                read['Train'] = True

            sim_inputs.append(read)

#            read = quick_read(os.path.join(exp_id, self.RESULTS_FILENAME), 'Sheet1') # ck4, change 'Sheet1' to 'Values' ... and update the result write to do so as well
#            read['Exp_Id'] = exp_id
#            read['Sample_Id'] = read['Sample'].apply(lambda x: '%s.%06d'%(exp_id,x)) # ck4, change 'Sample' to 'id' ... and update the result write to do so as well
#            sim_results.append(read.set_index('Sample_Id').sort_index())

#            read = quick_read(), 'Sheet1') # ck4, change 'Sheet1' to 'Values' ... and update the result write to do so as well
            result_filename = os.path.join(exp_id, self.RESULTS_FILENAME)
            read = SampleFile(result_filename).samples
            read['Exp_Id'] = exp_id
            read['Sample_Id'] = read.apply(lambda x: '%s.%06d'%(exp_id,x['id']), axis='columns').values # ck4, change 'Sample' to 'id' ... and update the result write to do so as well
            sim_results.append(read.set_index('Sample_Id').sort_index())

        inputs = pd.concat(sim_inputs)
        sim_results_all = pd.concat(sim_results)
        sim_results_all.set_index(['Exp_Id', 'id', 'Sim_Id'], append=True, inplace=True)
        results = sim_results_all['Sim_Result']
        
        return inputs, results
        
    # was originally bhm.py
    def fit(self, cut_name, training_directory, data_directories,
            target, target_std,
            training_fraction=0.75, force_optimize_glm=True, force_optimize_gpr=True,
            implausibility_threshold=3, remake_bases=False):
        
        from newlib.HistoryMatching import HistoryMatching
        from newlib.quick_read import quick_read
        from newlib.basis import Basis
 
        desired_result = target
        discrepancy_std = target_std # 0.1 * desired_result
        print 'Desired result is: ', desired_result

        # Data
        samples_fn = 'Samples.xlsx' # ck4, move this to a data dir class?
        results_fn = 'Results.xlsx'

        sim_inputs = []
        sim_results = []        

        inputs, results = self.setup_inputs_and_results(training_directory,
                                                        data_directories,
                                                        training_fraction)

        self.make_bases(cut_name=cut_name, force=remake_bases, inputs=inputs, results=results)
        cut = self.cuts[cut_name] # set in self.make_bases()
        
        param_info = self.parameters
        param_names = param_info.index.tolist()
        print 'All available parameters:'
        print ' *','\n * '.join(param_names)

        # History Matching!
        hm = HistoryMatching(
            cut_name = cut_name,
            param_info = param_info,
            inputs = inputs,
            results = results,
            desired_result = desired_result,
            iteration = self.iteration_number,
            implausibility_threshold = implausibility_threshold,
            discrepancy_var = discrepancy_std**2,
            training_fraction = training_fraction
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
        
        hm.training_data.to_excel(os.path.join('Cuts', cut_name, 'train_data.xlsx'))
        hm.test_data.to_excel(os.path.join('Cuts', cut_name, 'test_data.xlsx'))
        
        print 'Good'


    

        
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
    def in_same_case(cls, source_iteration, num):
        """
        Creates an Iteration object for the given iteration number using the same
        case directory as source_iteration
        :return: an Iteration object
        """
        return Iteration(directory = cls.directory_for_number(source_iteration.case_directory, num),
                         parameter_filename = source_iteration.parameter_filename)
    
    @classmethod
    def directory_for_number(cls, case_dir, num):
        """
        Constructs a full path canonical iteration directory name.
        :param case_dir: The case to construct the iteration directory name in.
        :param num: The iteration number to construct the directory name with.
        :return: A full path directory name
        """
        return os.path.join(case_dir, cls.ITERATION_DIR_PATTERN % int(num))
    
