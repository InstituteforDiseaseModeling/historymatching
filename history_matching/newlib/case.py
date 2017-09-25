import os
import pandas

from history_matching.newlib.HistoryMatchingCut import HistoryMatchingCut  # ck4, fix all newlib import statements eventually
from history_matching.newlib.iteration import Iteration
from history_matching.newlib.parameter_file import ParameterFile
from history_matching.newlib.ReferenceData import ReferenceData

class NoReferenceData(Exception): pass

class Case(object):
    """
    Represents a set of history matching iterations + in-common information and parameters
    """

    PARAMETERS_FILENAME = 'Params.xlsx'
    DATA_SOURCE_CSV_REQUIRED_KEYS = ['iteration_number', 'data_source', 'training_fraction', 'GLM', 'GPR']
    REFERENCE_DATA_FILE = 'reference.csv'

    def __init__(self, case_directory):
        self.directory = case_directory
        self.parameter_filename = os.path.join(self.directory, self.PARAMETERS_FILENAME)
        self.parameters = self._load_parameters_file()
        self.iterations = self._load_iterations()
        self.reference_data = self._load_reference_data()

    def _load_reference_data(self):
        filename = os.path.join(self.directory, self.REFERENCE_DATA_FILE)
        if not os.path.exists(filename):
            raise NoReferenceData('No reference.csv file exists at: %s' % filename)
        return ReferenceData(filename=filename)

    def _load_parameters_file(self):
        if not os.path.exists(self.parameter_filename):
            raise Exception('Invalid case. No parameters file at: %s' % self.parameter_filename)
        return ParameterFile(self.parameter_filename).parameters

    def _load_iterations(self):
        """
        Discover, load, and return iterations (as Iteration) in this Case.
        :return: a list of discovered Iteration objects.
        """
        candidates = os.listdir(self.directory)
        iterations = []
        for item in candidates:
            if Iteration.ITERATION_REGEX.match(item):
                iteration_directory = os.path.join(self.directory, item)
                iteration = Iteration(directory=iteration_directory, parameters=self.parameters)
                iterations.append(iteration)
        return iterations

    def get_iteration(self, iteration_number):
        """
        Accessor to get Iteration of the given number
        :param iteration_number: we want this one
        :return: an Iteration object
        """
        iterations = [ iteration for iteration in self.iterations if iteration.iteration_number == iteration_number]
        if len(iterations) == 0:
            iteration_directory = Iteration.directory_for_number(case_dir=self.directory, num=iteration_number)
            iteration = Iteration(directory=iteration_directory, create=True, parameters=self.parameters)
            self.iterations.append(iteration)
            iterations = [iteration]
        elif len(iterations) != 1:
            raise Exception('Could not determine which iteration is number %d . There are %d possibilities.' %
                            (iteration_number, len(iterations)))
        return iterations[0]

    def get_previous_iteration(self, iteration_number):
        """
        Obtain the Iteration that immediately precedes the specified iteration
        :param iteration_number: an int corresponding to the iteration we want the preceding iteration
        :return: an Iteration object if a preceding Iteration exists, or None if not possible (iteration_number is 0)
        """
        if iteration_number < 0:
            raise Exception('Invalid iteration number %d . Must be >= 0 .' % iteration_number)
        elif iteration_number == 0:
            previous_iteration = None
        else:
            previous_iteration = self.get_iteration(iteration_number=(iteration_number-1))
        return previous_iteration

    def iterations_up_to(self, iteration_number):
        """
        Accessor for getting all Iterations up to and including iteration_number.
        :param iteration_number: The highest numbered iteration to be returned
        :return: a list of Iteration objects in ascending iteration number order
        """
        iterations = []
        for iteration in self.iterations:
            if iteration.iteration_number <= iteration_number:
                iterations.append(iteration)
        return sorted(iterations, cmp = lambda x,y: cmp(x.iteration_number, y.iteration_number))

    def cut_param_space(self, iteration_number, n_desired_candidates, constraint=None): # ck4, fix up calls to this
        hm = HistoryMatchingCut(self, iteration_number)

        print "=" * 80, "\nCut\n", "=" * 80
        (_, rejected_percent) = hm.cut(output_filename=self.get_iteration(iteration_number).sample_candidates_filename,
                                       num_desired_candidates=n_desired_candidates,
                                       constraint=constraint)
        # ck4, move printing from hm.cut to here (of cut result)
        # ck4, move writing of candidates xlsx file to a hm.write_... call here.

    # ck4, move the load csv into DataSource class? (along with DATA_SOURCE_CSV_REQUIRED_KEYS). Keep
    # 'set of DataSources' checks here.
    def load_data_sources_csv(self, filename):
        data = pandas.read_csv(filename)
        file_keys = data.keys()

        if not sorted(self.DATA_SOURCE_CSV_REQUIRED_KEYS) == sorted(file_keys):
            raise Exception('data source csv columns must be: %s' % self.DATA_SOURCE_CSV_REQUIRED_KEYS)

        data_sources = []
        for row_index in range(len(data)):
            iteration_number = int(data.iteration_number[row_index])
            data_source_name = data.data_source[row_index]
            iteration = self.get_iteration(iteration_number=iteration_number)
            ds = iteration.get_data_source(source=data_source_name)
            ds.update_for_use(training_fraction=float(data.training_fraction[row_index]),
                              use_for_glm=data.GLM[row_index],
                              use_for_gpr=data.GPR[row_index])
            data_sources.append(ds)

        # do initial checking of the data sources to make sure there are test and training points
        # for glm and gpr.
        n_points = {
            'glm': {
                'train': 0,
                'test': 0
            },
            'gpr': {
                'train': 0,
                'test': 0
            }
        }
        for ds in data_sources:
            for step, modes_hash in n_points.iteritems():
                for mode in modes_hash.keys():
                    n_points[step][mode] += ds.n_points(step=step, mode=mode)

        empty_cases = []
        for step, modes_hash in n_points.iteritems():
            for mode in modes_hash.keys():
                print('n_points to use: %s %s : %d' % (step, mode, n_points[step][mode]))
                if n_points[step][mode] == 0:
                    empty_cases.append('%s:%s' % (step, mode))
        if len(empty_cases) != 0:
            raise Exception('No points were specified in data sources csv for the following cases: %s' % ' '.join(empty_cases))

        return data_sources
