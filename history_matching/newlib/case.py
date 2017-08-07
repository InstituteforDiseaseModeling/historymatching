import os

from newlib.HistoryMatchingCut import HistoryMatchingCut  # ck4, fix all newlib import statements eventually
from newlib.iteration import Iteration
from newlib.parameter_file import ParameterFile

class Case(object):
    """
    Represents a set of history matching iterations + in-common information and parameters
    """

    PARAMETERS_FILENAME = 'Params.xlsx'

    def __init__(self, case_directory):
        self.directory = case_directory
        self.parameter_filename = os.path.join(self.directory, self.PARAMETERS_FILENAME)
        self.parameters = self._load_parameters_file()
        self.iterations = self._load_iterations()

    def _load_parameters_file(self):
        if not os.path.exists(self.parameter_filename): # ck4, is it possible to not have a parameters file??
            # ... first iteration (iter0)? I don't remember for sure.
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
        if len(iterations) != 1:
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
