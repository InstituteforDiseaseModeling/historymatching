import os
import pandas as pd

class DataSource(object):
    """
    Represents a data directory sitting at the case/iteration/DATA_SOURCE level
    """

    SAMPLES_FILENAME = 'Samples.xlsx'
    RESULTS_FILENAME = 'Results.xlsx'
    SHEET = 'Values'

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.name = os.path.basename(self.directory)
        self.training_fraction = None
        self.use_for_glm = None
        self.use_for_gpr = None
        self.updated_for_use = False # this is set once self.update_for_use() is called

        if os.path.exists(self.samples_filename):
            self.samples_df = pd.read_excel(self.samples_filename, sheetname=self.SHEET)
        else:
            self.samples_df = None

        if os.path.exists(self.results_filename):
            self.results_df = pd.read_excel(self.results_filename, sheetname=self.SHEET)
        else:
            self.results_df = None

    @property
    def samples_filename(self):
        return os.path.join(self.directory, self.SAMPLES_FILENAME)

    @property
    def results_filename(self):
        return os.path.join(self.directory, self.RESULTS_FILENAME)

    @property
    def use_for_training(self):
        if self.training_fraction is None:
            raise Exception('training fraction must be set (0-1, float) for data source: %s' % self.directory)
        return self.training_fraction > 0

    @property
    def use_for_testing(self):
        if self.training_fraction is None:
            raise Exception('training fraction must be set (0-1, float) for data source: %s' % self.directory)
        return self.training_fraction < 1

    def update_for_use(self, training_fraction, use_for_glm, use_for_gpr):
        """
        Simple setter for necessary params before use in BHM fitting
        :param training_fraction, float, 0-1
        :param use_for_glm: True/False
        :param use_for_gpr: True/False
        :return: Nothing
        """
        tf = float(training_fraction)
        if training_fraction is None or tf < 0 or tf > 1:
            raise Exception('training fraction must be set (0-1, float) for data source: %s' % self.directory)
        self.training_fraction = tf

        self.use_for_glm = bool(use_for_glm)

        self.use_for_gpr = bool(use_for_gpr)

        # verify consistency for glm/gpr usage for if training points are present in this data source
        if self.use_for_training and self.use_for_gpr and not self.use_for_glm:
            raise Exception('All data sources used for GPR training must also be used for GLM training.'
                            ' Problem with usage of: %s' % self.directory)

        if not (self.use_for_glm or self.use_for_gpr):
            raise Exception('All data sources must be used for GLM and/or GPR basis generation and/or testing.')
        self.updated_for_use = True

    def n_points(self, step, mode):
        """
        Computes and returns the number of points in this DataSource that are used for different BHM purposes
        ( {GLM,GPR}X{training, testing} )
        :param step: 'glm' or 'gpr'
        :param mode: 'train' or 'test'
        :return: an Integer representing the number of points designated for the given purpose.
        """
        if not self.updated_for_use:
            raise Exception('data source must be updated with info from data sources csv before glm/gpr test/train'
                            'points can be counted.')

        allowed_steps = ['glm', 'gpr']
        if step not in allowed_steps:
            raise Exception('Cannot count points for step: %s . Must be one of: %s .' % (step, allowed_steps))

        allowed_modes = ['train', 'test']
        if mode not in allowed_modes:
            raise Exception('Cannot count points for mode: %s . Must be one of: %s .' % (step, allowed_modes))

        step_map = {'glm': self.use_for_glm, 'gpr': self.use_for_gpr}
        multiplier = self.training_fraction if mode == 'train' else (1 - self.training_fraction)
        n_total_points = len(self.samples_df.index)
        if step_map[step]:
            n_points = int(round(multiplier * n_total_points))
        else:
            n_points = 0
        return n_points

    def n_glm_points(self, mode):
        return self.n_points(step='glm', mode=mode)

    def n_gpr_points(self, mode):
        return self.n_points(step='gpr', mode=mode)

