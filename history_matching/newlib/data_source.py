import os

class DataSource(object):
    """
    Represents a data directory sitting at the case/iteration/DATA_SOURCE level
    """

    SAMPLES_FILENAME = 'Samples.xlsx'
    RESULTS_FILENAME = 'Results.xlsx'

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        self.name = os.path.basename(self.directory)
        self.training_fraction = None
        self.use_for_glm = None
        self.use_for_gpr = None

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

        if type(use_for_glm) is not bool:
            raise Exception('use_for_glm must be set (True/False) for data source: %s' % self.directory)
        self.use_for_glm = use_for_glm

        if type(use_for_gpr) is not bool:
            raise Exception('use_for_gpr must be set (True/False) for data source: %s' % self.directory)
        self.use_for_gpr = use_for_gpr

        # verify consistency for glm/gpr usage
        # ck4, verify with Dan that this is correct. Should constraints be held to training_data only?
        if self.use_for_gpr and not self.use_for_glm:
            raise Exception('All data sources used for GPR basis generation must also be used for GLM basis generation.'
                            ' Problem with usage of: %s' % self.directory)
