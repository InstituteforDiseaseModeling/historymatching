"""
Contains custom errors for History Matching
"""

class HMNotAnEmulator(Exception):
    def __init__(self, obs_name, wave=None):
        self.obs_name = obs_name
        if wave is None:
            super().__init__(f"{obs_name} was not associated with a valid emulator!")
        else :
            super().__init__(f"{obs_name} from Wave {wave} was not associated with a valid emulator!")

class HMNotADataFrame(Exception):
    def __init__(self, df_name):
        self.df_name = df_name
        super().__init__(f"{df_name} was not a DataFrame!")

class HMObservationIDsNotUnique(Exception):
    def __init__(self, df_name):
        self.df_name = df_name
        super().__init__(f"{df_name} had non-unique observations ids!")

class HMTwoObservationsAtOneTime(Exception):
    def __init__(self, df_name):
        self.df_name = df_name
        super().__init__(f"{df_name} contained the same observation made twice or more at the same time!")

class HMTimeIsNotMonotonic(Exception):
    def __init__(self, df_name):
        self.df_name = df_name
        super().__init__(f"{df_name}'s 'time' column was not monotonic!")

class HMExtraColumns(Exception):
    """Used to indicate that a dataframe has extra, unexpected columns"""
    def __init__(self, df_name):
        self.df_name = df_name
        super().__init__(f"{df_name} has unexpected columns!")

class HMMissingColumn(Exception):
    """Used to indicate that a dataframe is missing a column"""
    def __init__(self, df_name, col_name):
        self.df_name = df_name
        self.missing_column = col_name
        super().__init__(f"{df_name} is missing its '{col_name}' column!")

class HMMaxLessThanMin(Exception):
    """Used to indicate that a dataframe's max is below its min"""
    def __init__(self, df_name):
        self.df_name = df_name
        super().__init__(f"{df_name} has a max value smaller than a min value!")

class HMParameterSamplesEmpty(Exception):
    def __init__(self):
        super().__init__("ParameterSamplesFrame was empty! Cannot continue.")

class HMWrongColumnsInFrame(Exception):
    """Used to indicate that the wrong columns have been provided"""

class HistoryMatchingError(Exception):
    """The custom error used for everything not covered above"""
