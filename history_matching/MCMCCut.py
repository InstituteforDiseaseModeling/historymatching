import multiprocessing as mp
import os
import random

from history_matching import quick_read
from history_matching.MCMCCutWorker import MCMCCutWorker
import logging



class MCMCCut():

    def __init__(self, cut_dir, iteration, samples_fn):
        self.cut_dir = cut_dir
        self.iteration = iteration

        self.param_info = None
        self.Xcols_all_orig = None

        # Read in samples to choose initial points for the chains
        self.samples = quick_read(samples_fn, sheetname='Samples')

        self.param_info = quick_read( os.path.join('..', 'Params.xlsx'), sheetname='Params').set_index('Name')


    def cut(self, num_workers, num_desired_candidates, constraint):
        mp.log_to_stderr()
        logger = mp.get_logger()
        logger.setLevel(logging.INFO)

        rand_rows = [1,1]
        while len(rand_rows) > len(set(rand_rows)):
            rand_rows = random.sample(range(self.samples.shape[0]), num_workers)

        jobs = []
        for row in rand_rows:
            x0 = self.samples.loc[row,:]
            p = MCMCCutWorker(x0, 1+num_desired_candidates/num_workers, self.param_info, constraint)
            jobs.append(p)
            p.start()

        for j in jobs:
            j.join()
