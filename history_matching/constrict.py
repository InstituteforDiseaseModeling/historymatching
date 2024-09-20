# constrict.py
import time
from typing import Dict
from typing import Tuple

import numpy as np
import pandas as pd

from .config import Config
from .emulators import BaseEmulator
from .samplers import lhs

import logging
logger = logging.getLogger()


_tictimes = []


def tic() -> int:
    """Start a timer."""
    t = time.time_ns()
    _tictimes.append(t)
    return t


def toc(msg: str = "", dopop: bool = True) -> int:
    """Stop a timer and print the elapsed time."""
    t = time.time_ns()
    s = _tictimes.pop() if dopop else _tictimes[-1]
    elapsed = t - s
    print(f"{msg if msg else 'Elapsed: '} {elapsed} ns")
    return elapsed


def next_point_generation( parameter_space: pd.DataFrame,
                           observations   : pd.DataFrame,
                           emulator_bank  : Dict[int, Dict[str, BaseEmulator]],
                           config: Config,
                          ) -> Tuple[pd.DataFrame, float]:
    """Next Point Generation based on existing emulators and observations."""

    max_n_samples = 1_000  # TODO - add to configuration?
    max_candidates = 500_000  # ditto

    num_desired_candidates = config.n_candidates
    non_implausible_candidates = pd.DataFrame()
    num_candidates_considered = 0
    num_non_implausible_candidates = 0
    #while (num_non_implausible_candidates := len(non_implausible_candidates)) < num_desired_candidates:
    while num_non_implausible_candidates < num_desired_candidates:

        # Get the number of samples to generate
        if num_candidates_considered == 0:
            n_samples = num_desired_candidates
        elif num_non_implausible_candidates > 0:  # Generate a few more candidates based on the rejection rate
            n_samples = int( 1.25 * (num_desired_candidates - num_non_implausible_candidates)     \
                                  * num_candidates_considered     \
                                  / num_non_implausible_candidates
                            )
        else:   # Generate a few more candidates since rejection seems to be high
            n_samples = int( 1.25 * num_desired_candidates )
        n_samples = min( max_n_samples, n_samples )
        logging.debug( f'... generating {n_samples} new samples' )

        # Generate the samples
        #tic()
        new_candidates = lhs( parameter_space, n_samples )
        num_candidates_considered += n_samples
        #toc( f'    lhs({n_samples}): ' )
        # TODO - filter with "business rules" constraint, e.g. initial cases <= 10% of population
        # new_samples = new_samples[constraint(new_samples)]

        # Get non-implausible candidates
        plausibility = test_plausibility( new_candidates, emulator_bank, observations, config )
        plausible_candidates = new_candidates[plausibility]
        non_implausible_candidates = pd.concat( [non_implausible_candidates, plausible_candidates] )
        num_non_implausible_candidates = len( non_implausible_candidates )

        # Pring progress messages
        print_progress_bar( num_non_implausible_candidates, num_desired_candidates, num_candidates_considered )
        logging.debug( f'... found {len(plausible_candidates)} non-implausible candidates' )
        logging.debug( f'... {len(non_implausible_candidates)} non-implausible candidates so far from {num_candidates_considered} candidates ({len(non_implausible_candidates)/num_candidates_considered*100}% of candidates).' )

        # Abort if new candidates were not found
        if num_candidates_considered >= max_candidates:
            print( f'\n... unable to find new candidates after {num_candidates_considered} trials. Aborting the generation of new points.' )
            break
            
    # Finalize and return
    plausible_fraction = len(non_implausible_candidates) / num_candidates_considered
    print('')
    return non_implausible_candidates, plausible_fraction




def test_plausibility( candidates: pd.DataFrame, 
                       emulator_bank: Dict[int, Dict[str, BaseEmulator]], 
                       observations: pd.DataFrame, 
                       config: Config
                      ) -> pd.Series:
    """Run non-implausible candidates through each emulator and compare to observations."""

    # *** "non-implausible" is too hard to track - particularly when negated.
    # *** Use "plausible" instead, even if technically inaccurate.

    # Initially, all candidates are plausible
    plausible = np.ones( len(candidates), dtype=bool )

    # Visit iterations in order because earlier iterations will have been
    # trained on a wider range of parameter space.
    implausibility_threshold = config.implausibility_threshold
    for iteration in sorted(emulator_bank.keys()):

        logger.debug( f'    ... processing emulators from step {iteration}' )
        for feature in emulator_bank[iteration]:
            logger.debug( f'        loading emulator for feature {feature}' )
            emulator = emulator_bank[iteration][feature]

            #tic()
            logger.debug( f'        computing implausibility' )
            target      = observations[ observations['feature']==feature ]
            target_mean = target['mean'].values[0]
            target_var  = target['variance'].values[0]
            implausibility = emulator.get_implausibility( candidates, 
                                                          target_mean,
                                                          target_var,
                                                          config.model_discrepancy
                                                         )
            #toc( f'    ({feature}_estimate: ' )
            implausible = implausibility > implausibility_threshold

            # plausible candidates are _still_ plausible only if _not_ determined to be implausible
            plausible &= np.logical_not( implausible )
    
    return plausible




def print_progress_bar( n, n_target, n_considered, length=40, fill='█' ):
    percent = ('{0:.1f}').format( 100*(n/float(n_target)) )
    filled_length = int(length * n // n_target)
    bar = fill * filled_length + '-' * (length - filled_length)
    print( f'\rNew samples generated |{bar}| {percent}% (a total of {n_considered} samples have been considered)', end='\r' )
    if n == n_target: 
        print()
