"""History Matching Iterations

Collection of functions for the execution of one (or more) iterations 
of the history matching algorithm.
"""
import logging
import pandas as pd
from typing import Dict

from history_matching.emulators import BaseEmulator
from .config import Config
from .features import Diagnostics
from .emulators import GPR
from .constrict import next_point_generation

logger = logging.getLogger()




def do_step( config: Config, trace=None ):
    """
    Perform one step of history matching.

    Args:
        config : An instance of :class:`config.Config` containing
                 the configuration parameters.
        trace  : Array of items containing execution results and
                 information for (previously run) history matching 
                 steps. 

    Returns:
        An updated `trace` list that includes results and information
        for the current step.
    """
    print( f'Starting new History Matching iteration' )

    # Get general information and initialize step
    step_info = initialize_step( config, trace )
    print( f'... step_number = {step_info["step_number"]}' )
    # config.start_step_callback(situation)
    observations = config.observations.copy()
    test_points = config.sample_points.copy()
    
    # Run model (i.e., run simulator)
    if config.model is None:
        test_results = config.model_output
    else:
        print( '... running simulator' )
        test_results = config.model( test_points )
    
    # Train emulators  (need to extend to multiple emulators)
    if config.feature_selection_mode == 'manual':
        if config.feature is None:
            test_results_diagnostics = Diagnostics( test_points, test_results )
            test_results_diagnostics.interactive()
            selected_features = [ input( 'Please enter the feature or summary statistic to use as target for the emulator: ' ) ]
        else:
            selected_features = [ config.feature ]
    else:
        pass  # Need to check that this auto mode works correctly
        #selected_features = recipe.select_features(situation.iteration, situation.observations, situation.simulator_results, config)
    
    emulators = {}
    for feature in selected_features:
        logger.info( f'... training emulator for feature {feature}' )
        emulators[feature] = generate_emulator_for_feature( feature, observations, test_points, test_results )
    config.emulator_bank[ step_info['step_number'] ] = emulators
    
    # Generate new sample points
    (next_sample_points, non_implausible_fraction) = next_point_generation( config.parameter_space, 
                                                                            observations, 
                                                                            config.emulator_bank, 
                                                                            config
                                                                           )
    print( f'Remaining non-implausible space: {non_implausible_fraction*100:0.04}%' )

    #recipe.end_step_callback(situation)

    
    # Finalize and return
    step_info['test_points' ] = test_points
    step_info['test_results'] = test_results
    step_info['emulators'   ] = emulators
    step_info['new_samples' ] = next_sample_points
    step_info['non_implausible_fraction'] = non_implausible_fraction
    step_info['status'      ] = 'done'
    if trace is None:
        trace = []
    trace.append(step_info)
    logger.info( f'Finished step {step_info["step_number"]}.' )
    return trace




def initialize_step(config, trace):
    """ Initialize a history matching step """
    
    if trace is None:
        step_number = 1
    else:
        step_number = trace[-1]['step_number'] + 1

    step_info = { 'step_number'   : step_number,
                  'config'        : config,
                  'status'        : 'initialized'
                }
    return step_info



    
def generate_emulator_for_feature( feature          : str,
                                   observations     : pd.DataFrame,
                                   test_points      : pd.DataFrame,
                                   simulator_results: pd.DataFrame,
                                  ) -> BaseEmulator:
    """Generate an emulator for a single feature."""
    X = test_points
    y = simulator_results[feature].to_frame()    
    emulator = GPR(X, y)
    emulator.train()
    emulator.test()
    return emulator




def do_staircase(config: Config) -> None:
    """
    Run multiple steps of the history matching process until do_step() returns false.

    Args:
        config: the configuration for this history matching process

    Returns:
        None
    """

    # do_step() returns results of exit_predicate()
    # exit_predicate return True when it's time to quit
    #while not do_step(situation, recipe, config):
    #    pass  # all the work is in `do_step()`
    pass
    
    return
