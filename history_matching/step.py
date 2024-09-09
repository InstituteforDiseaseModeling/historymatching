# one step of history matching (see architecture diagram)
"""Version 2 of History Matching 2022

Description of this module.

Example:
    Or ``Examples``.

Attributes:

Todo:

"""

import logging
from typing import Dict

import pandas as pd

from history_matching.emulators import BaseEmulator

from .config import Config
from .recipe import Recipe
from .situation import Situation


from .features import Diagnostics
from .emulators import GPR
from .constrict import next_point_generation


logger = logging.getLogger()


def do_step( config: Recipe, trace=None ):
    """
    Perform one step of history matching.

    Args:
        config
        trace

    Returns:
        trace
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
    return trace




def initialize_step(config, trace):

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




def do_step_orig(situation: Situation, recipe: Recipe, config: Config) -> bool:
    """
    Perform one step of history matching.

    Args:
        situation: the current state of the history matching process
        recipe: the recipe for this history matching process
        config: the configuration for this history matching process

    Returns:
        True if the history matching process should continue, False otherwise
    """

    logger.info(f"Starting step {situation.iteration}...")

    situation.validate()

    recipe.start_step_callback(situation)

    test_points = get_test_points_for_iteration(situation.iteration, situation.sample_points)

    test_results = recipe.run_simulators(situation.iteration, test_points, config)

    merge_results(situation.iteration, test_results, situation, config)

    selected_features = recipe.select_features(situation.iteration, situation.observations, situation.simulator_results, config)

    new_emulators = recipe.generate_emulators(situation.iteration, selected_features, situation.observations, situation.simulator_results, recipe.generate_emulator_for_feature, config)

    deposit_emulators(situation.iteration, new_emulators, situation, config)

    (next_sample_points, non_implausible_fraction) = recipe.generate_next_sample_points(situation.iteration, situation.parameter_space, situation.observations, situation.emulator_bank, config)
    logger.info(f"Remaining non-implausible space: {non_implausible_fraction*100:0.04}%")

    update_test_points(situation.iteration, next_sample_points, situation)

    recipe.end_step_callback(situation)

    logger.info(f"Finished step {situation.iteration}...")

    situation.iteration += 1

    return recipe.exit_predicate(situation.iteration, non_implausible_fraction, config)


def get_test_points_for_iteration(iteration: int, sample_points: pd.DataFrame) -> pd.DataFrame:
    """Get the sample points specified or generated in the previous iteration."""
    logger.info(f'getting test points for iteration {iteration} in the sample points dataframe')
    test_points = sample_points[sample_points.iteration == iteration].copy()

    return test_points


def merge_results(iteration: int, test_results: pd.DataFrame, situation: Situation, config: Config) -> None:
    """Add simulator results from this iteration into the full set of simulator results."""
    logger.info(f"Merging {len(test_results)} new simulator results with {len(situation.simulator_results)} existing results...")
    assert all(test_results.iteration == iteration), "Test results include results from a different iteration."
    print(f"Concatenating {len(situation.simulator_results)} existing results with {len(test_results)} new results.")
    situation.simulator_results = pd.concat([df for df in [situation.simulator_results, test_results] if len(df)])
    situation.simulator_results.reset_index(drop=True)

    return


def deposit_emulators(iteration: int, new_emulators: Dict[str, BaseEmulator], situation: Situation, config: Config) -> None:
    """Add emulator(s) from this iteration to the complete set of emulators."""
    logger.info(f"Adding {len(new_emulators.keys())} emulator(s) to emulator_bank on step {iteration}...")
    situation.emulator_bank.update({iteration: new_emulators})

    return


def update_test_points(iteration: int, next_sample_points: pd.DataFrame, situation: Situation) -> None:
    """Add sample points generated on this iteration to the full set of sample points."""
    logger.info(f"Adding {len(next_sample_points)} new sample points on step {iteration}...")
    next_sample_points["iteration"] = iteration + 1
    situation.sample_points = pd.concat([df for df in [situation.sample_points, next_sample_points] if len(df)]).reset_index(drop=True)

    return


def do_staircase(situation: Situation, recipe: Recipe, config: Config) -> None:
    """
    Run multiple steps of the history matching process until do_step() returns false.

    Args:
        situation: the current state of the history matching process
        recipe: the recipe for this history matching process
        config: the configuration for this history matching process

    Returns:
        None
    """

    # do_step() returns results of exit_predicate()
    # exit_predicate return True when it's time to quit
    while not do_step(situation, recipe, config):
        pass  # all the work is in `do_step()`

    return
