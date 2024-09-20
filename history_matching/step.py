"""History Matching Iterations

Collection of functions for the execution of one (or more) iterations 
of the history matching algorithm.
"""
import logging
import pandas as pd
from enum import Enum
from typing import Dict

from history_matching.emulators import BaseEmulator
from .config import Config
from .samplers import lhs
from .features import Diagnostics
from .emulators import GPR
from .constrict import next_point_generation

logger = logging.getLogger()



class StepStatus( Enum ):
    """ Define status for diferent stages of an iteration. """
    INITIALIZED = 'initialized'
    WAITING_SIM_RESULTS = 'waiting for simulation results'
    SIMS_COMPLETED = 'simulations completed'
    FEATURES_SELECTED = 'features selected'
    EMULATORS_TRAINED = 'emulators trained'
    NEW_SAMPLES_GENERATED = 'in progress'
    DONE = 'done'




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
    step_info, status, test_results, features, emulators = initialize_step( config, trace )
    print( f'... step_number = {step_info["step_number"]}' )
    observations = config.observations.copy()
    test_points = config.sample_points.copy()

    # We are just starting, let's run some simulations
    if status == StepStatus.INITIALIZED.value:
        test_results, status = run_model( test_points, config, step_info )
        step_info['sim_results'] = test_results

    # Read simulation results from file (only if the user runs the model externally)
    if status == StepStatus.WAITING_SIM_RESULTS.value:
        test_results, status = read_model_results( config )
        step_info['sim_results'] = test_results

    # Select features or summary statistics to use as emulator target
    if status == StepStatus.SIMS_COMPLETED.value:
        features, status = get_features( test_points, test_results, config )
        step_info['features'] = features
    
    # Train the emulators
    if status == StepStatus.FEATURES_SELECTED.value:
        emulators, status = train_emulators( test_points, test_results, observations, features )
        step_info['emulators'] = emulators
        config.emulator_bank[ step_info['step_number'] ] = emulators

    # Sample the parameter space
    if status == StepStatus.EMULATORS_TRAINED.value:
        ( next_sample_points, \
          non_implausible_fraction ) = next_point_generation( config.parameter_space, 
                                                              observations, 
                                                              config.emulator_bank, 
                                                              config
                                                             )
        print( f'Remaining non-implausible space: {non_implausible_fraction*100:0.04}%' )
        step_info['new_samples'] = next_sample_points
        step_info['non_implausible_fraction'] = non_implausible_fraction
        status = StepStatus.NEW_SAMPLES_GENERATED.value

    # We are done with the step; let's do everything that we have to do at the end.    
    if status == StepStatus.NEW_SAMPLES_GENERATED.value:
        print( f'Finished step {step_info["step_number"]}.' )
        status = StepStatus.DONE.value
    
    # Finalize and return
    trace = update_trace( step_info, status, trace )
    return trace




def initialize_step(config, trace):
    """ Initialize a history matching step. """

    # Let's assume nothing has happened in this step; these variables will be 
    # updated if this is a continuation of an unfinished step.
    test_results = None
    features = None
    emulators = None

    # This is the first step, let's initialize some values
    if trace is None:
        step_number = 1
        status = StepStatus.INITIALIZED.value
        if config.sample_points is None:
            logger.info( '... Sampling the parameter space to generate "sample_points"' )
            config.sample_points = lhs( config.parameter_space, config.n_candidates )

    # This is not the first step
    else:
        last_step = trace[-1]

        # The last step finished already; we are starting something new.
        if last_step['status'] == StepStatus.DONE.value:
            step_number = last_step['step_number'] + 1
            status = StepStatus.INITIALIZED.value
            if config.sample_points is None:
                config.sample_points = last_step['new_samples']

        # The last step has not finished; let's continue with its execution
        else:    
            step_number = last_step['step_number']
            status = last_step['status']
            config.sample_points = last_step.get( 'test_points', config.sample_points )
            test_results = last_step.get( 'sim_results' )
            features = last_step.get( 'features' )
            emulators = last_step.get( 'emulators' )

    # Save relevant information and return
    step_info = { 'step_number'   : step_number,
                  'config'        : config,
                  'status'        : status
                 }
    return step_info, step_info['status'], test_results, features, emulators




def run_model( test_points, config, step_info ):
    """ Run model (i.e., run simulator). """

    # Initialization
    test_results = None  # Nothing has been done yet
    status = StepStatus.INITIALIZED.value    # The only way to be here is if this
                                             # was the status; this value gets 
                                             # updated if something happens here.

    # Run the model if the user provides a function to call the model
    if config.model is not None:
        print( '... Running simulator' )
        test_results = config.model( test_points )
        status = StepStatus.SIMS_COMPLETED.value

    # Or output the samples for the user to run the model externaly
    else:
        print( '... Please run simulations using the following samples.' )
        default_filename = f'./sample_points_step_{step_info["step_number"]}.csv'
        user_filename = input( f'    Please enter the file name to save the samples (press ENTER to use "{default_filename}")' )
        filename = temp_filename   if not user_filename   else user_filename
        test_points.to_csv( filename )
        status = StepStatus.WAITING_SIM_RESULTS.value
        
    # Return updated results and status
    return test_results, status




def read_model_results( config ):
    """ Read model results from file. """
    test_results = None  # Nothing has been done yet
    status = StepStatus.WAITING_SIM_RESULTS.value    # The only way to be here is if this
                                                     # was the status; this value gets 
                                                     # updated if something happens here.

    # The user provided a filename to access the results, let's read it
    if config.model_output:
        print( f'... Reading simulation results from {config.model_output}.' )
        test_results = pd.read_csv( config.model_output )
        status = StepStatus.SIMS_COMPLETED.value
                
    # There is no filename! Let's tell the user how to provide the simulation/model results
    else:
        print( '... Simulation results should be saved in "config.model_output"' )
        print( '    Waiting for simulation results' )
        
    # Return updated results and status
    return test_results, status




def get_features( samples, sim_results, config ):
    """ Select features to use as targets for the emulators. Currently supporting
    only one feature, but could be extended to support multiple features. """
    features = None  # Nothing has been done yet
    status = StepStatus.SIMS_COMPLETED.value    # The only way to be here is if this
                                                # was the status; this value gets 
                                                # updated if something happens here.

    # Manual mode: read the feature indicated by the user, or ask the user to select a feature
    if config.feature_selection_mode == 'manual':
        if config.feature is not None:
            features = [ config.feature ]
        else:
            test_results_diagnostics = Diagnostics( samples, sim_results )
            test_results_diagnostics.interactive()
            features = [ input( 'Please enter the feature or summary statistic to use as target for the emulator: ' ) ]
        status = StepStatus.FEATURES_SELECTED.value

    # Auto mode: an algorithm automatically selects the feature
    else:
        print( '... "auto" mode for feature selection is not supported yet' )
        # Need to check that this auto mode works correctly
        #selected_features = recipe.select_features(situation.iteration, situation.observations, situation.simulator_results, config)
    
    # Return selected features and status
    return features, status



        
def train_emulators( test_points, test_results, observations, features ):
    """ Train emulators. Currently only one emulator is supported, but this 
    should be extended to support a configuration of the desired emulator
    (via a config attribute). """
    emulators = {}  # Nothing has been done yet
    status = StepStatus.FEATURES_SELECTED.value # The only way to be here is if this
                                                # was the status; this value gets 
                                                # updated if something happens here.
    
    for feature in features:
        print( f'... training emulator for feature {feature}' )
        emulators[feature] = generate_emulator_for_feature( feature, observations, test_points, test_results )
    status = StepStatus.EMULATORS_TRAINED.value

    # Return trained emulators and status
    return emulators, status




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




def update_trace( step_info, status, trace=None ):
    """ Update trace dict. """
    # Create a new trace if it doesn't exist
    if trace is None:
        trace = []

    # Add additional data to step_info
    step_info['status'] = status

    # Add a new item to the trace, if the step just started or already finished;
    # otherwise overwrite the last item of the trace
    if (status==StepStatus.WAITING_SIM_RESULTS.value) or (status==StepStatus.DONE.value):
        trace.append( step_info )
    else:
        trace[-1] = step_info
    
    return trace

    


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
