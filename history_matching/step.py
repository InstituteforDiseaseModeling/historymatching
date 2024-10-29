"""History Matching Iterations

Collection of functions for the execution of one (or more) iterations 
of the history matching algorithm.
"""
import logging
import pandas as pd
import copy
from enum import Enum
from typing import Dict

from history_matching.emulators import BaseEmulator
from .config import Config
from .samplers import get_samples
from .features import Diagnostics, compute_stats, feature_selection
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
    # Get general information and initialize step
    step_info, status, test_points, test_results, features, emulators = initialize_step( config, trace )
    print( f'... step_number = {step_info["step_number"]}' )
    observations = config.observations.copy()
    test_points = config.sample_points.copy()

    # We are just starting, let's run some simulations
    if status == StepStatus.INITIALIZED:
        test_results, status = run_model( test_points, config, step_info )
        step_info['test_points'] = test_points
        step_info['sim_results'] = test_results

    # Read simulation results from file (only if the user runs the model externally)
    if status == StepStatus.WAITING_SIM_RESULTS:
        test_results, status = read_model_results( config )
        step_info['sim_results'] = test_results

    # Select features or summary statistics to use as emulator target
    if status == StepStatus.SIMS_COMPLETED:
        features, status = get_features( test_points, test_results, config )
        step_info['features'] = features
    
    # Train the emulators
    if status == StepStatus.FEATURES_SELECTED:
        emulators, status = train_emulators( test_points, test_results, observations, features )
        step_info['emulators'] = emulators
        config.emulator_bank[ step_info['step_number'] ] = emulators

    # Sample the parameter space
    if status == StepStatus.EMULATORS_TRAINED:
        ( next_sample_points, \
          non_implausible_fraction ) = next_point_generation( config.parameter_space, 
                                                              observations, 
                                                              config.emulator_bank, 
                                                              config
                                                             )
        print( f'Remaining non-implausible space: {non_implausible_fraction*100:0.04}%' )
        step_info['new_samples'] = next_sample_points
        step_info['non_implausible_fraction'] = non_implausible_fraction
        status = StepStatus.NEW_SAMPLES_GENERATED

    # We are done with the step; let's do everything that we have to do at the end.    
    if status == StepStatus.NEW_SAMPLES_GENERATED:
        print( f'Finished step {step_info["step_number"]}.' )
        status = StepStatus.DONE
    
    # Finalize and return
    trace = update_trace( step_info, status, trace )
    return trace




def initialize_step(config, trace):
    """ Initialize a history matching step. """

    # Let's assume nothing has happened in this step; these variables will be 
    # updated if this is a continuation of an unfinished step.
    test_points = config.sample_points
    test_results = None
    features = None
    emulators = None

    # This is the first step, let's initialize some values
    if trace is None:
        print( f'Starting first History Matching iteration' )
        step_number = 1
        status = StepStatus.INITIALIZED
        if config.sample_points is None:
            logger.info( '... Sampling the parameter space to generate "sample_points"' )
            config.sample_points = get_samples( config.parameter_space, 
                                                config.n_candidates, 
                                                config.draw_samples 
                                               )
    
    # This is not the first step
    else:
        last_step = trace[-1]

        # The last step finished already; we are starting something new.
        if last_step['status'] == StepStatus.DONE:
            step_number = last_step['step_number'] + 1
            print( f'Starting new History Matching iteration (iter={step_number})' )

            status = StepStatus.INITIALIZED
            if config.sample_points is None:
                config.sample_points = last_step['new_samples']

        # The last step has not finished; let's continue with its execution
        else:    
            step_number = last_step['step_number']
            print( f'Resuming a History Matching iteration (iter={step_number})' )
            
            status = last_step['status']
            config.sample_points = last_step.get( 'test_points', config.sample_points )
            test_points = last_step.get( 'test_points', config.sample_points )
            test_results = last_step.get( 'sim_results' )
            features = last_step.get( 'features' )
            emulators = last_step.get( 'emulators' )

    # Load all the emulators that have been previously trained
    if trace is not None:
        for i in range( len(trace) ):
            if 'emulators' in trace[i]:
                config.emulator_bank[ trace[i]['step_number'] ] = trace[i]['emulators']
              

    # Data validation
    validate_sample_points( config.sample_points, config.parameter_space )
    
    # Save relevant information and return
    step_info = { 'step_number'   : step_number,
                  'config'        : config,
                  'status'        : status
                 }
    return step_info, step_info['status'], test_points, test_results, features, emulators




def run_model( test_points, config, step_info ):
    """ Run model (i.e., run simulator). """

    # Initialization
    test_results = None  # Nothing has been done yet
    status = StepStatus.INITIALIZED    # The only way to be here is if this
                                       # was the status; this value gets 
                                       # updated if something happens here.

    # Run the model if the user provides a function to call the model
    if config.model is not None:
        print( '... Running simulator' )
        test_results = config.model( test_points )
        status = StepStatus.SIMS_COMPLETED

    # Or output the samples for the user to run the model externaly
    else:
        print( '... Please run simulations using the following samples.' )
        default_filename = f'./sample_points_step_{step_info["step_number"]}.csv'
        user_filename = input( f'    Please enter the file name to save the samples (press ENTER to use "{default_filename}")' )
        filename = default_filename   if not user_filename   else user_filename
        test_points.to_csv( filename )
        status = StepStatus.WAITING_SIM_RESULTS
        
    # Return updated results and status
    return test_results, status




def read_model_results( config ):
    """ Read model results from file. """
    test_results = None  # Nothing has been done yet
    status = StepStatus.WAITING_SIM_RESULTS    # The only way to be here is if this
                                               # was the status; this value gets 
                                               # updated if something happens here.

    # The user provided a filename to access the results, let's read it
    if config.model_output:
        print( f'... Reading simulation results from {config.model_output}.' )
        test_results = pd.read_csv( config.model_output )
        status = StepStatus.SIMS_COMPLETED
                
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
    status = StepStatus.SIMS_COMPLETED    # The only way to be here is if this
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
        status = StepStatus.FEATURES_SELECTED

    # Auto mode: an algorithm automatically selects the feature
    else:
        print( '... Selecting feature (running in "auto" mode)' )
        features_stats = compute_stats( sim_results, 'fano' ) # We only need tco compute the Fano factor 
                                                              # for every summary statistic
        features = feature_selection( sim_results, features_stats, 'fano' )
        status = StepStatus.FEATURES_SELECTED
    
    # Return selected features and status
    return features, status



        
def train_emulators( test_points, test_results, observations, features ):
    """ Train emulators. Currently only one emulator is supported, but this 
    should be extended to support a configuration of the desired emulator
    (via a config attribute). """
    emulators = {}  # Nothing has been done yet
    status = StepStatus.FEATURES_SELECTED    # The only way to be here is if this
                                             # was the status; this value gets 
                                             # updated if something happens here.
    
    for feature in features:
        print( f'... Training emulator for feature {feature}' )
        emulators[feature] = generate_emulator_for_feature( feature, observations, test_points, test_results )
    status = StepStatus.EMULATORS_TRAINED

    # Return trained emulators and status
    return emulators, status




def generate_emulator_for_feature( feature          : str,
                                   observations     : pd.DataFrame,
                                   test_points      : pd.DataFrame,
                                   simulator_results: pd.DataFrame,
                                  ) -> BaseEmulator:
    """Generate an emulator for a single feature."""
    validate_simulator_results( simulator_results, observations, feature )

    X = test_points
    y = simulator_results[feature].to_frame()

    emulator = GPR( X, y )
    emulator.train()
    emulator.test()
    
    return emulator




def update_trace( step_info, status, trace=None ):
    """ Update trace dict. """
    # Create a new trace if it doesn't exist
    if trace is None:
        new_trace = []
        last_step = 0
    else:
        last_step = trace[-1]['step_number']
        new_trace = trace.copy()

    # Add additional data to step_info
    step_info['status'] = status

    # Add a new item to the trace, if the step just started or already finished;
    # otherwise overwrite the last item of the trace
    if  last_step != step_info['step_number']:
        new_trace.append( step_info )
    else:
        for key, item in step_info.items():
            new_trace[-1][key] = item
    
    return new_trace



    
def validate_sample_points(sample_points: pd.DataFrame, parameter_space: pd.DataFrame) -> None:
    """
    Validate the sample points.

    Args:
        sample_points: the sample points
        parameter_space: the parameter space

    Raises:
        TypeError: if the sample points is not a DataFrame
        ValueError: if the sample points is empty
        ValueError: if the sample points has duplicate parameter names
        ValueError: if the sample points has a parameter with a single value
        ValueError: if the sample points has a parameter with a negative value
        ValueError: if the sample points has a parameter with a zero value
        ValueError: if the sample points has a parameter with a non-numeric value
        ValueError: if the sample points has a parameter with a value outside the parameter space
    """
    if not isinstance(sample_points, pd.DataFrame):
        raise TypeError(f"Sample points should be Pandas DataFrame, not '{type(sample_points)}'")
    required_columns = []
    required_columns.extend(parameter_space.parameter)
    if not all(column in sample_points.columns for column in required_columns):
        raise RuntimeError(f"Sample points must contain the columns {required_columns}. Found {sample_points.columns}.")
    if len(sample_points) == 0:
        raise RuntimeError("Sample points must specify at least one point in parameter space. Found none.")
    valid = True
    msg = ""
    for irow in range(len(sample_points)):
        row = sample_points.iloc[irow]
        for parameter_spec in parameter_space.itertuples():
            if (row[parameter_spec.parameter] < parameter_spec.minimum) or (row[parameter_spec.parameter] > parameter_spec.maximum):
                valid = False
                msg += f"Sample parameter, {row}, is outside parameter space."
    if not valid:
        raise ValueError(msg)

    return




def validate_simulator_results( simulator_results: pd.DataFrame,
                                observations: pd.DataFrame,
                                feature : str
                               ) -> None:
    """
    Validate the simulator results.

    Args:
        simulator_results: the simulator results
        observations: the observations
        feature: name of the feature to process

    Raises:
        TypeError: if the simulator results is not a DataFrame
        ValueError: if the simulator results is empty
        ValueError: if the simulator results has duplicate parameter names
        ValueError: if the simulator results has a parameter with a single value
        ValueError: if the simulator results has a parameter with a negative value
        ValueError: if the simulator results has a parameter with a zero value
        ValueError: if the simulator results has a parameter with a non-numeric value
        ValueError: if the simulator results has a parameter with a value outside the parameter space
    """
    if not isinstance(simulator_results, pd.DataFrame):
        raise TypeError( f'Simulator results should be Pandas DataFrame, not "{type(simulator_results)}"' )
    required_columns = [feature]
    if not all( column in simulator_results.columns for column in required_columns ):
        raise RuntimeError( f'Simulator results must contain the columns {required_columns}. Found {simulator_results.columns}' )
    if simulator_results[required_columns].isna().any().any():
        raise ValueError( f'Simulator results must contain numeric values. Missing data or NaNs are not supported' )

    return

    


def reduce_space( config             : Config, 
                  min_space_fraction : float = 0.05,
                  max_iter           : int   = 5,
                  trace              : list  = None
                 ) -> list :
    """
    Run multiple steps of the history matching process, iteratively reducing 
    the parameter space, until the parameter space reduction reaches the 
    specified threshold or the maximum number of iterations is reached.

    Args:
        config: The configuration object for the history matching process.
        min_space_fraction (float, optional): The threshold for stopping 
                based on the fraction of the remaining parameter space. 
                Iterations stop once this fraction is reached. Defaults to 0.05.
        max_iter (int, optional): The maximum number of iterations to perform, 
                regardless of the space reduction. Defaults to 5.
        trace (list, optional): A list to store the trace of the history 
                matching process. If None, a new trace will be created. 
                Defaults to None.

    Returns:
        list: The trace of the history matching process after completing 
              the iterations or stopping early.    
    """

    for i in range( max_iter ):
        trace = do_step( config, trace )
        print( '' )
        if trace[-1]['non_implausible_fraction'] <= min_space_fraction:
            print( f'... Stopping early at iteration {i+1} due to space fraction reduction' )
            break
    
    return trace