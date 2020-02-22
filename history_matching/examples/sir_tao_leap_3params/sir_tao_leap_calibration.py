""" sir_tao_leap_calibration

  This script illustrates the use of history matching for identifying
  non-implausible regions for the parameters of a simple simulator, namely,
  an SIR model.

  The SIR model for this example is based on Gillespie's tao-leap method.
  A description of the algorithm can be found in:
    Keeling, M. and Rohani, P. Modeling Infectious Diseases in Animals and
    Humans. Princeton University Press, 2008, p.204.

  A description of the process followed in this script, as well as an example
  of the results obtained after its execution can be found at:

    https://wiki.idmod.org/display/EPI/History+Matching+Demos


"""
# Python libraries (native, 3rd party)
import pandas
import numpy
import os
import time
import datetime
import warnings
import matplotlib
matplotlib.pyplot.switch_backend('TKAgg')


# Python libraries (internal)
from history_matching         import HistoryMatching,                          \
                                     HistoryMatchingCut,                       \
                                     Basis
from runModel_sirTaoLeap      import runModel_sirTaoLeapIncidence
from sampleParameters         import sampleParameters
from getFeatures              import getFeatures
from selectFeature            import selectFeature
from getObservations          import getObservations
from getSampleFromObservation import getSampleFromObservation
from checkConvergence         import checkConvergence
from plotParams               import plotParams
from plotCostVsParams         import plotCostVsParams






#==== Parameters ===============================================================

# General
jobId = "SEAflu_simData_demo2"  # This will be used for creating
                                # a path with the job execution
                                # files

# Input data
inputDataFile = "./data/simulated_subject_database.csv"
pathogen      = 'h1n1pdm'

# History matching parameters
cutName = "incidence_simulatedFlu_SIRtaoLeap"
implausibilityThreshold = 3  # Typically 3 or 4
nProbes = 100                # Number of points to simulate per iteration
trainingFraction  = 0.9      # Fraction of sims used for emulator training
discrepancyStd = 2           # Model discrepancy
maxIter = 3                  # Maximum number of history matching iterations
glmBasis = "Poisson"         # Either "Gaussian" or "Poisson"

# Model parameters
modelParams_name = [ "beta", "gamma",     "N" ]
modelParams_min  = [   1e-6,    1e-6,     100 ]
modelParams_max  = [   0.25,       1,  100000 ]

# Other parameters
featureSelectionMetric = "mcorgmic"  # "mean", "var", "fano", "rsd", "qcd", "snr", 
                                # "mcor", "mcormic "mcorgmic"
seed = 101010                 # Seed for random number generators
verbose = True
#
#===============================================================================




#==== Calibration ==============================================================

# Initialization

# (1) Display input parameters
if (True):
    paramSummary = "\n"                                                        \
     + "-------------------------------------------------------------------\n" \
     + " Input Parameters:"                                             + "\n" \
     + "   inputDataFile           = " + str( inputDataFile           ) + "\n" \
     + "   pathogen                = " + str( pathogen                ) + "\n" \
     + "   cutName                 = " + str( cutName                 ) + "\n" \
     + "   implausibilityThreshold = " + str( implausibilityThreshold ) + "\n" \
     + "   nProbes                 = " + str( nProbes                 ) + "\n" \
     + "   trainingFraction        = " + str( trainingFraction        ) + "\n" \
     + "   discrepancyStd          = " + str( discrepancyStd          ) + "\n" \
     + "   maxIter                 = " + str( maxIter                 ) + "\n" \
     + "   glmBasis                = " +      glmBasis                  + "\n" \
     + "   modelParams_name        = " + str( modelParams_name        ) + "\n" \
     + "   modelParams_min         = " + str( modelParams_min         ) + "\n" \
     + "   modelParams_max         = " + str( modelParams_max         ) + "\n" \
     + "   seed                    = " + str( seed                    ) + "\n" \
     + "   verbose                 = " + str( verbose                 ) + "\n" \
     + "\n"                                                                    \
     + "[script: " + str(__file__) + "]\n"                                     \
     + "-------------------------------------------------------------------"
print( paramSummary )



# (2) Create file tree for the job, populate as needed, and move there
inputDataFile = os.path.abspath( inputDataFile )
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
newPath = "./" + jobId + "--" + timestamp
os.makedirs( newPath )
os.chdir( newPath )
os.makedirs( './main' )
os.chdir( './main' )
with open( "parameters.txt", "w" ) as paramFile:
    paramFile.write( paramSummary )
with open( "history.txt", "w" ) as historyFile:
    historyFile.write( \
       "Iteration  \t Rejection \t lambda(D) \t lambda(Cov(D)) \t Time (s) \n" )
    historyFile.write( \
       "           \t Rate (%)  \t           \t                \t          \n" )



# (3) Set general parameters for history_matching
xInfo = pandas.DataFrame( {  'Name': modelParams_name,
                             'Min' : modelParams_min,
                             'Max' : modelParams_max
                          } ).set_index('Name')

glmBasis = Basis.make_polynomial_basis( params       = xInfo.index.values,
                                   intercept    = True,
                                   first_order  = True,
                                   second_order = False,
                                   third_order  = False,
                                   param_info   = xInfo
                                  )

gprBasis = Basis.make_polynomial_basis( params      = xInfo.index.values,
                                   intercept   = False,
                                   first_order = True,
                                   param_info  = xInfo
                                  )


# (4) Initialize random number generator
if seed:
    numpy.random.seed(seed=seed)


# (5) Initialize arrays with general information about the job execution
rejectionPercent_historic = []
iterationTime_historic = []
lambdaD_historic = []
lambdaCovD_historic = []

featureBlacklist = []

selectedFeatures_name = []
selectedFeatures_mean = []
selectedFeatures_var  = []
selectedFeatures_selectionMetric = []

parameters_historic = pandas.DataFrame()
features_historic = pandas.DataFrame()
featureStats_historic = pandas.DataFrame()
#
#---- End of Initialization




# Get observations
#
# In this example, observations is a Pandas DataFrame with 3 columns, namely:
# (A) Time (in days);  (B) Incidence;   and (C) StdDev (standard deviation)
#
y = getObservations( inputDataFile, pathogen, verbose )


# History Matching calibration
#
it = 0
while (it < maxIter):

    tStartThisIter = time.time()
    print(" ", "*"*72, "\n  Iteration ", it, "\n ", "*"*72, "\n")


    # Sample model parameters to use in this iteration -------------------------
    if (it == 0):    # Generate parameters (covering the parameter space)
        x        = sampleParameters( xInfo, nProbes )

    else:            # Read candidates generated by history matching
        x = pandas.read_csv( os.path.join('Candidates_for_iter%d.csv'%it) )
        x.index.name = 'Sample_Id'

    plotParams( x,
                modelParams_min,
                modelParams_max,
                "Candidate Parameters at Iteration " + str(it),
                "Parameters_iter_" + str(it) + ".pdf",
                drawfigure=False
               )


    # Run simulation and get features/summary statistics  ----------------------
    print("... running simulations")
    xCommon = { "i0": 2,   "r0": 0,   "p_sampling": 1,   "nDays": 250 }
    ySimulated = runModel_sirTaoLeapIncidence( x, xCommon, "iter-"+str(it) )


    # Compute summary statistics
    print("... extracting features")
    [features, featureStats ] = getFeatures( numpy.stack(ySimulated),
                                             y["Incidence"].values,
                                             x.values
                                            )


    # Save historic data 
    features_historic = features_historic.append( features, ignore_index=True )
    featureStats_historic \
               = featureStats_historic.append( featureStats, ignore_index=True )
    parameters_historic = parameters_historic.append( x, ignore_index=True )

    features.to_csv( "features_historic_it" + str(it) + ".csv" )
    featureStats.to_csv( "featureStats_historic_it" + str(it) + ".csv" )
    x.to_csv( "parameters_historic_it" + str(it) + ".csv" )
    
    
    # Select feature for fitting emulator in history matching ------------------
    featureLoc, featureBlacklist = selectFeature( features, 
                                                  featureStats, 
                                                  featureSelectionMetric,
                                                  featureBlacklist,
                                                  timeToLiveInBlacklist = 5
                                                 )
    feature_1 = features.iloc[:, featureLoc ]


    # Prepare inputs for history matching --------------------------------------
    print("... history matching execution  --featureLoc = ", featureLoc)
    [desiredResult,
     desiredResultVar] = getSampleFromObservation( y, featureLoc )
    
    results = pandas.DataFrame( { "Sample_Id" : range(0, len(feature_1)),
                                  "Sim_Id"    : range(0, len(feature_1)),
                                  "Feature"   : feature_1.values,
                                  }
                               ).set_index( ["Sample_Id", "Sim_Id"] )["Feature"]
    

    # HM Init
    hm = HistoryMatching( cut_name = cutName,
                          param_info = xInfo,
                          inputs = x,
                          results = results,
                          desired_result = desiredResult,
                          desired_result_var = desiredResultVar**2,  # Why squared? sigma^2 instead of var?
                          iteration = it,
                          implausibility_threshold =implausibilityThreshold,
                          discrepancy_var = discrepancyStd**2, # Why squared? sigma^2 instead of var?
                          training_fraction = trainingFraction
                         )
    hm.save()


    # History matching computations --------------------------------------------
    #     NOTE: Repeat iteration (with new simulations and feature selection) if
    #           an error occurs during history matching computations. 
    #
    try:

        # GLM fit
        if (glmBasis == "Gaussian"):        
            hm.glm( basis = glmBasis,
                    family = 'Gaussian',
                    force_optimize_glm = True,
                    glm_fit_maxiter = int(1e5),
                    plot = True,
                    plot_data = True
                    )   
            
        else:  # Poisson basis
            hm.glm( basis = glmBasis,
                    family = "Poisson",
                    force_optimize_glm = True,
                    glm_fit_maxiter = int(1e5),
                    plot = True,
                    plot_data = True
                   )
            

        # GPR fit
        hm.gpr( basis = gprBasis,
                force_optimize_gpr = True,
                optimize_sigma2_n = True,
                log_transform = True,
                verbose = verbose,
                optimizer_options = { 'eps'    : 1e-3,
                                      'disp'   : False,
                                      'maxiter': int(1e5),
                                      'ftol'   : 1e-16,
                                      'gtol'   : 1e-16,
                                     },
                # optimizer_options = { 'eps'    : 1e-3,
                                      # 'disp'   : True,
                                      # 'maxiter': int(1e5),
                                      # 'ftol'   : 2*numpy.finfo(float).eps,
                                      # 'gtol'   : 2*numpy.finfo(float).eps,
                                     # },
                plot = True,
                plot_data = True
               )


        # Compute implausibility
        hm.calc_and_plot_implausibility( plot = True,
                                         do_plot_data = True,
                                         plot_data_highlight=pandas.DataFrame(),
                                         log_scale = True
                                        )
        hm.training_data.to_excel( "train_data__iter" + str(it) + ".xlsx" )
        hm.test_data.to_excel( "test_data__iter" + str(it) + ".xlsx" )


        # Select samples for next iteration
        hmc = HistoryMatchingCut( cut_dir = 'Cuts',
                                  iteration = it
                                 )

        (nonImplausibleCandidates,
         rejectedPercent          ) = hmc.cut( num_desired_candidates = nProbes,
                                               constraint = None
                                              )


        # Get ready for next iteration -----------------------------------------
        tEndThisIter = time.time()
        iterationTime_historic.append( tEndThisIter - tStartThisIter )
        rejectionPercent_historic.append( rejectedPercent['Rejected Percent'] )
        ( lambdaD, lambdaCovD ) = checkConvergence( nonImplausibleCandidates )
        lambdaD_historic.append( lambdaD )
        lambdaCovD_historic.append( lambdaCovD )
        with open( "history.txt", "a" ) as historyFile:
            historyFile.write( "   %d \t\t %4.2f \t\t %.4e \t %.4e \t\t %.2f\n"\
                               % ( it,
                                   rejectionPercent_historic[it],
                                   lambdaD_historic[it],
                                   lambdaCovD_historic[it],
                                   iterationTime_historic[it]
                                  )
                              )
                              
        selectedFeatures_name.append( feature_1.name )
        selectedFeatures_mean.append( featureStats["mean"].values[featureLoc] )
        selectedFeatures_var.append( featureStats["var"].values[featureLoc] )
        selectedFeatures_selectionMetric.append( \
                       featureStats[featureSelectionMetric].values[featureLoc] )

        it = it + 1
        
        if ( rejectedPercent['Rejected Percent'] > 99 ):
            break


    except:  # Manage error conditions (when executing history matching --------
             # operations (to do: print detailed error information)
             # 
        warnings.warn( "An exception occurred during the execution of " +     \
                       "history matching operations. Calibration process " +  \
                       "aborted."
                      )
#
#---- End of history matching iterations ---------------------------------------




# Closing
plotCostVsParams( parameters_historic,
                  features_historic["errorL1"].values,
                  modelParams_min,
                  modelParams_max,
                  "L1error",
                  "L1error.pdf",
                  drawfigure=False
                 )



print("\n\n*****************************************************************\n")
print( "Iteration  \t Rejection \t lambda(D) \t lambda(Cov(D)) \t Time (s)" )
print( "           \t Rate (%)  \t           \t                \t         " )
for j in range(0, it):
    print( "   %d \t\t %4.2f \t\t %.4e \t %.4e \t\t %.2f" % ( j,
                                            rejectionPercent_historic[j],
                                            lambdaD_historic[j],
                                            lambdaCovD_historic[j],
                                            iterationTime_historic[j]
                                      )
          )

print("\n\n*****************************************************************\n")

selectedFeatures = pandas.DataFrame( { "iteration"    : range(0, it),
                                       "feature name" : selectedFeatures_name,
                                       "mean"         : selectedFeatures_mean,
                                       "variance"     : selectedFeatures_var,
                    featureSelectionMetric : selectedFeatures_selectionMetric,
                                 }
                                ).set_index( "iteration" )
print( selectedFeatures )
selectedFeatures.to_csv("selectedFeaturesPerIteration.csv")
features_historic.to_csv( "features_historic_all.csv" )
parameters_historic.to_csv( "parameters_historic_all.csv" )




print("")
print("*******************************************************************")
print("Total time (in hours): ", str( sum( iterationTime_historic )/3600 ) )
print("*******************************************************************\n\n")

#plt.show()  # Uncomment this line to make sure Python doesn't exit and close
             # open figures before they were analyzed
#
#
#==== End of Calibration =======================================================

