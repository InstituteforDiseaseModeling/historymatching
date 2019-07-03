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
import re
import time
import datetime
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')


# Python libraries (internal)
from history_matching         import HistoryMatching,                          \
                                     HistoryMatchingCut,                       \
                                     Basis
from runModel_sirTaoLeap      import runModel_sirTaoLeapIncidence
from sampleParameters         import sampleParameters
from getFeatures              import getFeatures
from getObservations          import getObservations
from getSampleFromObservation import getSampleFromObservation
from checkConvergence         import checkConvergence
from plotParams               import plotParams




#==== Parameters ===============================================================

# General
jobId = 'seattleFluSimulated_SirTaoLeap'        # This will be used for creating
                                                # a path with the job execution
                                                # files

# Input data
inputDataFile = "./data/simulated_subject_database.csv"
pathogen      = 'h1n1pdm'

# History matching parameters
cutName = "incidence_simulatedFlu_SIRtaoLeap"
implausibilityThreshold = 3  # Typically 3 or 4
nProbes = 1000               # Number of points to simulate per iteration
trainingFraction  = 0.9      # Fraction of sims used for emulator training
discrepancyStd = 2           # Model discrepancy
maxIter = 25                 # Maximum number of history matching iterations

# Model parameters
modelParams_name = [ "beta", "gamma",     "N" ]
modelParams_min  = [   1e-6,    1e-6,     100 ]
modelParams_max  = [   0.25,       1,  100000 ]

# Other parameters
weightAttenationDn = 0
weightAttenationUp = 1e4
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

glmBasis = Basis.polynomial_basis( params       = xInfo.index.values,
                                   intercept    = True,
                                   first_order  = True,
                                   second_order = False,
                                   third_order  = False,
                                   param_info   = xInfo
                                  )

gprBasis = Basis.polynomial_basis( params      = xInfo.index.values,
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

summaryStatistics_1_name_historic = [] # Summary statistic used for GLM/GPR
summaryStatistics_1_mean_historic = [] # fitting at each iteration
summaryStatistics_1_var_historic  = []
summaryStatistics_2_name_historic = [] # Next summary statistic considered for
summaryStatistics_2_mean_historic = [] # fitting (but not used)
summaryStatistics_2_var_historic  = []
summaryStatistics_3_name_historic = [] # 3rd summary statistic considered for
summaryStatistics_3_mean_historic = [] # fitting (but not used)
summaryStatistics_3_var_historic  = []

#
#---- End of Initialization




# Get observations
#
# In this example, observations is a Pandas DataFrame with 3 columns, namely:
# (A) Time (in days);  (B) Incidence;   and (C) StdDev (standard deviation)
#
y = getObservations( inputDataFile, pathogen, verbose )
featureSelectionWeight = numpy.ones( 4 + len(y) )  # 4 additional features
onesArray = numpy.ones( 4 + len(y) )               # 4 additional features


# History Matching calibration
#
it = 0
while (it < maxIter):

    tStartThisIter = time.time()
    print(" ", "*"*72, "\n  Iteration ", it, "\n ", "*"*72, "\n")


    # Sample model parameters to use in this iteration.
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
               )

    # Run simulation for the parameters sampled
    xCommon = { "i0": 2,   "r0": 0,   "p_sampling": 1,   "nDays": 250 }
    ySimulated = runModel_sirTaoLeapIncidence( x, xCommon, "iter-"+str(it) )


    # Compute summary statistics
    [features, featureMean, featureVar ] = getFeatures( ySimulated,
                                                        y["Incidence"].values
                                                       )


    # Select feature to match in history matching
    #sortedFeatureIndex = numpy.argsort( -featureVar )   # Sort in reverse order
    sortedFeatureIndex = numpy.argsort( numpy.multiply( -featureSelectionWeight, 
                                                      featureVar/(1+featureMean)
                                                       )
                                       )
    
    feature_1 = features.iloc[:, sortedFeatureIndex[0] ]
    feature_2 = features.iloc[:, sortedFeatureIndex[1] ]
    feature_3 = features.iloc[:, sortedFeatureIndex[2] ]

    feature_1_mean = featureMean[ sortedFeatureIndex[0] ]
    feature_2_mean = featureMean[ sortedFeatureIndex[1] ]
    feature_3_mean = featureMean[ sortedFeatureIndex[2] ]

    feature_1_var = featureVar[ sortedFeatureIndex[0] ]
    feature_2_var = featureVar[ sortedFeatureIndex[1] ]
    feature_3_var = featureVar[ sortedFeatureIndex[2] ]

    print("feature_1: ", feature_1.name, 
          " -- ", feature_1_mean, 
          " -- ", feature_1_var )


 
    # Prepare inputs for history matching
    [desiredResult,
     desiredResultVar] = getSampleFromObservation( y, sortedFeatureIndex[0] )
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


    # GLM fit
    hm.glm( basis = glmBasis,
            family = 'Gaussian',
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


    # Get ready for next iteration
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

    summaryStatistics_1_name_historic.append( feature_1.name )
    summaryStatistics_2_name_historic.append( feature_2.name )
    summaryStatistics_3_name_historic.append( feature_3.name )

    summaryStatistics_1_mean_historic.append( feature_1_mean )
    summaryStatistics_2_mean_historic.append( feature_2_mean )
    summaryStatistics_3_mean_historic.append( feature_3_mean )

    summaryStatistics_1_var_historic.append( feature_1_var )
    summaryStatistics_2_var_historic.append( feature_2_var )
    summaryStatistics_3_var_historic.append( feature_3_var )
    
    featureSelectionWeight = weightAttenationUp * featureSelectionWeight
    featureSelectionWeight = numpy.minimum( onesArray, featureSelectionWeight )
    featureSelectionWeight[ sortedFeatureIndex[0] ] \
     = max( 1e-12, 
            weightAttenationDn * featureSelectionWeight[ sortedFeatureIndex[0] ]
           )
           
    print("featureSelectionWeight ===> ", featureSelectionWeight )
    print("feature_1: ", feature_1.name, 
          " -- ", feature_1_mean, 
          " -- ", feature_1_var )
           
           
    it = it + 1
#
#---- End of history matching iterations ---------------------------------------




# Closing
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

top3Features = pandas.DataFrame( { "Iteration" : range(0, it),
                                "(#1) Name" : summaryStatistics_1_name_historic,
                                "(#1) Mean" : summaryStatistics_1_mean_historic,
                                "(#1) Var"  : summaryStatistics_1_var_historic,
                                "(#2) Name" : summaryStatistics_2_name_historic,
                                "(#2) Mean" : summaryStatistics_2_mean_historic,
                                "(#2) Var"  : summaryStatistics_2_var_historic,
                                "(#3) Name" : summaryStatistics_3_name_historic,
                                "(#3) Mean" : summaryStatistics_3_mean_historic,
                                "(#3) Var"  : summaryStatistics_3_var_historic,
                                 }
                                ).set_index( "Iteration" )
print( top3Features )
top3Features.to_csv("top3Features.csv")

print("")
print("*******************************************************************")
print("Total time (in hours): ", str( sum( iterationTime_historic )/3600 ) )
print("*******************************************************************\n\n")

#plt.show()  # Uncomment this line to make sure Python doesn't exit and close
             # open figures before they were analyzed
#
#
#==== End of Calibration =======================================================