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
from history_matching   import HistoryMatching,    \
                               HistoryMatchingCut, \
                               Basis
from getObservations    import getObservations
from sampleObservations import sampleObservations
from sampleParameters   import sampleParameters
from runModel           import runModel
from checkConvergence   import checkConvergence
from dataframePlot      import dataframePlot




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
trainingFraction  = 0.75     # Fraction of sims used for emulator training 
discrepancyStd = 2           # Model discrepancy 
maxIter = 10                 # Maximum number of history matching iterations

nSamples = 1                 # Number of (sampled) observations per iteration
samplingMode = 'random'      # Sampling mode for observations

# Model parameters
minBeta  = 1e-6
maxBeta  = 0.25
minGamma = 1e-6
maxGamma = 1

# Other parameters 
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
     + "   nSamples                = " + str( nSamples                ) + "\n" \
     + "   samplingMode            = " + str( samplingMode            ) + "\n" \
     + "   minBeta                 = " + str( minBeta                 ) + "\n" \
     + "   maxBeta                 = " + str( maxBeta                 ) + "\n" \
     + "   minGamma                = " + str( minGamma                ) + "\n" \
     + "   maxGamma                = " + str( maxGamma                ) + "\n" \
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
xInfo = pandas.DataFrame( {  'Name': [ 'beta',  'gamma'  ],
                             'Min' : [ minBeta, minGamma ],
                             'Max' : [ maxBeta, maxGamma ]
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

    # Sample model parameters to use in this iteration. The sampled parameters
    # are saved into ySampled
    if (it == 0):
        x        = sampleParameters( xInfo, nProbes, it, verbose )
        ySampled = sampleObservations( y, 1, "max" )
        
    else:
        x = pandas.read_csv( os.path.join('Candidates_for_iter%d.csv'%it) )
        x.index.name = 'Sample_Id'
        ySampled = sampleObservations( y, nSamples, samplingMode )
        
            
    # Run simulation for the parameters sampled
    ySimulated = runModel( 'sirTaoLeap_betaGamma', x, it, ySampled, verbose )

   
    for index, yCurrent in ySampled.iterrows():   
           
        yCurrent['Times'] = int( yCurrent['Times'] )
        
        # Extract simulated incidence corresponding to the one in yCurrent
        t = yCurrent['Times']
        ySimulated_at_t = ySimulated                                           \
            .query('ObsTime==@t')[['Sample_Id', 'Sim_Id',  'Incidence']]       \
            .set_index( ['Sample_Id', 'Sim_Id'] )['Incidence']

        # HM Init
        hm = HistoryMatching( cut_name = cutName,
                              param_info = xInfo,
                              inputs = x,
                              results = ySimulated_at_t,
                              desired_result = yCurrent['Incidence'],
                              desired_result_var = yCurrent['StdDev']**2,
                              iteration = it,
                              implausibility_threshold =implausibilityThreshold,
                              discrepancy_var = discrepancyStd**2,
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
        hm.gpr( basis = gprBasis,             # See note (A)
                force_optimize_gpr = True,
                optimize_sigma2_n = True,     # See note (A)
                log_transform = True,         # See note (A)
                verbose = verbose, 
                optimizer_options = { 'eps'    : 1e-3,
                                      'disp'   : True,
                                      'maxiter': int(1e5),
                                      'ftol'   : 2*numpy.finfo(float).eps,
                                      'gtol'   : 2*numpy.finfo(float).eps,
                                     },
                plot = True,
                plot_data = True
               )
        
        
        # Compute implausibility
        hm.calc_and_plot_implausibility( plot = True,
                                         do_plot_data = True,
                                         plot_data_highlight=pandas.DataFrame(),
                                         log_scale = True    # See note (A)
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
        dataframePlot( nonImplausibleCandidates, 
                       ["beta", "gamma"], 
                       "paramater candidates for iteration " + str(it+1),
                       "parameters-iter-" + str(it+1) + ".png"
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
        it = it + 1



# Closing 
print("\n\n*******************************************************************")
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
print("*******************************************************************")
print("Total time (in hours): ", str( sum( iterationTime_historic )/3600 ) )
print("*******************************************************************\n\n")

#plt.show()  # Uncomment this line to make sure Python doesn't exit and close 
             # open figures before they were analyzed
#
#
#==== End of Calibration =======================================================