""" sir_tao_leap_trajectorySelection

Selection of trajectories for the SIR Tao-Leap calibration example
"""

# Python libraries (native, 3rd party)
import pandas
import numpy
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')   # Using this on Windows 10 (?)
from collections import Counter


# Python libraries (internal)
from getObservations     import getObservations
from runModel_sirTaoLeap import runModel_sirTaoLeapIncidence
from dataframePlot       import dataframePlot




# Input data
inputDataFile = "./data/simulated_subject_database.csv"
pathogen      = 'h1n1pdm'
solutionFile  = \
"./SEAflu_simData_GaussianBasis_SIR_mcorgmic--20190731-183900/main/Candidates_for_iter10.csv"

historyFile  = \
"./SEAflu_simData_GaussianBasis_SIR_mcorgmic--20190731-183900/main/history.txt"

outputSuffix = "Gaussian_MCORGMIC"



# Other parameters
nSims = 10
percentileCutoff = 0.02
seed = 101010                 # Seed for random number generators
if seed:
    numpy.random.seed(seed=seed)


# Get observations
y = getObservations( inputDataFile, pathogen, False ) 
yArray = y['Incidence'].to_numpy()
nDays = len(y)
yArray_inv = numpy.zeros(len(yArray))
for j in range(0, len(yArray)):
    if (yArray[j] != 0):        
        yArray_inv[j] = 1/yArray[j] 


# Get parameter space
x = pandas.read_csv( solutionFile )
dataframePlot( x, ['beta', 'gamma'], solutionFile, 'params.png' )


# Compute error norm for each simulation
#nParamSets = len(x)
nParamSets = 30
error = numpy.zeros( nParamSets*nSims )
incidenceAll = []
for j in range(0,nSims):

    print("... running set of simulations ", j+1, "/", nSims)
    xCommon = { "i0": 2,   "r0": 0,   "p_sampling": 1,   "nDays": 224 }
    
    for index, xCurrent in x.iterrows():   
 
        if ( (index%10) == 0 ):
            print( "... index: ", index )
            
        if (index >= nParamSets):
            break
            
        # Run simulation on the current parameter set
        incidence = runModel_sirTaoLeapIncidence( x.iloc[[index],:], xCommon, "ts" )[0]       

        # Compute error (select an error metric)
        errorCurrent = numpy.linalg.norm( (yArray-incidence), ord=1 )
        #errorCurrent = numpy.linalg.norm( (yArray-incidence), ord=2 )
        #errorCurrent = numpy.linalg.norm( (yArray-incidence), ord=math.inf )
        
        # Relative error ----
        #relativeError = numpy.multiply( abs(yArray-incidence), yArray_inv )
        #relativeError = numpy.multiply( relativeError, relativeError )
        #relativeError = numpy.multiply( (yArray-incidence), yArray_inv )
        #errorCurrent  = numpy.sum( relativeError )
        #----

        # Save results and get ready for next iteration
        error[j*nParamSets + index] = errorCurrent/len(incidence)
        incidenceAll.append( incidence )



# Plot error 
sortedErrorIndex = numpy.argsort( error )
cutoffIndex = int( percentileCutoff*len(error) )

plt.figure()
plt.semilogy( error[ sortedErrorIndex ], 'r' )
plt.semilogy( [0, len(error)-1],                       \
              [error[sortedErrorIndex[cutoffIndex]],   \
               error[sortedErrorIndex[cutoffIndex]]], 
              'b--' 
             )
plt.title( "Error per simulation")
plt.xlabel( "Simulation" )
plt.legend( ["Error", "Cutoff"] )
#plt.show()
plt.savefig("ts_errorPerSimulation_"+outputSuffix, bbox_inches="tight")


# Get indices of the parameters that generate the lowest errors
topBeta = []
topGamma = []
topN = []
for j in range(0,cutoffIndex):
    k = sortedErrorIndex[j]
    topBeta.append( x.iloc[k%nParamSets]['beta'])
    topGamma.append( x.iloc[k%nParamSets]['gamma'] )
    topN.append( x.iloc[k%nParamSets]['N'] )


# Plot N cases with lowest error, and get the index to their corresponding 
# parameter set
N = 5
timeInDays = range(0,nDays)

for i in range(0,5):

    plt.figure()
    ax = plt.subplot(111)
    ax.plot( yArray, 'k', linewidth=1.8, zorder=(N+1) )
    legend = []
    legend.append( 'observations' )

    for j in range(0,N):
        k = sortedErrorIndex[i*N+j]
        ax.plot( incidenceAll[k], zorder=(N-j) )
        legend.append(      "beta = "                                          \
                            + "{:.4f}".format( x.iloc[k%nParamSets]['beta']  ) \
                       + "; gamma = "                                          \
                            + "{:.4f}".format( x.iloc[k%nParamSets]['gamma'] ) \
                       + "; N = "                                              \
                            + "{:.2e}".format( x.iloc[k%nParamSets]['N']     ) \
                      )
        
    plt.xlabel("Day")
    plt.ylabel("Incidence")
    if i==0:
        plt.title("Best Realizations")
    else:
        plt.title("Best Realizations (" + str(i*N) +" to " + str((i+1)*N) + ")")
    ax.legend( legend, loc="upper center", bbox_to_anchor=(1.45, 0.95), ncol=1 )
    #plt.show()
    plt.savefig("ts_bestRealizations_"+str(i)+"_"+outputSuffix, bbox_inches="tight")


# Plot parameters leading to lowest error
plt.figure()
plt.scatter( topBeta, topGamma, s=1, marker="." )
plt.title( "Parameters with lowest error (cutoff: "               \
               + str( int(percentileCutoff*len(error)) )          \
               + "/"                                              \
               + str( int(len(error)) )                           \
               + ")"                                              \
          )
plt.xlabel("beta")
plt.ylabel("gamma")
plt.xlim(-0.02, 0.52)
plt.ylim(-0.02, 1.02)
#plt.show()
plt.savefig("ts_paramsWithLowestError_gammaBeta_"+outputSuffix, bbox_inches="tight")

plt.figure()
plt.scatter( topBeta, topN, s=1, marker="." )
plt.title( "Parameters with lowest error (cutoff: "               \
               + str( int(percentileCutoff*len(error)) )          \
               + "/"                                              \
               + str( int(len(error)) )                           \
               + ")"                                              \
          )
plt.xlabel("beta")
plt.ylabel("N")
plt.xlim(-0.02, 0.52)
plt.ylim(-0.02, 100000.02)
#plt.show()
plt.savefig("ts_paramsWithLowestError_nBeta_"+outputSuffix, bbox_inches="tight")



# Plot parameters leading to lowest error
combos = list(zip(topBeta, topGamma))
weightCounter = Counter(combos)
weights = [ 10*weightCounter[(topBeta[i], topGamma[i])] \
            for i, _ in enumerate(topBeta) ]
plt.figure()
plt.scatter( topBeta, topGamma, s=weights )
plt.title( "Parameters with lowest error (cutoff: "               \
               + str( int(percentileCutoff*len(error)) )          \
               + "/"                                              \
               + str( int(len(error)) )                           \
               + ")"                                              \
          )
plt.xlabel("beta")
plt.ylabel("gamma")
plt.xlim(-0.02, 0.52)
plt.ylim(-0.02, 1.02)
#plt.show()
plt.savefig("ts_paramsWithLowestError_gammaBeta_B_"+outputSuffix, bbox_inches="tight")

combos = list(zip(topBeta, topN))
weightCounter = Counter(combos)
weights = [ 10*weightCounter[(topBeta[i], topN[i])] \
            for i, _ in enumerate(topBeta) ]
plt.figure()
plt.scatter( topBeta, topN, s=weights )
plt.title( "Parameters with lowest error (cutoff: "               \
               + str( int(percentileCutoff*len(error)) )          \
               + "/"                                              \
               + str( int(len(error)) )                           \
               + ")"                                              \
          )
plt.xlabel("beta")
plt.ylabel("N")
plt.xlim(-0.02, 0.52)
plt.ylim(-0.02, 100000.02)
#plt.show()
plt.savefig("ts_paramsWithLowestError_nBeta_B_"+outputSuffix, bbox_inches="tight")



# Closing 
#plt.show()  # Keep this here to make sure Python doesn't exit and close 
            # open figures before they were analyzed
#
# End of main script
#-------------------------------------------------------------------------------
