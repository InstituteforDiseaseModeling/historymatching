""" sir_tao_leap_trajectorySelection

Selection of trajectories for the SIR Tao-Leap calibration example
"""

# Python libraries (native, 3rd party)
import pandas
import numpy
from scipy import stats
import math
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')   # Using this on Windows 10 (?)
from collections import Counter


# Python libraries (internal)
from getObservations    import getObservations
from sampleObservations import sampleObservations
from sirTaoLeap         import sirTaoLeap
from dataframePlot      import dataframePlot
from historyPlot        import historyPlot




# Input data
inputDataFile = "./data/simulated_subject_database.csv"
pathogen      = 'h1n1pdm'
solutionFile  = \
"../tempResults/incidence_fluSimulated_SIR-tao-leap--20190612-223448/main/Candidates_for_iter67.csv"
historyFile  = \
"../tempResults/incidence_fluSimulated_SIR-tao-leap--20190612-223448/main/history.txt"


# Other parameters
nSims = 20
percentileCutoff = 0.02
seed = 101010                 # Seed for random number generators
if seed:
    numpy.random.seed(seed=seed)


# Get observations
y = getObservations( inputDataFile, pathogen, False ) 
yArray = y['Incidence'].to_numpy()
nDays = len(y)
#yArray_inv = numpy.zeros(len(yArray))
yArray_inv = 0*numpy.ones(len(yArray))
for j in range(0, len(yArray)):
    if (yArray[j] != 0):        
        yArray_inv[j] = 1/yArray[j] 


# Get parameter space
x = pandas.read_csv( solutionFile )
dataframePlot( x, ['beta', 'gamma'], solutionFile, 'params.png' )


# Compute error norm for each simulation
nParamSets = len(x)
error = numpy.zeros( nParamSets*nSims )
incidenceAll = []
for j in range(0,nSims):

    print("... running set of simulations ", j+1, "/", nSims)
    
    for index, xCurrent in x.iterrows():   
 
        # Run simulation on the current parameter set
        model = sirTaoLeap( beta = xCurrent['beta'] , 
                            gamma = xCurrent['gamma'],
                            s0=100000, i0=2, nDays=nDays 
                           )
        incidence = model.getIncidence()
     
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
        #error[j*nParamSets + index] = error[index] + errorCurrent/nSims
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
plt.show()


# Get indices of the parameters that generate the lowest errors
topBeta = []
topGamma = []
for j in range(0,cutoffIndex):
    k = sortedErrorIndex[j]
    topBeta.append( x.iloc[k%nParamSets]['beta'])
    topGamma.append( x.iloc[k%nParamSets]['gamma'] )


# Plot N cases with lowest error, and get the index to their corresponding 
# parameter set
N = 10
timeInDays = range(0,nDays)

for i in range(0,5):

    plt.figure()
    plt.plot( yArray, 'k', linewidth=2.5 )
    legend = []
    legend.append( 'observations' )

    for j in range(0,N):
        k = sortedErrorIndex[i*N+j]
        plt.plot( incidenceAll[k] )
        legend.append(      "beta = "                                          \
                            + "{:.4f}".format( x.iloc[k%nParamSets]['beta']  ) \
                       + "; gamma = "                                          \
                            + "{:.4f}".format( x.iloc[k%nParamSets]['gamma'] ) \
                      )
        
    plt.xlabel("Day")
    plt.ylabel("Incidence")
    if i==0:
        plt.title("Best Realizations")
    else:
        plt.title("Best Realizations (" + str(i*N) +" to " + str((i+1)*N) + ")")
    plt.legend( legend )
    plt.show()


# Plot parameters leading to lowest error
plt.figure()
plt.scatter( topBeta, topGamma )
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
plt.show()




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
plt.show()


# Closing 
plt.show()  # Keep this here to make sure Python doesn't exit and close 
            # open figures before they were analyzed
#
# End of main script
#-------------------------------------------------------------------------------
