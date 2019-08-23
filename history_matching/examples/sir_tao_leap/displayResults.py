""" displayResults

Generate plots summarizing results of SIR Tao-Leap calibration example
"""

# Python libraries (native, 3rd party)
import pandas
import numpy
from scipy import stats
import math
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')   # Using this on Windows 10 (?)


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

# Generate some initial plots: parameter space and execution time
# dataframePlot( x, ['beta', 'gamma'], "Iteration 67", 'params.png' )
# historyPlot( historyFile, 'Time', "Time per Iteration", "seconds", 
             # 'timePerIteration-log.png' )



# Compute error norm for each simulation
error = numpy.zeros( len(x) )
ySampled = sampleObservations( y, 1, "max" )
Nsims = 20
for j in range(0,Nsims):

    print("... running set of simulations ", j+1, "/", Nsims)
    incidenceAll = []
    for index, xCurrent in x.iterrows():   
 
        # Run simulation on the current parameter set
        model = sirTaoLeap( beta = xCurrent['beta'] , 
                            gamma = xCurrent['gamma'],
                            s0=100000, i0=2, nDays=nDays 
                           )
        incidence = model.getIncidence()
     
        # Compute error (select an error metric)
        #errorCurrent = numpy.linalg.norm( (yArray-incidence), ord=1 )
        #errorCurrent = numpy.linalg.norm( (yArray-incidence), ord=2 )
        #errorCurrent = numpy.linalg.norm( (yArray-incidence), ord=math.inf )
        
        # Relative error ----
        #relativeError = numpy.multiply( abs(yArray-incidence), yArray_inv )
        #relativeError = numpy.multiply( relativeError, relativeError )
        relativeError = numpy.multiply( (yArray-incidence), yArray_inv )
        errorCurrent  = numpy.sum( relativeError )
        #----


        # Save results and get ready for next iteration
        error[index] = error[index] + errorCurrent/Nsims
        incidenceAll.append( incidence )


# Find top sets of parameters and plot
sortedErrorIndex = numpy.argsort( error )
N = 10
timeInDays = range(0,nDays)
legend = []

plt.figure()
plt.plot( yArray, 'k', linewidth=2.5 )
legend.append( 'observations' )



for j in range(0,N):
    
    k = sortedErrorIndex[j]
    plt.plot( incidenceAll[k] )  
    legend.append(      "beta = " + "{:.4f}".format( x.iloc[k]['beta']  )     \
                   + "; gamma = " + "{:.4f}".format( x.iloc[k]['gamma'] )     \
                  )
plt.xlabel("day")
plt.ylabel("incidence")
plt.title("best models")
plt.legend( legend )



# Average realizations for the parameters
incidenceAverageFig = plt.figure()
plt.plot( yArray, 'k', linewidth=2.5 )

for j in range(0, N):

    averageIncidence = numpy.zeros( len(yArray) )
    incidenceThisSet = numpy.zeros( (Nsims, len(yArray)) )
    i = sortedErrorIndex[j]
    print(x.iloc[i]['beta'])
    
    for k in range(0, Nsims):
    
        model = sirTaoLeap( beta = x.iloc[i]['beta'] , 
                            gamma = x.iloc[i]['gamma'],
                            s0=100000, i0=2, nDays=nDays 
                           )
        incidence = model.getIncidence()

        incidenceThisSet[k,:] = incidence
        averageIncidence = averageIncidence + (1/Nsims)*incidence


    stdev = numpy.std( incidenceThisSet, axis=0 )
    interval = 0.95
    testStat = stats.t.ppf( (interval+1)/2, Nsims )
    upperBound = averageIncidence + testStat * stdev / math.sqrt(Nsims)
    lowerBound = numpy.maximum( numpy.zeros( len(averageIncidence) ), 
                                averageIncidence - testStat * stdev 
                                                   / math.sqrt(Nsims)
                               )

    
    plt.plot( averageIncidence )
    
    plt.figure()
    plt.plot( yArray )
    plt.plot( averageIncidence )
    plt.fill_between( range(0,len(lowerBound)), lowerBound, upperBound, 
                      facecolor="orange",
                      alpha=0.4
                     )
    plt.title( "beta = " + "{:.4f}".format( x.iloc[i]['beta']  )     \
               + "; gamma = " + "{:.4f}".format( x.iloc[i]['gamma'] )      \
              )
    plt.legend( ["observed", "simulated" ] )
    plt.xlabel("day")
    plt.ylabel("incidence")
    
    plt.figure( incidenceAverageFig.number )
    
    
    
        
plt.xlabel("day")
plt.ylabel("incidence")
plt.title("average of best models")
plt.legend( legend )

    




# Closing 
plt.show()  # Keep this here to make sure Python doesn't exit and close 
            # open figures before they were analyzed
#
# End of main script
#-------------------------------------------------------------------------------
