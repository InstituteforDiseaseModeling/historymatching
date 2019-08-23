""" getObservations

Get observations from a csv file containing simulated Seattle Flu data

Arguments:

    inputDataFile: csv file that includes, among other, the following columns: 
                     0   (A): individual
                     9   (J): residence_cra_name
                     10  (K): residence_neighborhood_district_name
                     12  (M): pathogen
                     13  (N): encountered_date
                     
    pathogen:      string containing the pathogen of interest. All rows 
                   containing this patogen will be returned.
             
    verbose:       Boolean selection of verbose mode.
    
Output:

    observations: Pandas DataFrame with selected columns of interest (for the 
                  SIR Tao-Leap calibration example) for all the rows containing
                  the pathogen indicated in the arguments. 

"""
import numpy
import math
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')




def getObservations( inputDataFile, pathogen, verbose=False ):


    # Read file and extract columns of interest
    inputData = pandas.read_csv( inputDataFile )
    inputSummary = pandas.DataFrame( inputData )
    summaryColumns = [0,9,12,13]
    inputSummary = inputSummary[ inputSummary.columns[summaryColumns] ]

    if verbose:
        print("inputData\n");     print( inputData.head(5)  );    print("\n")
        print("inputSummary\n");  print( inputSummary.head(5) );  print("\n")



    # Get general information from inputSummary
    inputSummary['encountered_date']                                           \
             = pandas.to_datetime( inputSummary['encountered_date'] )
    firstDay = inputSummary['encountered_date'].min()
    lastDay  = inputSummary['encountered_date'].max()
    nDays    = (lastDay - firstDay).days

    if verbose:
        print(" firstDay:   ", firstDay )
        print(" nDays:      ", nDays)
        print("")


    # Extract incidence per pathogen
    inputSummary['days_since_firstDay']    \
                = inputSummary.encountered_date.map( lambda x: x - firstDay )
    inputSummary_currentPathogen    \
                = inputSummary[ inputSummary['pathogen'] == pathogen ]
    nEntries_currentPathogen = len(inputSummary_currentPathogen)
    
    incidence = numpy.zeros( nDays )    
    for j in range (0, nEntries_currentPathogen) :
    
        currentDay    \
              = inputSummary_currentPathogen.iloc[j]['days_since_firstDay'].days
        incidence[currentDay] = incidence[currentDay] + 1

    incidence = numpy.trim_zeros( incidence )
    nDays_currentPathogen = len( incidence )
        
        
    # Estimate standard deviation
    windowSize = 5
    window = numpy.ones(windowSize) / windowSize
    smoothedIncidence = numpy.convolve( incidence, window, mode='same' )
    stdDev = numpy.zeros(nDays_currentPathogen)
    for j in range (0, nDays_currentPathogen):
        stdDev[j] = math.sqrt( math.pow(incidence[j]-smoothedIncidence[j], 2) )

    #smoothedIncidence    \
    #      = scipy.signal.savgol_filter( incidence, windowSize, polynomialOrder )
        
        
    # Plot incidence
    x = range(0, nDays_currentPathogen)
    yMax = max(incidence) + 2
    plt.plot(x, incidence, 'b' )
    plt.plot(x, smoothedIncidence, 'r' )
    plt.legend(['incidence', 'incidence (smoothed)'])
    plt.title(pathogen)
    plt.ylabel('cases')
    plt.ylim(0, yMax)
    plt.grid(linestyle=':')
    plt.xlabel('days')
    #plt.show(block=False)
    plt.savefig("observations-incidence.png", bbox_inches="tight")


    # Prepare data for history_matching
    #
    #    history_matching works with Pandas DataFrames. The data for 
    #    history_matching is saved into a dataframe called observations
    #
    timesForHistoryMatching = numpy.arange(0, nDays_currentPathogen,dtype='int')
    
    #stdDev = numpy.ones(nDays_currentPathogen) 
    observations = pandas.DataFrame( { 'Times'    : timesForHistoryMatching,
                                       'Incidence': incidence,
                                       'StdDev'    : stdDev,
                                 } )


    # Finalize and return
    return observations
#
# End of getObservations()
#-------------------------------------------------------------------------------
