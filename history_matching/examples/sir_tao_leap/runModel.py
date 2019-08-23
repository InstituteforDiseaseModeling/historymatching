""" runModel

Run a simulation model on a set of parameters. his function creates a figure
with the simulation results. The figure is also saved on the current directory. 


Arguments: 

    modelName  : Name of simulation model to run (string). Only the following
                 model is currently supported:
                   'sirTaoLeap_betaGamma':  SIR tao-leap with beta and gamma
                                            parameters.

    x          : Pandas DataFrame with each row containing a set of parameters
                 to be used in simulations.

    iteration  : Iteration count. This value is used for labeling figures and 
                 output files.

    y          : Pandas DataFrame with observations. These observations are 
                 shown in an output figure.

    verbose    : Boolean selection of verbose mode.


Output: 

    ySimulated : Pandas DataFrame containing the results of the simulations.

"""
# Python libraries (native, 3rd party)
import numpy
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')   # Using this on Windows 10 (?)

# Python libraries (internal)
from sirTaoLeap import sirTaoLeap




def runModel( modelName, x, iteration, y, verbose=False, showFigure=False ):


    # Initialization
    nDays = 250
    if ( modelName != "sirTaoLeap_betaGamma" ):
        print("WARNING: modelName=", samplingMode, " not supported.",
              " Using modelName='sirTaoLeap_betaGamma'")


    plt.figure()
    timeInDays = range(0,nDays+1)


    # Run the model
    incidenceAll = []
    for j, thisX in x.iterrows():
   
        model = sirTaoLeap( beta = thisX['beta'] , 
                            gamma = thisX['gamma'],
                            s0=100000, i0=2, nDays=nDays 
                           )
        incidence = model.getIncidence()
        
        for k, value in enumerate( incidence ):        
            incidenceAll.append( [j, k, value] )
        
        # Plot simulations
        plt.plot( incidence )



    # Finalize plot and save figure to file
    for i, thisY in y.iterrows():       # Show observations
        
        plt.plot( thisY['Times'], thisY['Incidence'], 'ko' )
        plt.plot( [ thisY['Times'],  thisY['Times'] ],
                  [ thisY['Incidence']-2*thisY['StdDev'],
                    thisY['Incidence']+2*thisY['StdDev'] ],
                  'k-'
                 )

    plt.xlabel('days')
    plt.ylabel('incidence')
    title = "Simulation results at iteration " + str(iteration)
    plt.title( title )

    figureFileName = "simulation-iter-" + str(iteration) + ".png"
    plt.savefig( figureFileName, bbox_inches="tight")

    if showFigure:
        plt.show(block=False)




    # Finalize and return
    # Simulated results must be converted into a Pandas DataFrame
    ySimulated = pandas.DataFrame( incidenceAll, 
                                   columns=['Sample_Id', 'ObsTime', 'Incidence']
                                  )
    ySimulated['Sim_Id'] = ySimulated['Sample_Id']
    return ySimulated
#
# End of runModel()
#-------------------------------------------------------------------------------
    
    