""" sampleParameters

Sample parameters for history_matching. This function creates a scatter plot
showing the selection of the first two parameters. plt.show() must be called
after a call to sampleParameters for the figure to be actually shown. The 
figure is, however, saved on the current directory. 


Arguments: 

    xInfo       : Pandas DataFrame with general the name and range of the 
                  parameters. It contains three columns: (A) Name, (B) Min, 
                  and (C) Max.
                  
    nSamples    : Number of samples to generate.
    
    iteration   : Iteration count. In general, sampling for history matching
                  may change depending on the iteration number. In this 
                  function, iteration is used only for labeling figures and file
                  names.
                  
    verbose     : Boolean selection of verbose mode.


Outputs: 

    x           : Pandas DataFrame containing 'nSamples' rows, one row per set
                  of parameter generated.

"""
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')

from pyDOE import lhs




def sampleParameters( xInfo, nSamples, iteration, verbose=False ) :

    xName = xInfo.index.values
    nParams = xInfo.shape[0]

    x = pandas.DataFrame( lhs(nParams, nSamples), columns=xName )
    x.index.name = 'Sample_Id'


    # Scale samples to parameter range
    for paramName, paramRange in x.iteritems():
    
        x[paramName] = xInfo.loc[paramName,'Min'] \
                       + paramRange*( xInfo.loc[paramName,'Max'] 
                                      - xInfo.loc[paramName,'Min'] 
                                     )


    # Plot samples
    plt.figure()
    plt.scatter( x=x[xInfo.index.values[0] ],  y=x[xInfo.index.values[1]] )
    plt.xlabel(xInfo.index.values[0])
    plt.ylabel(xInfo.index.values[1])
    title = "{beta, gamma} for iteration " + str(iteration)
    plt.title( title )
    plt.grid(linestyle=':')
    figureFileName = "parameters-iter-" + str(iteration) + ".png"
    plt.savefig( figureFileName, bbox_inches="tight")


    # Finalize and return
    return x
#
# End of sampleParameters()
#-------------------------------------------------------------------------------
