""" plotParams

Generate scatter plots of pairs of parameters defined in a Pandas DataFrame.

Arguments:

    parms:         DataFrame containing the set of parameters.
                     
    paramMin:      Numeric array of containing the lower bound for each 
                   parameter.

    paramMax:      Numeric array of containing the upper bound for each 
                   parameter.

    title:         String containing the title of the figure.
    
    filename:      String containing the name of the file that will contain the 
                   figure.
                   
    drawfigure:    Boolean selection for drawing the figure (default=False)


"""


import numpy
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')


def plotParams( params, 
                paramMin,
                paramMax,
                title,
                filename,
                drawfigure=False
               ) :

    nSamples = params.shape[0]
    nParams  = params.shape[1]
    paramNames = params.columns.values
    ax = numpy.zeros(nParams*nParams, dtype=int)

    plt.figure()
    #plt.suptitle(title)
    
    for i in range(0, nParams):

        x = params.ix[:,i].values
        for j in range(0, nParams):

            if j < i:  continue

            y = params.ix[:,j]

            #k = i*nParams + j + 1
            k  = j*nParams + i + 1
            plt.subplot( nParams+1, nParams, k )           
            plt.scatter( x, y )

            plt.xlabel( paramNames[i] )
            plt.ylabel( paramNames[j] )

            plt.xlim( paramMin[i], paramMax[i] )
            plt.ylim( paramMin[j], paramMax[j] )
            plt.grid(linestyle=':')

    # Plot histograms
    for i in range(0, nParams):
        
        x = params.ix[:,i].values
        k = nParams*nParams + i + 1
        plt.subplot( nParams+1, nParams, k )
        
        plt.hist( x, 
                  bins=50, #"sqrt",
                  range=(paramMin[i], paramMax[i]), 
                  density=False 
                 )
        plt.xlabel( paramNames[i] )
        #plt.ylabel( "Frequency" )
        
        plt.grid(linestyle=':')


    # Final formatting
    plt.tight_layout()
    plt.savefig( filename, 
                 bbox_inches="tight",
                 orientation="landscape",
                 papertype="A2"
                )

    if drawfigure:
        plt.show()
    else:
        plt.close()

    return
