""" Plots for Pandas DataFrames

"""
import numpy
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')



def dataframePlot( x, columns, title, fileName, showFigure=False ):
    """Scatter plot of the first two columns of x.

        Arguments: 
            x      : Input data (to be used in plots)
            columns: List of columns for plots
    """
    
    n = len(columns)
    plt.figure()
    
    if ( n == 2 ):

        xPlot = x[ columns[0] ]
        yPlot = x[ columns[1] ]
        plt.scatter( xPlot, yPlot )
        plt.xlabel( columns[0] )
        plt.ylabel( columns[1] )


    plt.title( title )
    plt.grid( linestyle=':' )
    plt.savefig( fileName )

    if showFigure:
        plt.show( block=False )


    return