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


Outputs: 

    x           : Pandas DataFrame containing 'nSamples' rows, one row per set
                  of parameter generated.

"""
import numpy
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')

from pyDOE import lhs




def sampleParameters( xInfo, nSamples ) :

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


    # Finalize and return
    return x
#
# End of sampleParameters()
#-------------------------------------------------------------------------------
