""" sampleObservations

Sample observations from (an observations) DataFrame.

Arguments: 

    y           : Pandas DataFrame
    nSamples    : Number of samples to extract from  y
    samplingMode: Type of sampling. It can be any of the following:
                      'max'   :  Returns the sample with the maximum value. Only
                                 one sample is returned.
                      'random':  Random sampling.       

Output: 

    ySampled    : Subset of y containing nSamples rows (or the maximum number of
                  rows of y if nSamples < len(y)

"""
import pandas
import random




def sampleObservations( y, nSamples, samplingMode='random' ):

    # Initialization
    if ( ( samplingMode != "random" ) and ( samplingMode != "max" )  ):
        print("WARNING: samplingMode=", samplingMode, " not supported.",
              " Using samplingMode='random'")


    # Sampling
    if ( samplingMode == "max" ):
        ySampledTransposed = pandas.DataFrame( y.loc[ y['Incidence'].idxmax() ])
        ySampled = ySampledTransposed.transpose()
        
    
    else:  # samplingMode == "random"
        if ( nSamples <= len(y) ):
            ySampled = y.sample( n = nSamples )
        else:
            ySampled = y


    # Finalize and return
    return ySampled
#
# End of sampleObservations()
#-------------------------------------------------------------------------------
