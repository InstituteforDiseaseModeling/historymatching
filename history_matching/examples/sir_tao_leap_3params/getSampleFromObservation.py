# Python libraries (native, 3rd party)
import numpy
import math
import pandas


def getSampleFromObservation( y, pos ):

    sample = 0
    sampleVar = 1e-3
    offset = 4
    
    if ( pos >= offset ):  # Feature is a sample at time (pos-offset)
        sample    = y.iloc[ pos-offset, 1 ]
        sampleVar = y.iloc[ pos-offset, 2 ]
    
    elif ( pos == 3 ):  # Feature is the sum of all observations
        sample    = numpy.sum( y.iloc[:,1].values )
        sampleVar = numpy.sum( y.iloc[:,2].values ) # Is this math correct?
    
    else:   # Do nothing --mean and variance of an error norm should be 0
        pass

    return sample, sampleVar