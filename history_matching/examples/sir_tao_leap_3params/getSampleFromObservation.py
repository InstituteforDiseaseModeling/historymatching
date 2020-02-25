""" getSampleFromObservation

Generate scatter plots of pairs of parameters defined in a Pandas DataFrame.

Arguments:

    y         : Observation data. This is a Pandas dataframe with columns of 
                interest (as needed for history matching), namely: times,
                incidence, and standard deviation.
                
    pos       : Index of the feature to be extracted (note that this is an 
                index based on the output of "getFeatures", not based on "y").
    
Output:

    sample    : Observation value (scalar).
    
    sampleVar : Variance of sample (scalar).

"""

# Python libraries (native, 3rd party)
import numpy


def getSampleFromObservation( y, pos ):

    sample = 0
    sampleVar = 1e-3
    offset = 5
    
    if ( pos >= offset ):  # Feature is a sample at time (pos-offset)
    
        if ( pos < (offset+len(y)) ) : # Feature is direct sample
            sample    = y.iloc[ pos-offset, 1 ]
            sampleVar = y.iloc[ pos-offset, 2 ]
        
        else:  # Feature is log
            sample    = numpy.log10( y.iloc[ pos-offset-len(y), 1 ] )
            sampleVar = numpy.log10( y.iloc[ pos-offset-len(y), 2 ] ) # Is this 
                            # correct?  There is probably a better approximation
            
    
    elif ( pos == 3 ):  # Feature is the sum of all observations
        sample    = numpy.sum( y.iloc[:,1].values )
        sampleVar = numpy.sum( y.iloc[:,2].values ) # Is this math correct?
        
    elif ( pos == 4 ):  # Feature is the lof of sum of all observations
        sample    = numpy.log10( numpy.sum( y.iloc[:,1].values ) ) 
        sampleVar = numpy.log10( numpy.sum( y.iloc[:,2].values ) )
        
    
    else:   # Do nothing --mean and variance of an error norm should be 0
        pass

    return sample, sampleVar