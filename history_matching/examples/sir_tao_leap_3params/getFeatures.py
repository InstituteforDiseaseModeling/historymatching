# Python libraries (native, 3rd party)
import numpy
import math
import pandas




def getFeatures( x, ref ):

    # Initialization
    n = len(x)
    nObs = len(ref)
    featureOffset = 4
    nFeatures = featureOffset + nObs
    
    errorL1   = numpy.zeros(n)
    errorL2   = numpy.zeros(n)
    errorLinf = numpy.zeros(n)
    total     = numpy.zeros(n)
    
    featureMean = numpy.zeros(nFeatures)
    featureVar  = numpy.zeros(nFeatures)
    
    
    
    # Compute features
    for i in range(0, n):
    
        xCurrent = x[i][0:nObs]
    
        # Compute error norms
        errorL1  [i] = numpy.linalg.norm( (xCurrent-ref), ord=1 ) / nObs
        errorL2  [i] = numpy.linalg.norm( (xCurrent-ref), ord=2 ) / nObs
        errorLinf[i] = numpy.linalg.norm( (xCurrent-ref), ord=math.inf ) / nObs
        
        # Compute cumulative metrics
        total[i] = numpy.sum( xCurrent )
        
    

    # Compute statistics for features
    featureMean[0] = numpy.mean( errorL1   )
    featureMean[1] = numpy.mean( errorL2   )
    featureMean[2] = numpy.mean( errorLinf )
    featureMean[3] = numpy.mean( total     )
    
    featureVar[0] = numpy.var( errorL1   )
    featureVar[1] = numpy.var( errorL2   )
    featureVar[2] = numpy.var( errorLinf )
    featureVar[3] = numpy.var( total     )

    for i in range(0,nObs):
        thisFeature = numpy.zeros(n)
        for j in range(0,n):
            thisFeature[j] = x[j][i]
        featureMean[ featureOffset + i ] = numpy.mean( thisFeature )
        featureVar [ featureOffset + i ] = numpy.var ( thisFeature )
    


    # Prepare output
    features = pandas.DataFrame( { "errorL1"  : errorL1,
                                   "errorL2"  : errorL2,
                                   "errorLinf": errorLinf,
                                   "total"    : total,
                                  }
                                )
    for i in range(0, nObs):
        featureName = "t_" + str(i)
        thisFeature = numpy.zeros(n)
        for j in range(0,n):
            thisFeature[j] = x[j][i]
        
        features[featureName] = thisFeature


    # Finalize and return
    return features, featureMean, featureVar
