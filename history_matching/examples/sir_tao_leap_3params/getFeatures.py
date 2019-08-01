""" getFeatures

Compute and extract features (i.e., summary statistics) from a set of 
simulations. This is done by calling the function called "getFeatures". This
function also computes some statistics on the features. 
"""
# Python libraries (native, 3rd party)
import numpy
import scipy
import math
import pandas

import minepy   # Needed for computing multiple correlation metrics MIC and GMIC




# Auxiliary functions ----------------------------------------------------------
def fanoFactor(x):
    numpy.seterr(divide="ignore", invalid="ignore")
    y = numpy.divide( numpy.var(x), numpy.mean(x) ) 
    numpy.seterr(divide="warn", invalid="warn")
    return y

    
def multipleCorrelation(y,x):
    mcor = numpy.nan    # will be computed if no error conditions are detected
    m = numpy.shape(x)[1]
    c = numpy.zeros(m)
    numpy.seterr(invalid="ignore")

    for i in range(0,m):
        #c[i] = numpy.correlate( y, x[:,i] ) / (numpy.std(y)*numpy.std(x[:,i]))
        c[i] = scipy.stats.pearsonr( y, x[:,i] )[0] 

    numpy.seterr(invalid="warn")
    mcor = numpy.dot( c, c )
    return mcor
    
    
def multipleCorrelationMIC(y,x):
    mcor = numpy.nan    # will be computed if no error conditions are detected
    m = numpy.shape(x)[1]
    c = numpy.zeros(m)
    numpy.seterr(invalid="ignore")

    for i in range(0,m):
        mine = minepy.MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score( x[:,i], y )
        c[i] = mine.mic() 

    numpy.seterr(invalid="warn")
    mcor = numpy.amax( numpy.absolute(c) )
    return mcor


def multipleCorrelationGMIC(y,x):
    mcor = numpy.nan    # will be computed if no error conditions are detected
    m = numpy.shape(x)[1]
    c = numpy.zeros(m)
    numpy.seterr(invalid="ignore")

    for i in range(0,m):
        mine = minepy.MINE(alpha=0.6, c=15, est="mic_approx")
        mine.compute_score( x[:,i], y )
        c[i] = mine.gmic() 

    numpy.seterr(invalid="warn")
    mcor = numpy.dot(c,c)
    return mcor


def quartileCoefficient(x):
    numpy.seterr(divide="ignore", invalid="ignore")
    y = numpy.divide( numpy.percentile(x,75) - numpy.percentile(x,25),
                      numpy.percentile(x,75) + numpy.percentile(x,25) 
                     )
    numpy.seterr(divide="warn", invalid="warn")
    return y
    

def relativeStandardDeviation(x):
    numpy.seterr(divide="ignore", invalid="ignore")
    y = numpy.divide( numpy.sqrt( numpy.var(x) ), numpy.mean(x) )
    numpy.seterr(divide="warn", invalid="warn")
    return y
#
# End of auxiliary functions ---------------------------------------------------



 
# getFeatures ------------------------------------------------------------------
def getFeatures( x, ref, z=numpy.nan ):
    """ getFeatures

    Compute and extract features (i.e., summary statistics) from a set of 
    simulations. This is done by calling the function called "getFeatures". This
    function also computes some statistics on the features. 

    Arguments:
    
        x          : List containing simulation results. Each element i of this 
                     list contains a numpy array with the incidence obtained 
                     when using the set of parameters z(i) (see description of
                     z below).
        
        ref:       : Array with actual measurements (i.e., observations). This
                     argument is a numpy array. The length of the array is the 
                     same as the length of each element x(i). 
                 
        z:         : Array of parameters used for generating simulation results 
                     x. This argument is a numpy array. More precisely, it is 
                     a matrix where each row i contains the set of parameters 
                     used for generating x(i), and each column is a simulation
                     parameter.
        
    Output:

        features:  : Pandas dataframe containing the features extracted from the
                     simulation (i.e., from x, ref, and z).
                     
        stats:     : Pandas dataframe with statistics of "features". Includes
                     the mean, variance, as well as several dispersion metrics,
                     namely, the Fano factor, Relative Standard Deviation (RSD),
                     Quartile Coefficient of Disperssion (QCD), as well as
                     Multivariate correlation using the Pearson correlation 
                     coefficient, Mutual Information Coefficient (MIC), and 
                     Generalized Mutual Information Coefficient (GMIC).

    """

    # Constants
    MEAN     = 0
    VAR      = 1
    FANO     = 2
    RSD      = 3
    QCD      = 4
    SNR      = 5
    MCOR     = 6
    MCORMIC  = 7
    MCORGMIC = 8
    NMETRICS = 9


    # Initialization
    n = len(x)
    nObs = len(ref)
    featureOffset = 5
    nFeatures = featureOffset + nObs
    
    errorL1    = numpy.zeros(n)
    errorL2    = numpy.zeros(n)
    errorLinf  = numpy.zeros(n)
      
    total      = numpy.zeros(n)
    log10total = numpy.zeros(n)
       
    log10features = numpy.full(x.shape, numpy.nan)   
    
    featureStats = numpy.zeros( (NMETRICS,nFeatures) )   
                    # Feature analysis:
                    #   
                    #   [UNIVARIATE ANALYSIS]
                    #   0: mean     = \mu
                    #   1: variance = \sigma^2
                    #   2: fano     = \sigma^2 / \mu 
                    #   3: RSD (Relative Standard Deviation) = \sigma / \mu
                    #   4: QCD (Quartile Coefficient of Dispersion) 
                    #      = (Q3 - Q1)/(Q3+Q1)
                    #   5: SNR (Signal-to-Noise Ratio) = \mu /  \sigma 
                    #      (this is the reciprocal of the coefficient of 
                    #      variation, i.e., the RSD)
                    #
                    #   [MULTIVARIATE ANALYSIS - X vs z]
                    #   6: MCOR (Multiple Correlation - Pearson) 
                    #      = c_{x_i,z}^T \times c_{x_i,z}, where c is the 
                    #      Pearson correlation
                    #   7: MCORMIC (Multiple Correlation -MIC). This metric is 
                    #      computed using the Mutual Information Coefficient
                    #   8: MCORGMIC (Multiple Correlation -GMIC). This metric is
                    #      computed using the Generalized Mutual Information
                    #      Coefficient
                    #

    
    featureNames = ["errorL1", "errorL2", "errorLinf", "total", "log10(total)"]
    featureNamesExtended = []
    for i in range(0,nObs):
        featureNames.append( "t_" + str(i) )
        featureNamesExtended.append( "log10(t_" + str(i) + ")" )

   
    
    # Compute additional (secondary) features
    #
    
        # Auxiliary arrays for log(x) features
    nExtended = 2*nObs
    numpy.seterr(divide="ignore")
    log10features = numpy.log10(x)
    log10ref = numpy.log10(ref)
    numpy.seterr(divide="warn")
    xExtended = numpy.concatenate( (x, log10features), axis=1 )
    refExtended = numpy.concatenate( (ref, log10ref) )
    featureStatsExtended = numpy.zeros( (NMETRICS,nObs) )   
    

    
        # Error and total (cumulative) features
    for i in range(0, n):
    
        xCurrent = x[i][0:nObs]
    
        # Compute error norms
        errorL1  [i] = numpy.linalg.norm( (xCurrent-ref), ord=1 ) / nObs
        errorL2  [i] = numpy.linalg.norm( (xCurrent-ref), ord=2 ) / nObs
        errorLinf[i] = numpy.linalg.norm( (xCurrent-ref), ord=math.inf )
        
        # Compute cumulative metrics/features
        total[i] = numpy.sum( xCurrent )
        numpy.seterr(divide="ignore", invalid="ignore")
        log10total[i] = numpy.log10( total[i] )
        numpy.seterr(divide="warn", invalid="warn")
        #log10total[ log10total == -numpy.inf ] = -1e100 # Cap
                                               # log of 0 to a very large 
                                               # negative number; it could be
                                               # -sys.float_info.max, but this
                                               # would lead to overflows when 
                                               # computing statistics (e.g., 
                                               # mean)
    #
    # End of Compute additional (secondary) features



    # Compute statistics for features
    #   Statistics for error norms and total events
    featureStats[MEAN, 0] = numpy.mean( errorL1    )
    featureStats[MEAN, 1] = numpy.mean( errorL2    )
    featureStats[MEAN, 2] = numpy.mean( errorLinf  )
    featureStats[MEAN, 3] = numpy.mean( total      )
    featureStats[MEAN, 4] = numpy.mean( log10total )

    numpy.seterr(invalid="ignore")
    featureStats[VAR, 0] = numpy.var( errorL1    )
    featureStats[VAR, 1] = numpy.var( errorL2    )
    featureStats[VAR, 2] = numpy.var( errorLinf  )
    featureStats[VAR, 3] = numpy.var( total      )
    featureStats[VAR, 4] = numpy.var( log10total )
    numpy.seterr(invalid="warn")

    featureStats[FANO,0] = fanoFactor( errorL1    )
    featureStats[FANO,1] = fanoFactor( errorL2    )
    featureStats[FANO,2] = fanoFactor( errorLinf  )
    featureStats[FANO,3] = fanoFactor( total      )
    featureStats[FANO,4] = fanoFactor( log10total )

    featureStats[RSD, 0] = relativeStandardDeviation( errorL1    )
    featureStats[RSD, 1] = relativeStandardDeviation( errorL2    )
    featureStats[RSD, 2] = relativeStandardDeviation( errorLinf  )
    featureStats[RSD, 3] = relativeStandardDeviation( total      )
    featureStats[RSD, 4] = relativeStandardDeviation( log10total )
    
    featureStats[QCD, 0] = quartileCoefficient( errorL1    )
    featureStats[QCD, 1] = quartileCoefficient( errorL2    )
    featureStats[QCD, 2] = quartileCoefficient( errorLinf  )
    featureStats[QCD, 3] = quartileCoefficient( total      )
    featureStats[QCD, 4] = quartileCoefficient( log10total )

    featureStats[SNR, 0] = numpy.divide( 1, featureStats[RSD, 0] )
    featureStats[SNR, 1] = numpy.divide( 1, featureStats[RSD, 1] )
    featureStats[SNR, 2] = numpy.divide( 1, featureStats[RSD, 2] )
    featureStats[SNR, 3] = numpy.divide( 1, featureStats[RSD, 3] )
    featureStats[SNR, 4] = numpy.divide( 1, featureStats[RSD, 4] )
    
    # Multiple correlation 
    featureStats[MCOR,0] = multipleCorrelation( errorL1,    z )
    featureStats[MCOR,1] = multipleCorrelation( errorL2,    z )
    featureStats[MCOR,2] = multipleCorrelation( errorLinf,  z )
    featureStats[MCOR,3] = multipleCorrelation( total,      z )
    featureStats[MCOR,4] = multipleCorrelation( log10total, z )

    featureStats[MCORMIC,0] = multipleCorrelationMIC( errorL1,    z )
    featureStats[MCORMIC,1] = multipleCorrelationMIC( errorL2,    z )
    featureStats[MCORMIC,2] = multipleCorrelationMIC( errorLinf,  z )
    featureStats[MCORMIC,3] = multipleCorrelationMIC( total,      z )
    featureStats[MCORMIC,4] = multipleCorrelationMIC( log10total, z )
           
    featureStats[MCORGMIC,0] = multipleCorrelationGMIC( errorL1,    z )
    featureStats[MCORGMIC,1] = multipleCorrelationGMIC( errorL2,    z )
    featureStats[MCORGMIC,2] = multipleCorrelationGMIC( errorLinf,  z )
    featureStats[MCORGMIC,3] = multipleCorrelationGMIC( total,      z )
    featureStats[MCORGMIC,4] = multipleCorrelationGMIC( log10total, z )


    # Compute statistics for features
    #   Statistics for all (direct) events
    for i in range(0,nObs):
        thisFeature = numpy.zeros(n)
        thisFeatureExtended = numpy.zeros(n)
        for j in range(0,n):
            thisFeature[j] = x[j][i]
            thisFeatureExtended[j] = log10features[j][i]

        k = featureOffset + i
        featureStats[MEAN,k] = numpy.mean( thisFeature )
        featureStats[VAR ,k] = numpy.var ( thisFeature )
        featureStats[FANO,k] = fanoFactor( thisFeature )
        featureStats[RSD, k] = relativeStandardDeviation( thisFeature )
        featureStats[QCD, k] = quartileCoefficient( thisFeature )
        featureStats[SNR, k] = numpy.divide( 1, featureStats[RSD, k] )
        featureStats[MCOR,k] = multipleCorrelation( thisFeature, z )
        featureStats[MCORMIC, k] = multipleCorrelationMIC( thisFeature, z )
        featureStats[MCORGMIC,k] = multipleCorrelationGMIC( thisFeature, z )
        
        
        numpy.seterr(invalid="ignore")
        featureStatsExtended[MEAN,i] = numpy.mean( thisFeatureExtended )
        featureStatsExtended[VAR ,i] = numpy.var ( thisFeatureExtended )
        featureStatsExtended[FANO,i] = fanoFactor( thisFeatureExtended )
        featureStatsExtended[RSD, i] \
                              = relativeStandardDeviation( thisFeatureExtended )
        numpy.seterr(invalid="warn")
                              
        
        featureStatsExtended[QCD, i] = quartileCoefficient( thisFeatureExtended)
        featureStatsExtended[SNR, i] \
                              = numpy.divide( 1, featureStatsExtended[RSD, i] )
        featureStatsExtended[MCOR,i] \
                              = multipleCorrelation( thisFeatureExtended, z )
        featureStatsExtended[MCORMIC,i] \
                              = multipleCorrelationMIC( thisFeatureExtended, z )
        featureStatsExtended[MCORGMIC,i] \
                              = multipleCorrelationGMIC( thisFeatureExtended, z )
    

    # Prepare output
    features = pandas.DataFrame( { "errorL1"      : errorL1,
                                   "errorL2"      : errorL2,
                                   "errorLinf"    : errorLinf,
                                   "total"        : total,
                                   "log10(total)" : log10total,
                                  }
                                )
    for i in range(0, nObs):
        featureName = "t_" + str(i)
        thisFeature = numpy.zeros(n)
        for j in range(0,n):
            thisFeature[j] = x[j][i]
        
        features[featureName] = thisFeature

    for i in range(0, nObs):
        featureName = "log10(" + str(i) + ")"
        thisFeature = numpy.zeros(n)
        for j in range(0,n):
            thisFeature[j] = log10features[j][i]
            
        features[featureName] = thisFeature

    
    stats = pandas.DataFrame( { "feature" : featureNames + featureNamesExtended,
                                "mean"    : numpy.concatenate( \
                                              ( featureStats[MEAN],
                                                featureStatsExtended[MEAN] ) ),
                                "var"     : numpy.concatenate( \
                                              ( featureStats[VAR],
                                                featureStatsExtended[VAR] ) ),
                                "fano"    : numpy.concatenate( \
                                              ( featureStats[FANO],
                                                featureStatsExtended[FANO] ) ),
                                "rsd"     : numpy.concatenate( \
                                              ( featureStats[RSD],
                                                featureStatsExtended[RSD] ) ),
                                "qcd"     : numpy.concatenate( \
                                              ( featureStats[QCD],
                                                featureStatsExtended[QCD] ) ),
                                "snr"     : numpy.concatenate( \
                                              ( featureStats[SNR],
                                                featureStatsExtended[SNR] ) ),
                                "mcor"    : numpy.concatenate( \
                                              ( featureStats[MCOR],
                                                featureStatsExtended[MCOR] ) ),
                                "mcormic" : numpy.concatenate( \
                                              ( featureStats[MCORMIC],
                                                featureStatsExtended[MCORMIC] ) ),
                                "mcorgmic": numpy.concatenate( \
                                              ( featureStats[MCORGMIC],
                                                featureStatsExtended[MCORGMIC] ) ),
                               }
                             )


    # Finalize and return
    return features, stats
