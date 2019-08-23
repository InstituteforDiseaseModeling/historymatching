""" selectFeatures

Select a suitable feature (from a set of features) for fitting using history 
matching. The feature selected is the one with the largest selection metric that
is not blacklisted.

Arguments:

    features         : Pandas dataframe containing the features extracted from
                       the simulation
                
    featureStats     : Pandas dataframe with statistics of "features". 

    featureSelectionMetric : Name of the metric to be used for feature 
                       selection. This argument is a string with the name
                       of one column in featureStats.

    featureBlacklist : List of features that have been blacklisted. This is a 
                       list of [index, timeLeft] pairs, where index is the 
                       column index in "features" and timeLeft is an integer 
                       that will decrease after the call to selectFeatures. 
                       Every feature returned by selectFeatures is added to 
                       featureBlacklist and its corresponding timeLeft is set
                       to the value indicated by the argument called 
                       "timeToLiveInBlacklist". A feature whose timeLeft value
                       decreases to 0 is removed from featureBlacklist.
                       
    closeCorrelationThreshold : Value between 0 and 1 that indicates what value
                       of (Pearson) correlation defines that 2 features are 
                       correlated. Features that are highly correlated to any
                       feature in featureBlacklist are not selected by 
                       selectFeatures.

    timeToLiveInBlacklist : Integer indicating the time that a selected feature
                       will remain in featureBlacklist (see description of 
                       featureBlacklist above).

Output:

    featureLoc       : Index of the selected feature in "features".
    
    featureBlacklist : Updated featureBlacklist list.
    
"""
import numpy
import pandas
import warnings


def selectFeature( features, 
                   featureStats, 
                   featureSelectionMetric, 
                   featureBlacklist,    # [index, timeLeft] pairs
                   closeCorrelationThreshold = 0.85,
                   timeToLiveInBlacklist = 1000
                  ):

    # Initialization
    nFeatures = len(features.columns)
    lastValidLoc = 1

    # Sort (in reverse order) feature stat metric 
    sortedFeatureIndex \
      = numpy.argsort( -numpy.abs(featureStats[featureSelectionMetric].values) ) 



    # Find best feature 
    for i in range(0, nFeatures):
        incumbentIndex = i
        incumbentLoc = sortedFeatureIndex[ incumbentIndex ]
        incumbentCor = features.corr(method="pearson").iloc[:,incumbentLoc]
        acceptIncumbent = True

        # Stop search if already in "nan" or "inf" region
        if ( not \
             numpy.isfinite( \
                   featureStats[featureSelectionMetric].values[incumbentLoc]   \
             ) 
            ):
            
            warnings.warn( "Unable to find valid feature" +    \
                           "(stopping search at position " +   \
                           str(i) +                            \
                           " of " +                            \
                           str(nFeatures) +                    \
                           " potential features)"              \
                          )
            
            
            incumbentLoc = sortedFeatureIndex[0]

        # Validate that incumbent is not on black list
        else:
            for (iBlacklist, tBlacklist) in featureBlacklist:

                if (   ( numpy.abs(incumbentCor.iloc[ iBlacklist ]) \
                                      > closeCorrelationThreshold ) \
                        or ( incumbentLoc == iBlacklist )
                    ):
                    acceptIncumbent = False

        # if (incumbentLoc > nFeatures/2):  # Do not work with log features
            # acceptIncumbent = False


        # Finalize search if the incumbent is valid
        if (acceptIncumbent):
            break
    #
    #---- end of "Find best feature" loop



    # Accept incumbent and update blacklist
    if ( not \
         numpy.isfinite( \
               featureStats[featureSelectionMetric].values[incumbentLoc] 
                        ) 
        ):
        featureLoc = lastValidLoc
    
    featureLoc = incumbentLoc
    featureBlacklist_updated = []
    for (iBlacklist, tBlacklist) in featureBlacklist:  # age blacklist
    
        tBlacklist = tBlacklist-1
        if (tBlacklist > 0):
            featureBlacklist_updated.append( [iBlacklist, tBlacklist] )
    
    featureBlacklist_updated.append( [featureLoc, timeToLiveInBlacklist ] )
    featureBlacklist = featureBlacklist_updated


    return featureLoc, featureBlacklist
