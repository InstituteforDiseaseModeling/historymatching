import numpy
import pandas


def selectFeature( features, 
                   featureStats, 
                   featureSelectionMetric, 
                   featureBlacklist,    # [index, timeLeft] pairs
                   closeCorrelationThreshold = 0.9,
                   timeToLiveInBlacklist = 1000,
                   debug = False
                  ):

    # Temp stuff
    #featureBlacklist.append([5,100])
    #featureBlacklist.append([6,1])
    print("!!!!!!  INITIAL BLACKLIST !!!!!!")
    print(featureBlacklist)
    print("________________________________")

    # Initialization
    nFeatures = len(features.columns)
    

    # Sort (in reverse order) feature stat metric 
    sortedFeatureIndex \
                 = numpy.argsort( -featureStats[featureSelectionMetric].values ) 


    # Find best feature 
    for i in range(0, 10): #nFeatures):
        incumbentIndex = i
        incumbentLoc = sortedFeatureIndex[ incumbentIndex ]
        incumbentCor = features.corr(method="pearson").iloc[:,incumbentLoc]
        acceptIncumbent = True

        if (debug):
            print( "incumbentIndex = ", i )
            print( "incumbentLoc = ", incumbentLoc )

        for (iBlacklist, tBlacklist) in featureBlacklist:
        
            if (debug):
                print( "iBlacklist = ", iBlacklist )
                print( "tBlacklist = ", tBlacklist )
                print( "thisCor = ", incumbentCor.iloc[ iBlacklist ] )
        
            if ((incumbentCor.iloc[ iBlacklist ] > closeCorrelationThreshold ) \
                                             or ( incumbentLoc == iBlacklist )):
                acceptIncumbent = False
            
        if (acceptIncumbent):
            break


    # Accept incumbent and update blacklist
    featureLoc = incumbentLoc
    featureBlacklist_updated = []
    for (iBlacklist, tBlacklist) in featureBlacklist:  # age blacklist
    
        tBlacklist = tBlacklist-1
        if (tBlacklist > 0):
            featureBlacklist_updated.append( [iBlacklist, tBlacklist] )
    
    featureBlacklist_updated.append( [featureLoc, timeToLiveInBlacklist ] )
    featureBlacklist = featureBlacklist_updated
        

    print("=======================================")
    print("nFeatures = ", nFeatures)
    print( "incumbentLoc final = ", incumbentLoc )
    print("---------------------------------------")
    print(featureBlacklist)
    print("---------------------------------------")
    print( features.iloc[:,incumbentLoc] )
    print("---------------------------------------")
    print( features.corr(method="pearson").iloc[:,incumbentLoc] )
    print("---------------------------------------")
    #print( features.corr(features["errorL1"], method="pearson") )
    print( features.corr(method="pearson") )
    print("=======================================")
    
    
    
    print("!!!!!!  FINAL BLACKLIST !!!!!!")
    print(featureBlacklist)
    print("______________________________")
    
    
    
                 
    return featureLoc, featureBlacklist
    
    
    
    
#     features.iloc[:, sortedFeatureIndex[0] ]