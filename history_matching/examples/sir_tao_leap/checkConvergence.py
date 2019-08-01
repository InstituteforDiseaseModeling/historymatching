""" Convergence metrics for history matching
"""
import numpy
import scipy
import pandas




def checkConvergence( nonImplausibleCandidates_dataFrame ):

    nonImplausibleCandidates = \
        nonImplausibleCandidates_dataFrame.drop( ['Implausible'], axis=1 )     \
        .to_numpy()

    D  = scipy.spatial.distance_matrix( nonImplausibleCandidates,
                                        nonImplausibleCandidates 
                                       )
    covD = numpy.cov( D )
         
    lambdaD    = scipy.sparse.linalg.eigsh( D,    k=1,
                                                  which='LM', 
                                                  return_eigenvectors=False
                                           )
    lambdaCovD = scipy.sparse.linalg.eigs ( covD, k=1,
                                                  which='LM', 
                                                  return_eigenvectors=False
                                           )
         
    return abs(lambdaD), abs(lambdaCovD)
