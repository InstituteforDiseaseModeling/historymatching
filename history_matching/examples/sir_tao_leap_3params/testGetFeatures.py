# testGetFeatures
#
import numpy
import scipy
import pandas

from getFeatures import getFeatures




z = numpy.array([[19.9,  10.15, 5],
                 [20.1,  10.99,  800.02]
                ]
                )   # parameters

ref = numpy.array([1, 2, 3, 4])  # observations

# x = numpy.array( [ [0, 1, 2, 3],     # simulations
                   # [4, 5, 6, 7] 
                 # ]
                # )
                
x = numpy.array( [ [0, 1, 4, 27],     # simulations
                   [1, 0, 1, 0] 
                 ]
                )                
                
                
getFeatures( x, ref, z )

