""" sirTaoLeapTest

Script for testing the simulator sirTaoLip


"""
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')

from sirTaoLeap import sirTaoLeap


'''
# Model with invalid parameters
model_0 = sirTaoLeap( nDays=-1 )
'''


'''
# Model with default parameters
print( "----- Model 1 -----" )
model_1 = sirTaoLeap()
model_1.printArguments()
model_1.plot()
print( "" )
'''



# Model with custom paramters
print( "----- Model 2 -----" )
model_2 = sirTaoLeap( s0=500, i0 = 2, nDays=30, seed=1 )
model_2.printArguments()
s, i, r = model_2.simulate()
incidence = model_2.getIncidence()
model_2.plot()
model_2.plotIncidence()



print("END")
plt.show();
