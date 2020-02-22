""" Plots for history file in SIR Tao-Leap calibration example

"""
import pandas
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')



def historyPlot( historyFile, column, title, ylabel, outputFileName ):

    history = pandas.read_csv( historyFile, delim_whitespace=True )
    history = history.drop( history.index[0] )
    history = history.drop( '(s)', axis=1 )
    history.rename( columns={'Rejection' : 'Rejection Rate'}, inplace=True )

    plt.figure()
    #plt.plot( history[ column ] )
    plt.semilogy( history[ column ] )
    plt.title( title )
    plt.xlabel( "iteration" )
    plt.ylabel( ylabel )
    plt.grid( linestyle=':' )
    plt.savefig( outputFileName )

    return
