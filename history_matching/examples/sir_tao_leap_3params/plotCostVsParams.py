""" plotCostVsParams

Generate scatter plots of pairs of parameters defined in a Pandas DataFrame.

Arguments:

    parms:         DataFrame containing the set of parameters.
                     
    paramMin:      Numeric array of containing the lower bound for each 
                   parameter.

    paramMax:      Numeric array of containing the upper bound for each 
                   parameter.

    title:         String containing the title of the figure.
    
    filename:      String containing the name of the file that will contain the 
                   figure.
                   
    drawfigure:    Boolean selection for drawing the figure (default=False)

"""


import numpy
import pandas
import seaborn
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')


def plotCostVsParams( params, 
                      cost,
                      paramMin,
                      paramMax,
                      title,
                      filename,
                      drawfigure=False
                     ) :

    nSamples = params.shape[0]
    nParams  = params.shape[1]
    paramNames = params.columns.values
    

    # hexbins (simple count)
    plt.figure()
    plt.suptitle( "Frequency", fontsize=8 )
    for i in range(0, nParams):

        x = params.ix[:,i].values
        
        for j in range(0, nParams):

            y = params.ix[:,j].values
            k  = j*nParams + i + 1
            plt.subplot( nParams, nParams, k )
            
            plt.hexbin( x, y, gridsize=20,
                        cmap="Reds",
                       )
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=5)
            
            if (j==nParams-1):
                plt.xlabel( paramNames[i], fontsize=6 )
            if (i==0):
                plt.ylabel( paramNames[j], fontsize=6 )
            plt.xlim( paramMin[i], paramMax[i] )
            plt.ylim( paramMin[j], paramMax[j] )
            plt.tick_params( axis="x", labelsize=5 )
            plt.tick_params( axis="y", labelsize=5 )

    # Final formatting
    plt.tight_layout()
    plt.savefig( "summary_hexbin_" + filename, 
                 bbox_inches="tight",
                 orientation="landscape",
                 papertype="A2"
                )

    if drawfigure:
        plt.show()
    else:
        plt.close()


    # hexbins with error
    plt.figure()
    plt.suptitle( title, fontsize=8 )
    for i in range(0, nParams):

        x = params.ix[:,i].values
        
        for j in range(0, nParams):

            y = params.ix[:,j].values
            k  = j*nParams + i + 1
            plt.subplot( nParams, nParams, k )
            
            plt.hexbin( x, y, C=cost, gridsize=20, 
                        bins="log", 
                        cmap="plasma",
                        reduce_C_function=numpy.amin
                       )
            cbar = plt.colorbar()
            cbar.ax.tick_params(labelsize=5)
            
            if (j==nParams-1):
                plt.xlabel( paramNames[i], fontsize=6 )
            if (i==0):
                plt.ylabel( paramNames[j], fontsize=6 )
                
            plt.xlim( paramMin[i], paramMax[i] )
            plt.ylim( paramMin[j], paramMax[j] )
            plt.tick_params( axis="x", labelsize=5 )
            plt.tick_params( axis="y", labelsize=5 )

    # Final formatting
    plt.tight_layout()
    plt.savefig( "summary_hexbin_weighted_" + filename, 
                 bbox_inches="tight",
                 orientation="landscape",
                 papertype="A2"
                )

    if drawfigure:
        plt.show()
    else:
        plt.close()



    # Pair plots
    plt.figure()
    seaborn.pairplot( params, 
                      markers = ".",
                      diag_kind = "kde" 
                     )
    #plt.show()
    plt.savefig( "summary_pairplot_" + filename, 
                 bbox_inches="tight",
                 orientation="landscape",
                 papertype="A2"
                )


    plt.figure()
    g = seaborn.PairGrid( params )
    g = g.map_diag( plt.hist, bins=50, 
                    histtype="stepfilled"
                   )
    g = g.map_offdiag( plt.scatter, s=1, marker="." )
    plt.savefig( "summary_pairplot_hist_" + filename, 
                 bbox_inches="tight",
                 orientation="landscape",
                 papertype="A2"
                )


    params_cost = pandas.DataFrame( params )
    params_cost["cost"] = cost
    plt.figure()
    g = seaborn.PairGrid( params_cost, 
                          hue="cost", 
                          vars = ["beta", "gamma", "N"], 
                          palette="plasma" )
    # g = seaborn.PairGrid( params_cost, 
                          # hue="cost", 
                          # vars = list(params.columns.values), 
                          # palette="plasma" )
    g = g.map( plt.scatter, s=1, marker="." )
    plt.savefig( "summary_pairplot_cost_" + filename, 
                 bbox_inches="tight",
                 orientation="landscape",
                 papertype="A2"
                )

    return


