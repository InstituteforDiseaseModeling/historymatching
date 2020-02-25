""" runModel_sirTaoLeap

Event-driven simulation of an SIR model based on Gillespie's tao-leap method.

"""
# Python libraries (native, 3rd party)
import matplotlib
from matplotlib import pyplot as plt
matplotlib.pyplot.switch_backend('TKAgg')

# Python libraries (internal)
from sirTaoLeap import sirTaoLeap




def runModel_sirTaoLeapIncidence( x, xCommon, id, showFigure=False ):
    """
    Event-driven simulation of an SIR model based on Gillespie's tao-leap
    method. The output contains the incidence for each of the combinations in 
    in the dataframe of input parameters.

    Arguments: 

        x          : Pandas DataFrame with each row containing a set of 
                     parameters to be used in simulations.

        xCommon    : Dictionary containing the parameters that are common for 
                     all the simulations.

        id         : String that names the figure to be saved.

        showFigure : Boolean indicating if figure should be displayed on screen.

    Output: 

        ySimulated : Pandas DataFrame containing the results of the simulations.
    """

    # Model parameters (defaults)
    beta  = 0.25
    gamma = 0.25
    s0    = -1
    i0    = -1
    r0    = -1
    N     = -1
    p     = 0.01
    nDays = 500


    # Run the simulations for each set of parameters in x
    plt.figure()
    incidenceAll = []
    for j, thisX in x.iterrows():

        # Prepare model parameters
        s0 = -1
        i0 = -1
        r0 = -1
        N  = -1
        
        if "beta"       in xCommon:    beta  = xCommon["beta"]
        if "gamma"      in xCommon:    gamma = xCommon["gamma"]
        if "s0"         in xCommon:    s0    = xCommon["s0"]
        if "i0"         in xCommon:    i0    = xCommon["i0"]
        if "r0"         in xCommon:    r0    = xCommon["r0"]
        if "N"          in xCommon:    N     = xCommon["N"]
        if "p_sampling" in xCommon:    p     = xCommon["p_sampling"]
        if "nDays"      in xCommon:    nDays = xCommon["nDays"]
        
        if "beta"       in x.columns:    beta  = thisX["beta"]
        if "gamma"      in x.columns:    gamma = thisX["gamma"]
        if "s0"         in x.columns:    s0    = thisX["s0"]
        if "i0"         in x.columns:    i0    = thisX["i0"]
        if "r0"         in x.columns:    r0    = thisX["r0"]
        if "N"          in x.columns:    N     = thisX["N"]
        if "p_sampling" in x.columns:    p     = thisX["p_sampling"]
        if "nDays"      in x.columns:    nDays = thisX["nDays"]
        
        if   ( s0 < 0 ):  s0 = p*N - i0 - r0
        elif ( i0 < 0 ):  i0 = p*N - s0 - r0
        elif ( r0 < 0 ):  r0 = p*N - s0 - i0
        elif ( N  < 0 ):  N  = s0 + i0 + r0
        else           :  return -1  # Error condition
        if ( (s0<0) or (i0<0) or (r0<0)) : return -2  # Error condition
        
        
        # Run simulations
        model = sirTaoLeap( beta, gamma, int(s0), int(i0), int(r0), nDays)
        incidence = model.getIncidence()
        
        
        # Save results
        plt.plot( incidence )
        incidenceAll.append( incidence )
       
        
        
        
    # Finalize plot and save figure to file
    plt.xlabel('days')
    plt.ylabel('incidence')
    title = "Simulation results at " + str(id)
    plt.title( title )

    figureFileName = "simulation-" + str(id) + ".pdf"
    plt.savefig( figureFileName, bbox_inches="tight")
    figureFileName = "simulation-" + str(id) + ".png"
    plt.savefig( figureFileName, bbox_inches="tight")
    
    if showFigure:
        plt.show()
    else:
        plt.close()




    # Finalize and return
    return incidenceAll
    