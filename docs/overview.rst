========
Overview
========

The |HM| package is designed for:
 
     * Disease modelers, fluent in one or more of the following programming/script languages:

         * C, MATLAB, Python, R
 
     * Model calibration experts, fluent in Python

Disease modelers
----------------
Disease modelers can configure and use |HM| to help:

    * quantify uncertainty in a model for better understanding of parameter uncertainty, importance, relations among, and effects of changes in parameter values. This allows for:

        * fitting observed data
        * validating a model by rendering outputs that match observed data
        * identifying relevant and non-relevant parameters
        * understanding and estimating parameter correlations
        * characterizing uncertainty of parameters to help know which parameter combinations or regions of parameter space could explain observed data
        * gaining a better understanding of disease transmission, causal processes, and interventions

     * calibrate a model where the (pseudo-) likelihood function may be unknown or hard to define in a model with heterogenous data. For example, combining timeseries with phylogenetic summary statistics, or mortality timeseries with incidence by age group. This allows for:

        * fitting observed data.

     * calibrate a model where it is difficult to guess an initial solution or a small parameter space in a model with high dimensionality. For example, when there are three or more parameters. This allows for:

         * obtaining quick initial data points (guesses) to feed into more efficient calibration methods.

Model calibration experts
-------------------------

Model calibration experts can configure and use |HM| to help:

    * customize and improve a calibration and uncertainty quantification framework so that new methods can be quickly integrated by modelers in their workflows. This allows for:

        * designing fast (reaching convergence quickly), efficient (use of costly simulations) model calibration and uncertainty algorithms to help identify and select useful summary statistics and relevant, active parameters
        * prototyping and integrating algorithms easily
        * moving quickly from prototypes to useful software for modelers
        * evaluating the performance of algorithms, such as: convergence, timing, and numerical performance and accuracy reports.
