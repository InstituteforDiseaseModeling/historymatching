Standard DataFrames
===================

The HistoryMatching library uses several standard Pandas DataFrames for
calculation. To use the library the data you pass to it must conform to these
dataframes.



Observations
---------------------

HistoryMatching matches a model's outputs to observations. There are two types
of observations: those tied to specific timepoints and those which represent
some aggregation of time points. Each has a specific DataFrame associated with
it.



ObservationsFrame
~~~~~~~~~~~~~~~~~~~~~

An ObservationsFrame_ contains all observations tied to specific time points.
Each observation must have a unique `observation_id` which is associated with
the `time` (e.g. `5` seconds) a particular `observation` (e.g. `mosquito_count`)
was made. The observation itself must have a `value` (e.g. `300`) and an
uncertainty expressed as a standard deviation `stdev` (e.g. `5`). Exact values
have an uncertainty of `0` and `time` must be monotonically increasing.

Each observation must occur only once at each time point. In other words,
`(time,observation)` is a unique key.

In the future, a special `time` value will denote a summary value; one that is
calculated outside of time. An example might be the total number of individuals
infected throughout the simulation.

Both your actual observations and your model results must be in
ObservationsFrame_ s.

An example table is shown below.

::

    observation_id      observation  time  value stdev
                 0   mosquito_count     3   3000    30
                 1  people_infected     3     10     0
                 2   mosquito_count    15   1000    10
                 3  people_infected    15    100     2

The data is stored in the so-called "tidy format". This data format may, at
first glance, seem unwieldy. Another "wide" format may seem preferable, e.g.:

::

    observation_id  time  mosquito_count mosquito_count_stdev people_count people_count_stdev
                 0     3            3000                   30           10                  0
                 1    15            1000                   10          100                  2

However, experience has shown that this format can create significant
difficulties in data processing. For instance, is `mosquito_count_stdev` a
observation, or the uncertainty in the `mosquito_count` observation? It is
better that you, the user, perform this conversion correctly than hope that we
are able to guess your intentions!

Hadley Wickham writes in detail about the benefits of the tidy format `here
<https://www.jstatsoft.org/index.php/jss/article/view/v059i10/v59i10.pdf>`_.



ParameterInfoFrame
------------------

This DataFrame is used to specify the names of parameters used by a WrappedModel
as well as the their minimum and maximum values. For a model with parameters
`beta` and `gamma`, the frame has the following form:

::

     name       min   max
     beta  0.000001  0.01
    gamma  0.000001  0.50



ParameterSamplesFrame
---------------------

HistoryMatching explores a parameter space by sampling it. Samples to be
explored are stored in a ParameterSamplesFrame_. For a model with
parameters `beta` and `gamma`, the frame would look like follows:

::

    param_id      beta     gamma
            0  0.004407  0.316147
            1  0.005409  0.433025
            2  0.003196  0.123237
            3  0.006439  0.280810
            4  0.007980  0.050390
          ...       ...       ...
           95  0.008666  0.483285
           96  0.006346  0.264908
           97  0.001813  0.036054
           98  0.000379  0.229818
           99  0.000878  0.116639

Note that all values in the `param_id` column are unique.



SimFrame
---------------------------------------

A SimFrame_ is like an ObservationsFrame_ except that it includes a `replicate`
and `param_id` column, like so:

::

        time      observation       value  stdev  observation_id  replicate  param_id
    0.000000      susceptible  190.000000      0               0          0       0.0
    0.000000       prevalence    5.000000      0            2376          0       0.0
    0.000000         infected   10.000000      0             396          0       0.0
    0.000000  per_susceptible   95.000000      0            1188          0       0.0
    0.000000     per_infected    5.000000      0            1584          0       0.0
         ...              ...         ...    ...             ...        ...       ...
  100.795317        recovered  191.000000      0            1187          0       0.0
  100.795317     per_infected   10.747664      0            1979          0       0.0
  100.795317         infected   23.000000      0             791          0       0.0
  100.795317  per_susceptible    0.000000      0            1583          0       0.0
  100.795317       prevalence   10.747664      0            2771          0       0.0

`run_replicates()` can be used to generate such a series of SimFrame_.



MatchedFrame
------------------------------------------

HistoryMatching requires matching simulated observations to actual observations
in order for the emulator to learn to approximate the model. This process is
fairly standard and so has been encapsulated in the
`match_sim_outputs_to_observations()` function. For time observations,  the
actual observation is identified as the one occuring closest in time to the
modeled observation.

The MatchedFrame_ consists of a `observation_id_a` column which is the
`observation_id` of the actual observation, as listed in an ObservationsFrame_.
The actual observation is associated with a `value` and its `stdev` as produced
by the simulation. The simulation might have run multiple times, as indicated by
the `replicate` column. The simulation may also have been run for different
parameters, as indicated by the `param_id` column which refers to the eponymous
column in a ParameterSamplesFrame_.

::

     observation_id_a  value  stdev  replicate  param_id
                    0    2.5      0          0       0.0
                    1    3.5      0          0       0.0
                    0    4.5      0          1       0.0
                    1    8.0      0          1       0.0
                    0    6.0      0          0       1.0
                  ...    ...    ...        ...       ...
                    1    5.0      0          1      48.0
                    0    9.5      0          0      49.0
                    1    1.5      0          0      49.0
                    0    3.5      0          1      49.0
                    1    0.0      0          1      49.0
