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



TimeObservationsFrame
~~~~~~~~~~~~~~~~~~~~~

A `TimeObservationsFrame` contains all observations tied to specific time
points. Each observation must have a unique `observation_id` which is associated
with the `time` (e.g. `5` seconds) a particular `observation` (e.g.
`mosquito_count`) was made. The observation itself must have a `value` (e.g.
`300`) and an uncertainty expressed as a standard deviation `stdev` (e.g. `5`).
Exact values have an uncertainty of `0`.

`time` must be monotonically increasing.

Each observation must occur only once at each time point. In other words,
`(time,observation)` is a unique key.

An example table is shown below.

::

    observation_id  time      observation    value stdev
                 0     3   mosquito_count     3000    30
                 1     3  people_infected       10     0
                 2    15   mosquito_count     1000    10
                 3    15  people_infected      100     2

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


SummaryObservationsFrame
~~~~~~~~~~~~~~~~~~~~~~~~

A `SummaryObservationsFrame` contains all observations which are not tied to
specific time points. Each observation has a name `observation` (e.g.
`cumulative_infections`). The observation itself must have a `value` (e.g.
`300`) and an uncertainty expressed as a standard deviation `stdev` (e.g. `5`).
Exact values have an uncertainty of `0`. The data is, again, arranged in tidy
format.

Each observation must occur only once in the table. In other words,
`observation` is a unique key.

An example table is shown below.

::

              observation    value stdev
    cumulative_infections     4500    50
       days_of_quarantine       10     0



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
explored are stored in a `ParameterSamplesFrame`. For a model with parameters
`beta` and `gamma`, the frame would look like follows:

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



TimeStandardAnalysisFrame
-------------------------

HistoryMatching requires matching simulated observations to actual observations
in order for the emulator to learn to approximate the model. This process is
fairly standard and so has been encapsulated in the `standard_analysis`
function. This function returns a `TimeStandardAnalysisFrame` with the following
tidy format:

::

    param_id replicate      observation value stdev  aobservation_id
           0         0   mosquito_count  3105     0                0
           0         0   mosquito_count  3092     0                1
           0         1   mosquito_count  3104     0                0
           0         1   mosquito_count  3093     0                1
           1         0   mosquito_count  3404     0                0
           1         0   mosquito_count  2993     0                1

Here, `param_id` identifies the parameter combination from a
`ParameterSamplesFrame` used to parameterize the model. The model might be run
many times for the same parameter combination, `replicate` indicates which one
of these repeates has been run, `observation` is the name of the modeled
observation and `stdev` is its uncertainty as a standard deviation.
`aobservation_id` indicates which of the actual observations the modeled
observation is to be compared against. In the `standard_analysis` the actual
observation is identified as the one occuring closest in time to the modeled
observation.



SummaryStandardAnalysisFrame
----------------------------

HistoryMatching requires matching simulated observations to actual observations
in order for the emulator to learn to approximate the model. This process is
fairly standard and so has been encapsulated in the `standard_analysis`
function. This function returns a `SummaryStandardAnalysisFrame` with the following
tidy format:

::

  param_id replicate      observation value stdev
         0         0  cumulative_infections  4532
         0         0     days_of_quarantine    12
         0         1  cumulative_infections  4498
         0         1     days_of_quarantine     8
         1         0  cumulative_infections  3700
         1         0     days_of_quarantine    57

Here, `param_id` identifies the parameter combination from a
`ParameterSamplesFrame` used to parameterize the model. The model might be run
many times for the same parameter combination, `replicate` indicates which time
has been run, `observation` is the name of the modeled observation (and its
matching actual observation) and `stdev` is its uncertainty as a standard
deviation. In the `standard_analysis` the actual observation is identified as
having the same name as the modeled observation.
