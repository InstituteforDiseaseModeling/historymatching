WrappedModel (TODO)
==============================


WrappedModelResults
-------------------

When your model is run, it must return a tuple with two entries corresponding to
a `TimeObservationsFrame` and a `SummaryObservationsFrame`, though either entry
can be `None` as well (if say, there are now summary observations).