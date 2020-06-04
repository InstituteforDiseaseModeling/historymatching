.. role:: py(code)
    :language: py

Wrapping A Model
================

HistoryMatching requires that the model be wrapped in a special function which standardizes HistoryMatching's interaction with models, like so:

.. code-block:: python

    def wrapped_model(**kwargs):
        model_instance = MODEL(**kwargs)
        time_observations, summary_observations = model_instance.sim()
        wrapped_results = (time_observations, summary_observations)
        return wrapped_results

The function :py:`wrapped_model()` is passed by reference to History Matching. At appropriate times History Matching calls this function. When it does so :py:`kwargs` is a dictionary of keyword arguments specifying the parameter values which should be used to initialize the model. The function should:

  1. Create an instance of the model using the parameter values in :py:`kwargs`
  2. Run the model
  3. Possibly post-process the results of the model
  4. Return a valid :ref:`WrappedModelResults`.
