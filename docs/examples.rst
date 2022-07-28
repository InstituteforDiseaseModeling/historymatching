Examples
==============================

Prebuilt models
------------------------------

Prebuilt models for experimenting with are stored in :mod:`hm2.models` and include:

 * An SIR model based on the Gillespie’s tao-leap method (:mod:`hm2.models.SIRTaoLeap`).


Worked Example: Stochastic SIR
------------------------------

Here, we use Gillespie’s tao-leap method (:mod:`hm2.models.SIRTaoLeap`) to demonstrate the basic usage of the History Matching package.

.. include:: ../examples/stochastic_sir_example.py
   :literal: