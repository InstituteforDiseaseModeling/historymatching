import abc

class ModelBase(abc.ABC): # pragma: no cover
  @abc.abstractmethod
  def run(self):
    """Runs the simulation.

    Args:
      X - Data to be transformed
    """
    pass

  def print_parameters(self):
    """Print the parameters governing the model"""
    pass
