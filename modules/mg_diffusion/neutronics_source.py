import sys
import numpy as np

from source import SourceBase

class NeutronicsSource(SourceBase):

  source_type = 'neutronics'

  def __init__(self, source_id=0, q=[], q_precursor=[]):
    super().__init__(source_id)
    assert q!=[], "q must be specified."
    self.n_grps = len(q)
    self.q = np.atleast_1d(q)
    self.n_precursors = 0
    if q_precursor != []:
      self.n_precursors = len(q_precursor)
      self.q_precursor = np.atleast_1d(q_precursor)
