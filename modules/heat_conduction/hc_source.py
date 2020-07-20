import sys
import numpy as np

from source import SourceBase

class HeatConductionSource(SourceBase):

  source_type = 'hc'

  def __init__(self, source_id=0, q=[]):
    super().__init__(source_id)
    assert q!=[], "q must be specified."
    self.q = q