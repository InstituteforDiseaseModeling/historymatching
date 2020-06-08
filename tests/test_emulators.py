import unittest

from hm2.basis import IdentityBasis
from hm2.error import HistoryMatchingError
import hm2.emulators as emu
import hm2.data_validation as dv

class TestGLM_GPR_Emulator(unittest.TestCase):
    def test_bad_basis(self):
      basis = IdentityBasis(intercept=True)
      self.assertRaises(TypeError, emu.GLM_GPR_Emulator, glm_basis="bad basis", gpr_basis=basis)
      self.assertRaises(TypeError, emu.GLM_GPR_Emulator, glm_basis=basis, gpr_basis="bad basis")