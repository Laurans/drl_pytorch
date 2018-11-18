import pytest
import unittest
from core.utils.params import MemoryParams
from core.memories import Memory
import numpy as np


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.memory_params = MemoryParams(0)
        self.memory = Memory("test_mem", self.memory_params)

    def test_missing_params_memory(self):
        with pytest.raises(TypeError):
            Memory()

    def test_missing_params_memory(self):
        with pytest.raises(TypeError):
            Memory(self.memory_params)

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.memory.sample(1)
