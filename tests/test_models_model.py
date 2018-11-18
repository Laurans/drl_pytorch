import pytest
import unittest
from core.models import Model
from core.utils.params import ModelParams


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model_params = ModelParams(0)
        self.model = Model("test_model", self.model_params)

    def test_missing_params_memory(self):
        with pytest.raises(TypeError):
            Model()

    def test_missing_params_memory(self):
        with pytest.raises(TypeError):
            Model("self.model_params")

    def test_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.model._init_weights()
