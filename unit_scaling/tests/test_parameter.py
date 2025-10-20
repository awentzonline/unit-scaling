# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import copy
import pickle

import torch
from torch import Tensor, empty, full, tensor
from torch.testing import assert_close
import pytest

from ..parameter import Parameter, UmupParameter, has_parameter_data


TORCH_VERSION = torch.torch_version.TorchVersion(torch.__version__)


def test_parameter() -> None:
    param = Parameter(torch.zeros(10), "weight")
    _test_parameter(param)


def _test_parameter(param) -> None:
    assert has_parameter_data(param)
    assert param.mup_type == "weight"
    assert param.mup_scaling_depth is None

    param.mup_scaling_depth = 8

    param_copy = copy.deepcopy(param)
    assert has_parameter_data(param_copy)  # type:ignore[arg-type]
    assert param_copy.mup_type == "weight"
    assert param_copy.mup_scaling_depth == 8

    param_pickle = pickle.loads(pickle.dumps(param))
    assert has_parameter_data(param_pickle)
    assert param_pickle.mup_type == "weight"
    assert param_pickle.mup_scaling_depth == 8

    param_pickle_copy = copy.deepcopy(param_pickle)
    assert has_parameter_data(param_pickle_copy)  # type:ignore[arg-type]
    assert param_pickle_copy.mup_type == "weight"
    assert param_pickle_copy.mup_scaling_depth == 8


def test_parameter_compile() -> None:
    parameter = Parameter(empty(3), mup_type="norm")
    _test_parameter_compile(parameter)


def _test_parameter_compile(parameter) -> None:
    def update_parameter(mult: Tensor) -> Tensor:
        parameter.data.mul_(mult)
        return parameter

    parameter.data.fill_(1)
    assert_close(update_parameter(tensor(123.0)), full((3,), 123.0))

    parameter.data.fill_(1)
    update_parameter = torch.compile(fullgraph=True)(update_parameter)
    assert_close(update_parameter(tensor(0.5)), full((3,), 0.5))
    assert_close(update_parameter(tensor(8.0)), full((3,), 4.0))
    assert_close(update_parameter(parameter), full((3,), 16.0))


@pytest.mark.skipif(TORCH_VERSION < "2.8", reason="Requires PyTorch >= 2.8")
def test_umup_parameter() -> None:
    param = UmupParameter(torch.zeros(10), "weight")
    _test_parameter(param)


@pytest.mark.skipif(TORCH_VERSION < "2.8", reason="Requires PyTorch >= 2.8")
def test_umup_parameter_compile() -> None:
    parameter = UmupParameter(empty(3), mup_type="norm")
    _test_parameter_compile(parameter)
