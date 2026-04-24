// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
#include <torch/extension.h>
#include <torch/library.h>
#include <torch/version.h>
#include <pybind11/pybind11.h>
#include "ops.h"
#include "utils.h"


TORCH_LIBRARY_EXPAND(_C, ops)
{
    // vLLM-MLU custom ops
    ops.def("weak_ref_tensor(Tensor input) -> Tensor");
    ops.impl("weak_ref_tensor", torch::kPrivateUse1, &vllm_mlu::weak_ref_tensor);
}

REGISTER_EXTENSION(_C)
