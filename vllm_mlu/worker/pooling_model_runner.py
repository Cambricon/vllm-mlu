# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Type

from vllm.logger import init_logger
from vllm.worker.pooling_model_runner import (ModelInputForGPUWithPoolingMetadata,
                                              PoolingModelRunner)

from vllm_mlu.worker.model_runner import (MLUModelRunnerBase,
                                          ModelInputForMLUBuilder)

logger = init_logger(__name__)


class MLUPoolingModelRunner(PoolingModelRunner,
                            MLUModelRunnerBase[ModelInputForGPUWithPoolingMetadata]):
    _model_input_cls: Type[ModelInputForGPUWithPoolingMetadata] = (
        ModelInputForGPUWithPoolingMetadata)
    _builder_cls: Type[ModelInputForMLUBuilder] = ModelInputForMLUBuilder
