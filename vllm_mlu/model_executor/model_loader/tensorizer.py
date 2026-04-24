# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import time
import torch
from torch import nn
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

from vllm.model_executor.model_loader.tensorizer import (
    TensorizerConfig, TensorDeserializer, TensorizerArgs,
    _check_tensors_on_meta_device, _resize_lora_embeddings,
    is_valid_deserialization_uri)
from vllm.platforms import current_platform
from vllm.logger import init_logger

try:
    from tensorizer.stream_io import open_stream
    from tensorizer.utils import (convert_bytes, get_mem_usage,
                                  no_init_or_tensor)

except ImportError:
    open_stream = tensorizer.placeholder_attr("stream_io.open_stream")
    convert_bytes = tensorizer.placeholder_attr("utils.convert_bytes")
    get_mem_usage = tensorizer.placeholder_attr("utils.get_mem_usage")
    no_init_or_tensor = tensorizer.placeholder_attr("utils.no_init_or_tensor")

logger = init_logger(__name__)


def deserialize_tensorizer_model(model: nn.Module,
                                 tensorizer_config: TensorizerConfig) -> None:
    tensorizer_args = tensorizer_config._construct_tensorizer_args()
    if not is_valid_deserialization_uri(tensorizer_config.tensorizer_uri):
        raise ValueError(
            f"{tensorizer_config.tensorizer_uri} is not a valid "
            f"tensorizer URI. Please check that the URI is correct. "
            f"It must either point to a local existing file, or have a "
            f"S3, HTTP or HTTPS scheme.")
    before_mem = get_mem_usage()
    start = time.perf_counter()
    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: use mlu device
    '''
    device = ''
    if current_platform.is_out_of_tree():
        device = f'mlu:{torch.mlu.current_device()}'
    elif current_platform.is_xpu():
        device = f'xpu:{torch.xpu.current_device()}'
    else:
        device = f'cuda:{torch.cuda.current_device()}'
    with open_stream(
            tensorizer_config.tensorizer_uri,
            mode="rb",
            **tensorizer_args.stream_kwargs) as stream, TensorDeserializer(
                stream,
                dtype=tensorizer_config.dtype,
                device=device,
                **tensorizer_args.deserialization_kwargs) as deserializer:
        deserializer.load_into_module(model)
        end = time.perf_counter()
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    total_bytes_str = convert_bytes(deserializer.total_tensor_bytes)
    duration = end - start
    per_second = convert_bytes(deserializer.total_tensor_bytes / duration)
    after_mem = get_mem_usage()
    deserializer.close()
    logger.info("Deserialized %s in %0.2fs, %s/s", total_bytes_str,
                end - start, per_second)
    logger.info("Memory usage before: %s", before_mem)
    logger.info("Memory usage after: %s", after_mem)

    _check_tensors_on_meta_device(model)
    _resize_lora_embeddings(model)
    del model.vllm_tensorized_marker

def serialize_extra_artifacts(
        tensorizer_args: TensorizerArgs,
        served_model_name: Union[str, list[str], None]) -> None:
    if not isinstance(served_model_name, str):
        raise ValueError(
            f"served_model_name must be a str for serialize_extra_artifacts, "
            f"not {type(served_model_name)}.")

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: use local file
    '''
    import shutil
    from pathlib import Path
    local_model_path = Path(served_model_name)
    if not local_model_path.exists() or not local_model_path.is_dir():
        raise ValueError(
            f"served_model_name must be a valid local directory in offline mode, "
            f"but got: {served_model_name}"
        )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''

    with tempfile.TemporaryDirectory() as tmpdir:
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: copy local file
        '''
        logger.info("Copying local model from %s to temporary directory %s", 
                    local_model_path, tmpdir)
        shutil.copytree(local_model_path, tmpdir, dirs_exist_ok=True)
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        for artifact in os.scandir(tmpdir):
            if not artifact.is_file():
                continue
            with open(artifact.path, "rb") as f, open_stream(
                    f"{tensorizer_args.tensorizer_dir}/{artifact.name}",
                    mode="wb+",
                    **tensorizer_args.stream_kwargs) as stream:
                logger.info("Writing artifact %s", artifact.name)
                stream.write(f.read())
    
