# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from typing import Any, TypeVar

from PIL import Image

from vllm import multimodal
from vllm.logger import init_logger
from vllm.multimodal.utils import MediaConnector


from vllm_mlu.mlu_hijack_utils import MluHijackObject


def vllm__multimodal__utils__fetch_image(
    image_url: str,
    image_io_kwargs: dict[str, Any] | None = None,
) -> Image.Image:
    """
    Args:
        image_url: URL of the image file to fetch.
        image_io_kwargs: Additional kwargs passed to handle image IO.
    """
    media_io_kwargs = None if not image_io_kwargs else {"image": image_io_kwargs}

    '''
    =============================
    Modify by vllm_mlu
    =============================
    @brief: set 'allowed_local_media_path' as default
    '''
    media_connector = MediaConnector(media_io_kwargs, 
        allowed_local_media_path=image_io_kwargs["allowed_local_media_path"]
    )
    '''
    ==================
    End of MLU Hijack
    ==================
    '''
    return media_connector.fetch_image(image_url)


MluHijackObject.apply_hijack(multimodal,
                             multimodal.utils,
                             vllm__multimodal__utils__fetch_image)