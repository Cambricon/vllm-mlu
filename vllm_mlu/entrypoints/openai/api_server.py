# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from fastapi import Request
from fastapi.responses import Response

import vllm_mlu._mlu_utils as mlu_envs
from vllm.entrypoints.openai.api_server import (
    router, engine_client
)
from vllm_mlu.logger import logger

if mlu_envs.VLLM_SCHEDULER_PROFILE:
    logger.info(
        "vLLM V1 Scheduler Profiler is enabled in the API server. Please use "
        "'tools/utils/post_scheduler_view_action.py' to dump profiling data "
        "after all requests finished.")

    @router.post("/v1/start_scheduler_profile")
    async def start_scheduler_profile(raw_request: Request):
        logger.info("VLLM-V1 starting scheduler profiler...")
        await engine_client(raw_request).start_scheduler_profile()
        return Response(status_code=200)

    @router.post("/v1/stop_scheduler_profile")
    async def stop_scheduler_profile(raw_request: Request):
        logger.info("VLLM-V1 scheduler stopping profiler...")
        await engine_client(raw_request).stop_scheduler_profile()
        return Response(status_code=200)