# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.v1.engine.async_llm import AsyncLLM

from vllm_mlu.mlu_hijack_utils import MluHijackObject


class AsyncLLM_MluHijack(AsyncLLM):

    async def start_scheduler_profile(self) -> None:
        await self.engine_core.start_scheduler_profile()

    async def stop_scheduler_profile(self) -> None:
        await self.engine_core.stop_scheduler_profile()


MluHijackObject.apply_hijack(AsyncLLM,
                             "start_scheduler_profile",
                             AsyncLLM_MluHijack.start_scheduler_profile)
MluHijackObject.apply_hijack(AsyncLLM,
                             "stop_scheduler_profile",
                             AsyncLLM_MluHijack.stop_scheduler_profile)