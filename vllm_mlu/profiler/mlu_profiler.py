# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

from vllm.logger import init_logger

logger = init_logger(__name__)


class MluProfilerWrapper:
    def __init__(self) -> None:
        self._profiler_running = False
        # Note: lazy import to avoid dependency issues if MLU is not available.
        import torch.mlu.profiler as mlu_profiler

        self._mlu_profiler = mlu_profiler

    def start(self) -> None:
        try:
            self._mlu_profiler.start()
            self._profiler_running = True
            logger.info_once("Started MLU profiler")
        except Exception as e:
            logger.warning_once("Failed to start MLu profiler: %s", e)

    def stop(self) -> None:
        if self._profiler_running:
            try:
                self._mlu_profiler.stop()
                logger.info_once("Stopped MLU profiler")
            except Exception as e:
                logger.warning_once("Failed to stop MLU profiler: %s", e)
            finally:
                self._profiler_running = False

    def shutdown(self) -> None:
        """Ensure profiler is stopped when shutting down."""
        self.stop()
