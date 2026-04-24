# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import logging
from typing import cast
from vllm.logger import _VllmLogger


class _ColorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith('vllm_mlu'):
            return True
        if record.levelno == logging.INFO:
            record.msg = f"\033[32m{record.msg}\033[0m"
        elif record.levelno == logging.WARNING:
            record.msg = f"\033[33m{record.msg}\033[0m"
        return True


def _apply_mlu_color(logger):
    if not logger.handlers:
        return
    for h in logger.handlers:
        if any(isinstance(f, _ColorFilter) for f in h.filters):
            return
        h.addFilter(_ColorFilter())


def _mlu_init_logger(name: str) -> logging.Logger:
    """Initialize loggers for vllm_mlu module,
    and keep the configuration consistent with the vllm module"""
    mlu_logger = logging.getLogger(name)
    vllm_logger = logging.Logger.manager.loggerDict.get('vllm', None)
    if vllm_logger:
        mlu_logger.setLevel(vllm_logger.level)
        mlu_logger.propagate = vllm_logger.propagate
        mlu_logger.handlers = vllm_logger.handlers
    return mlu_logger


def init_logger(name: str) -> _VllmLogger:
    vllm_logger = cast(_VllmLogger, _mlu_init_logger(name))
    _apply_mlu_color(vllm_logger)
    return vllm_logger


logger = init_logger(__name__)