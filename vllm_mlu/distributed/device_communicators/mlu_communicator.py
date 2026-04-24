# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)


class MLUCommunicator(DeviceCommunicatorBase):

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = ""
    ):
        super().__init__(cpu_group, device, device_group, unique_name)
        # init device according to rank
        self.device = torch.mlu.current_device()
        self.ca_comm: CustomAllreduce | None = None