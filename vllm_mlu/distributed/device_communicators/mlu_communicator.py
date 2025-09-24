# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
from typing import Optional

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator import \
    DeviceCommunicatorBase


class MLUCommunicator(DeviceCommunicatorBase):

    def __init__(self,
                 cpu_group: ProcessGroup,
                 device: Optional[torch.device] = None,
                 device_group: Optional[ProcessGroup] = None,
                 unique_name: str = ""):
        super().__init__(cpu_group, device, device_group, unique_name)
        # init device according to rank
        self.device = torch.mlu.current_device()
