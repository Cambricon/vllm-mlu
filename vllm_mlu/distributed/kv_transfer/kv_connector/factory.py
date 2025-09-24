# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory


MLUKVConnectors: dict[str, tuple[str, str]] = {
    "MLUSharedStorageConnector": (
        "vllm_mlu.distributed.kv_transfer.kv_connector.v1.shared_storage_connector",
        "SharedStorageConnector"
    ),
    "MLUNixlConnector": (
        "vllm_mlu.distributed.kv_transfer.kv_connector.v1.nixl_connector",
        "MLUNixlConnector"
    ),
}

for name, (module_path, class_name) in MLUKVConnectors.items():
    if name not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(name, module_path, class_name)