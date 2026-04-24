# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.ray.ray_env import get_env_vars_to_copy
from vllm.v1.executor.ray_executor import RayDistributedExecutor,  RayWorkerMetaData
from vllm.v1.executor.ray_utils import (
    RayWorkerWrapper,
    initialize_ray_cluster,
    ray,
)
from vllm.utils.network_utils import (
    get_distributed_init_method,
    get_ip,
    get_open_port,
)

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

from vllm_mlu.mlu_hijack_utils import MluHijackObject

logger = init_logger(__name__)


class RayDistributedExecutor_MluHijack(RayDistributedExecutor):

    def _init_executor(self) -> None:
        self.forward_dag: ray.dag.CompiledDAG | None = None

        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: For MLU, avoid compiling NVIDIA's NCCL
        '''
        # For TPU or XPU, avoid compiling NVIDIA's NCCL
        if current_platform.is_tpu() or current_platform.is_xpu() or \
            current_platform.is_out_of_tree():
            os.environ["VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE"] = "shm"
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        assert self.uses_ray
        initialize_ray_cluster(self.parallel_config)
        placement_group = self.parallel_config.placement_group

        # Disable Ray usage stats collection.
        ray_usage = os.environ.get("RAY_USAGE_STATS_ENABLED", "0")
        if ray_usage != "1":
            os.environ["RAY_USAGE_STATS_ENABLED"] = "0"

        # Create the parallel GPU workers.
        self._init_workers_ray(placement_group)

        # KV connector setup
        self.has_connector = self.vllm_config.kv_transfer_config is not None

        self.uses_sampler = self.vllm_config.model_config.runner_type != "pooling" and (
            self.vllm_config.ec_transfer_config is None
            or not self.vllm_config.ec_transfer_config.is_ec_producer
        )

        self.scheduler_output: SchedulerOutput | None = None

    def _configure_ray_workers_use_nsight(self, ray_remote_kwargs) -> dict[str, Any]:
        # If nsight profiling is enabled, we need to set the profiling
        # configuration for the ray workers as runtime env.
        runtime_env = ray_remote_kwargs.setdefault("runtime_env", {})
        '''
        =============================
        Modify by vllm_mlu
        =============================
        @brief: use default cnperf config.
        '''
        runtime_env.update({
            # use default cnperf config
            "nsight": "default"
        })
        '''
        ==================
        End of MLU Hijack
        ==================
        '''

        return ray_remote_kwargs

    def _init_workers_ray(self, placement_group: "PlacementGroup", **ray_remote_kwargs):
        num_gpus = envs.VLLM_RAY_PER_WORKER_GPUS

        # The driver dummy worker does not actually use any resources.
        # It holds the resource for the driver worker.
        self.driver_dummy_worker: RayWorkerWrapper | None = None
        # The remaining workers are the actual ray actors.
        self.workers: list[RayWorkerWrapper] = []

        # Used in ray compiled DAG: indexed first by PP rank,
        # and then TP rank. In other words, the inner list is
        # the TP group of workers for a PP rank.
        self.pp_tp_workers: list[list[RayWorkerWrapper]] = []

        if self.parallel_config.ray_workers_use_nsight:
            ray_remote_kwargs = self._configure_ray_workers_use_nsight(
                ray_remote_kwargs
            )

        # Create the workers.
        bundle_indices: list[int]
        if envs.VLLM_RAY_BUNDLE_INDICES:
            # Use the bundle indices specified by the user.
            bundle_indices = list(map(int, envs.VLLM_RAY_BUNDLE_INDICES.split(",")))
            assert len(bundle_indices) == self.parallel_config.world_size, (
                "VLLM_RAY_BUNDLE_INDICES must have the same size"
                f" as the world size, but got {bundle_indices=} "
                f"and {self.parallel_config.world_size=}"
            )
            assert len(set(bundle_indices)) == len(bundle_indices), (
                "VLLM_RAY_BUNDLE_INDICES cannot have duplicate values,"
                f" but got {bundle_indices=}"
            )
        else:
            # use the first N bundles that have GPU resources.
            bundle_indices = []
            for bundle_id, bundle in enumerate(placement_group.bundle_specs):
                if bundle.get(current_platform.ray_device_key, 0):
                    bundle_indices.append(bundle_id)
            bundle_indices = bundle_indices[: self.parallel_config.world_size]

        worker_metadata: list[RayWorkerMetaData] = []
        driver_ip = get_ip()
        for rank, bundle_id in enumerate(bundle_indices):
            '''
            =============================
            Modify by vllm_mlu
            =============================
            @brief: support ray + cnperf-cli
            '''
            if self.parallel_config.ray_workers_use_nsight:
                ray_remote_kwargs['runtime_env'].update({
                    "nsight": {
                        "o": f"cnperf_rank_{rank}",
                        "force_overwrite": "true"
                    }
                })
                if rank == 0:
                    ray_remote_kwargs['runtime_env'].update({
                        "nsight": {}
                    })
            '''
            ==================
            End of MLU Hijack
            ==================
            '''
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_id,
            )

            if current_platform.ray_device_key == "GPU":
                # NV+AMD GPUs, and Intel XPUs
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=num_gpus,
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(  # type: ignore[attr-defined]
                    vllm_config=self.vllm_config, rpc_rank=rank
                )
            else:
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=0,
                    resources={current_platform.ray_device_key: num_gpus},
                    scheduling_strategy=scheduling_strategy,
                    **ray_remote_kwargs,
                )(RayWorkerWrapper).remote(  # type: ignore[attr-defined]
                    vllm_config=self.vllm_config, rpc_rank=rank
                )
            worker_metadata.append(RayWorkerMetaData(worker=worker, created_rank=rank))

        worker_ips = ray.get(
            [
                each.worker.get_node_ip.remote()  # type: ignore[attr-defined]
                for each in worker_metadata
            ]
        )

        for each, ip in zip(worker_metadata, worker_ips):
            each.ip = ip

        logger.debug("workers: %s", worker_metadata)
        logger.debug("driver_dummy_worker: %s", self.driver_dummy_worker)

        ip_counts: dict[str, int] = {}
        for ip in worker_ips:
            ip_counts[ip] = ip_counts.get(ip, 0) + 1

        def sort_by_driver_then_worker_ip(item: RayWorkerMetaData):
            """
            Sort the workers based on 3 properties:
            1. If the worker is on the same node as the driver (vllm engine),
                it should be placed first.
            2. Then, if the worker is on a node with fewer workers, it should
                be placed first.
            3. Finally, if the work is on a node with smaller IP address, it
                should be placed first.
            """
            ip = item.ip
            return 0 if ip == driver_ip else 1, ip_counts[ip], ip

        # After sorting, the workers on the same node will be
        # close to each other, and the workers on the driver
        # node will be placed first.
        sorted_worker_metadata = sorted(
            worker_metadata, key=sort_by_driver_then_worker_ip
        )
        for i, item in enumerate(sorted_worker_metadata):
            item.adjusted_rank = i
        self.workers = [item.worker for item in sorted_worker_metadata]
        rerank_mapping = {
            item.created_rank: item.adjusted_rank for item in sorted_worker_metadata
        }
        self.collective_rpc("adjust_rank", args=(rerank_mapping,))

        # Get the set of GPU IDs used on each node.
        worker_node_and_gpu_ids = []
        for worker in [self.driver_dummy_worker] + self.workers:
            if worker is None:
                # driver_dummy_worker can be None when using ray spmd worker.
                continue
            worker_node_and_gpu_ids.append(
                ray.get(worker.get_node_and_gpu_ids.remote())
            )  # type: ignore[attr-defined]

        node_workers = defaultdict(list)  # node id -> list of worker ranks
        node_gpus = defaultdict(list)  # node id -> list of gpu ids

        for i, (node_id, gpu_ids) in enumerate(worker_node_and_gpu_ids):
            node_workers[node_id].append(i)
            # `gpu_ids` can be a list of strings or integers.
            # convert them to integers for consistency.
            # NOTE: gpu_ids can be larger than 9 (e.g. 16 GPUs),
            # string sorting is not sufficient.
            # see https://github.com/vllm-project/vllm/issues/5590
            gpu_ids = [int(x) for x in gpu_ids]
            node_gpus[node_id].extend(gpu_ids)
        for node_id, gpu_ids in node_gpus.items():
            node_gpus[node_id] = sorted(gpu_ids)

        all_ips = set(worker_ips + [driver_ip])
        n_ips = len(all_ips)
        n_nodes = len(node_workers)

        if n_nodes != n_ips:
            raise RuntimeError(
                f"Every node should have a unique IP address. Got {n_nodes}"
                f" nodes with node ids {list(node_workers.keys())} and "
                f"{n_ips} unique IP addresses {all_ips}. Please check your"
                " network configuration. If you set `VLLM_HOST_IP`"
                " environment variable, make sure it is unique for"
                " each node."
            )

        # Set environment variables for the driver and workers.
        all_args_to_update_environment_variables = [
            {
                current_platform.device_control_env_var: ",".join(
                    map(str, node_gpus[node_id])
                ),
            }
            for (node_id, _) in worker_node_and_gpu_ids
        ]

        # Environment variables to copy from driver to workers
        env_vars_to_copy = get_env_vars_to_copy(
            exclude_vars=self.WORKER_SPECIFIC_ENV_VARS,
            additional_vars=set(current_platform.additional_env_vars).union(
                self.ADDITIONAL_ENV_VARS
            ),
            destination="workers",
        )

        # Copy existing env vars to each worker's args
        for args in all_args_to_update_environment_variables:
            # TODO: refactor platform-specific env vars
            for name in env_vars_to_copy:
                if name in os.environ:
                    args[name] = os.environ[name]

        self._env_vars_for_all_workers = all_args_to_update_environment_variables

        self.collective_rpc(
            "update_environment_variables", args=(self._get_env_vars_to_be_updated(),)
        )

        if len(node_gpus) == 1:
            # in single node case, we don't need to get the IP address.
            # the loopback address is sufficient
            # NOTE: a node may have several IP addresses, one for each
            # network interface. `get_ip()` might return any of them,
            # while they might not work for communication inside the node
            # if the network setup is complicated. Using the loopback address
            # solves this issue, as it always works for communication inside
            # the node.
            driver_ip = "127.0.0.1"
        distributed_init_method = get_distributed_init_method(
            driver_ip, get_open_port()
        )

        # Initialize the actual workers inside worker wrapper.
        all_kwargs = []
        for rank, (node_id, _) in enumerate(worker_node_and_gpu_ids):
            local_rank = node_workers[node_id].index(rank)
            kwargs = dict(
                vllm_config=self.vllm_config,
                local_rank=local_rank,
                rank=rank,
                distributed_init_method=distributed_init_method,
                is_driver_worker=(not self.parallel_config)
                or (rank % self.parallel_config.tensor_parallel_size == 0),
            )
            all_kwargs.append(kwargs)
        self.collective_rpc("init_worker", args=(all_kwargs,))

        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

        for pp_rank in range(self.parallel_config.pipeline_parallel_size):
            self.pp_tp_workers.append([])
            for tp_rank in range(self.parallel_config.tensor_parallel_size):
                # PP=2, TP=4
                # pp_tp_workers = [[0, 1, 2, 3], [4, 5, 6, 7]]
                rank = (pp_rank * self.parallel_config.tensor_parallel_size) + tp_rank
                assert len(self.pp_tp_workers[pp_rank]) == tp_rank
                assert pp_rank < len(self.pp_tp_workers)
                self.pp_tp_workers[pp_rank].append(self.workers[rank])

MluHijackObject.apply_hijack(
    RayDistributedExecutor,
    RayDistributedExecutor._configure_ray_workers_use_nsight,
    RayDistributedExecutor_MluHijack._configure_ray_workers_use_nsight
)
MluHijackObject.apply_hijack(
    RayDistributedExecutor,
    RayDistributedExecutor._init_workers_ray,
    RayDistributedExecutor_MluHijack._init_workers_ray
)
MluHijackObject.apply_hijack(
    RayDistributedExecutor,
    RayDistributedExecutor._init_executor,
    RayDistributedExecutor_MluHijack._init_executor
)