<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project -->
### Cambricon vLLM (vllm_mlu)

#### 1. 项目描述

Cambricon vLLM（vllm_mlu）基于社区vLLM提供的[插件系统](https://docs.vllm.ai/en/latest/design/plugin_system.html)开发，旨在为用户提供在寒武纪MLU硬件平台上高效运行大语言模型（LLM）推理和服务的能力。

vllm_mlu支持包括但不限于Chunk Prefill、Prefix Caching、Spec Decode、Graph Mode、Sleep Mode等vLLM原生特性。

#### 2. 更新历史

[2026.04.24] vllm_mlu day0支持DeepSeek-V4

#### 3. 使用说明

软件环境依赖：Cambricon SDK，SDK获取请联系寒武纪官方支持渠道：[ecosystem@cambricon.com](mailto:ecosystem@cambricon.com)

*NOTE：vllm-mlu仓库仅支持MLU370以上的设备*

##### 3.1 镜像使用

使⽤寒武纪SDK提供的镜像 Cambricon vLLM Container。

```
# 加载镜像

docker load -i cambricon_vllm_container.tar.gz


# 进入镜像

docker run -it --net=host \
    --shm-size '64gb' --privileged -it \
    --ulimit memlock=-1 ${IMAGE_NAME} \
    /bin/bash

# 使⽤推理环境
source /torch/venv3/pytorch_infer/bin/activate
```

##### 3.2 ⾃定义安装步骤

安装Cambricon vLLM前需要保证依赖已正确安装。

安装步骤：

```bash
# 已经获取Cambricon vLLM源码,包含vllm源码

# 基于vllm源码安装
cd vllm-v{社区vLLM版本}/
VLLM_TARGET_DEVICE=empty pip install -e . # 使⽤开发者模式安装

# 基于vllm-mlu源码安装
git clone https://github.com/Cambricon/vllm-mlu
cd vllm-mlu
pip install -e . # 使⽤开发者模式安装

# 安装ray
# 1. 进⼊vllm-mlu源码中。
cd tools/ray_mlu/
# 2. 适配基于Ray安装
pip install --no-cache-dir --force-reinstall ray==2.51.1
# 3. 为了在寒武纪设备运⾏，Ray也需要适配寒武纪软件。
# PIP_INSTALL_LOC 指向pip的安装路径
cp __init__.py ${RAY_DIR}/_private/accelerators/__init__.py
cp mlu.py ${RAY_DIR}/_private/accelerators/
cp nsight.py ${RAY_DIR}/_private/runtime_env/nsight.py
cp node.py ${RAY_DIR}/_private/node.py
cp worker.py ${RAY_DIR}/_private/worker.py
cp device_manager/__init__.py ${RAY_DIR}/air/_internal/device_manager/__init__.py
cp device_manager/mlu.py ${RAY_DIR}/air/_internal/device_manager/
```

##### 3.3 运行步骤

Cambricon vLLM代码运⾏和vLLM社区⼀致。

###### 3.3.1 离线推理命令

```
# 运行推理命令

python examples/offline_inference/offline_inference.py ${MODEL_PATH}
```

###### 3.3.2 在线推理命令

分别启动server和client，完成推理服务，示例如下：

```
# server

vllm serve ${MODEL_PATH} \
    --port 8100 \
    --block-size 1 \
    --max-model-len 4096 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.96 \
    --trust-remote-code \
    --enable-expert-parallel \
    --no-enable-prefix-caching \
    --disable-log-requests \
    --enforce-eager


# client, we post a single request here.

curl -X POST http://localhost:8100/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": ${MODEL_PATH}, \
        "prompt": "The future of AI is", \
        "max_tokens": 128, "temperature": 0.7 \
    }'
```
