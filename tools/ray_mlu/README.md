# Ray MLU Adapter

This directory contains the Ray adapter for Cambricon MLU devices.

## Supported Ray Version

**Ray 2.51.1**

## Adaptation Method

After installing vLLM MLU, you need to copy the adapter files to your Ray installation path.

### Steps

1. Find your Ray installation path:
```bash
pip show ray | grep Location
```

2. Copy the adapter files (replace `${PIP_INSTALL_LOC}` with the path from step 1):

```bash
cd tools/ray_mlu/

cp __init__.py  ${PIP_INSTALL_LOC}/ray/_private/accelerators/
cp mlu.py       ${PIP_INSTALL_LOC}/ray/_private/accelerators/
cp nsight.py    ${PIP_INSTALL_LOC}/ray/_private/runtime_env/
cp node.py      ${PIP_INSTALL_LOC}/ray/_private/node.py
cp worker.py    ${PIP_INSTALL_LOC}/ray/_private/worker.py
cp device_manager/__init__.py ${PIP_INSTALL_LOC}/ray/air/_internal/device_manager/
cp device_manager/mlu.py ${PIP_INSTALL_LOC}/ray/air/_internal/device_manager/
```

## File Manifest

| File | Destination | Description |
|------|-------------|-------------|
| `__init__.py` | `ray/_private/accelerators/` | MLU accelerator package init |
| `mlu.py` | `ray/_private/accelerators/` | MLUAcceleratorManager - registers MLU as Ray accelerator |
| `nsight.py` | `ray/_private/runtime_env/` | NsightPlugin - profiling via cnperf-cli |
| `node.py` | `ray/_private/node.py` | Patched Ray node initialization for MLU |
| `worker.py` | `ray/_private/worker.py` | Patched Ray worker for MLU support |
| `device_manager/__init__.py` | `ray/air/_internal/device_manager/` | Device manager package init |
| `device_manager/mlu.py` | `ray/air/_internal/device_manager/` | MLUTorchDeviceManager for torch device management |

## Usage with vLLM

After adaptation, you can use Ray with MLU devices:

```python
import ray
import torch

# Initialize Ray - it will automatically detect MLU devices
ray.init()

# Verify MLU is available
print(ray.available_resources().get("MLU", 0))
```