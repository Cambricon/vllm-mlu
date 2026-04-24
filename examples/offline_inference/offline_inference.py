# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project

import sys
from vllm import LLM, SamplingParams


def main(model_path):
    # Sample prompts.
    prompts = [
        "The benefits of exercise include",
        "The importance of reading books is",
        "Gardening can be relaxing because",
        "A good night's sleep is essential for",
    ]
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, max_tokens=10)

    # Create an LLM.
    engine_args_dict = {
        "model": model_path,
        "tensor_parallel_size": 8,
        "enable_expert_parallel": True,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "trust_remote_code": True,
        "max_num_seqs": len(prompts),
        "max_model_len": 4096,
        "block_size": 1,
        "gpu_memory_utilization": 0.96,
    }
    llm = LLM(**engine_args_dict)

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python offline_inference.py <model_path>")
        sys.exit(1)
    main(sys.argv[1])
