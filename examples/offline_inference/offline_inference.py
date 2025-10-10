# SPDX-License-Identifier: Apache-2.0

import sys

from vllm import LLM, SamplingParams


def main(model_path):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The Mid-Autumn Festival is a",
        "The vLLM project is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(
        temperature=0.6, top_p=0.95, top_k=20, max_tokens=14)

    # Create an LLM.
    engine_args_dict = {
        "model": model_path,
        "tensor_parallel_size": 32,
        "distributed_executor_backend": "ray",
        "enable_expert_parallel": True,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "trust_remote_code": True,
        "num_gpu_blocks_override": 2048,
        "max_model_len": 16384,
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
    main(sys.argv[1])
