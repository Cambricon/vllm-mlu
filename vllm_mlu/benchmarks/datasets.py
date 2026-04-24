# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-MLU project
"""
This module defines a framework for sampling benchmark requests from various
datasets. Each dataset subclass of BenchmarkDataset must implement sample
generation. Supported dataset types include:
  - ShareGPT
  - Random (synthetic)
  - Sonnet
  - BurstGPT
  - HuggingFace
  - VisionArena
"""

from tempfile import NamedTemporaryFile

import numpy as np

from vllm.benchmarks.datasets import RandomMultiModalDataset
from vllm_mlu.mlu_hijack_utils import MluHijackObject


def vllm__benchmarks__datasets__RandomMultiModalDataset__generate_synthetic_video(
        self, width: int, height: int, num_frames: int
    ) -> dict:
        """Generate synthetic video with random values.

        Creates a video with random pixel values, encodes it to MP4 format,
        and returns the content as bytes.
        """
        import cv2

        random_pixels = self._rng.integers(
            0,
            256,
            (num_frames, height, width, 3),
            dtype=np.uint8,
        )

        # Create a temporary video file in memory
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30  # frames per second

        with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_path = temp_file.name

            # Create video writer
            video_writer = cv2.VideoWriter(
                temp_path, fourcc=fourcc, fps=fps, frameSize=(width, height)
            )

            if not video_writer.isOpened():
                raise RuntimeError("Failed to create video writer")

            for frame in random_pixels:
                video_writer.write(frame)

            video_writer.release()
            temp_file.close()

            # Read the video file content
            with open(temp_path, "rb") as f:
                video_content = f.read()

            return {"bytes": video_content}
        

MluHijackObject.apply_hijack(
    RandomMultiModalDataset,
    RandomMultiModalDataset.generate_synthetic_video,
    vllm__benchmarks__datasets__RandomMultiModalDataset__generate_synthetic_video,
)