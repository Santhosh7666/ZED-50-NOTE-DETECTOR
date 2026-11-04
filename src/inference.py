"""
Inference Engine Module for Real-Time Banknote Detection.
Supports ONNX Runtime (with CPU / CUDA execution provider) and OpenCV DNN fallback.
Provides clean session management, latency profiling, and error handling.
"""

import os
import time
from typing import Tuple, List, Optional
import numpy as np


class ONNXDetector:
    """
    ONNX Runtime Inference Engine wrapper optimized for real-time CPU/GPU execution.
    """

    def __init__(
        self,
        model_path: str = "models/model.onnx",
        prefer_gpu: bool = False,
    ):
        self.model_path = model_path
        self.prefer_gpu = prefer_gpu
        self.session = None
        self.input_name = None
        self.output_names = []
        self.input_shape = (1, 3, 640, 640)
        self.execution_provider = "CPUExecutionProvider"
        self._load_session()

    def _load_session(self):
        """Initializes ONNX Runtime inference session with optimal threading and provider."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found at '{self.model_path}'. "
                f"Please ensure export_onnx.py has been run or path is correct."
            )

        try:
            import onnxruntime as ort

            # Configure session options for minimal latency
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = max(1, os.cpu_count() // 2 if os.cpu_count() else 2)
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            available_providers = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if self.prefer_gpu and "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.execution_provider = "CUDAExecutionProvider"

            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers,
            )

            # Retrieve input & output node metadata
            model_inputs = self.session.get_inputs()
            self.input_name = model_inputs[0].name
            self.input_shape = tuple(
                dim if isinstance(dim, int) else 640 for dim in model_inputs[0].shape
            )
            self.output_names = [out.name for out in self.session.get_outputs()]

            print(
                f"[ONNXDetector] Successfully initialized model: {self.model_path} "
                f"using {providers[0]} | Input: {self.input_name} {self.input_shape}"
            )

        except ImportError:
            raise ImportError(
                "onnxruntime is not installed. Please install it via: pip install onnxruntime"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model via onnxruntime: {e}")

    def infer(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Executes model forward pass.

        Args:
            input_tensor: Normalized float32 numpy array [1, 3, H, W]

        Returns:
            predictions: Raw output array from YOLO model
            inference_time_ms: High-precision elapsed inference time in milliseconds
        """
        t0 = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        inference_time_ms = (time.perf_counter() - t0) * 1000.0

        return outputs[0], inference_time_ms


class OpenCVNetDetector:
    """
    Fallback inference engine utilizing OpenCV's built-in DNN module.
    Useful when running in environments without onnxruntime installed.
    """

    def __init__(self, model_path: str = "models/model.onnx"):
        self.model_path = model_path
        self.net = None
        self._load_net()

    def _load_net(self):
        import cv2

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at '{self.model_path}'.")

        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[OpenCVNetDetector] Loaded {self.model_path} using OpenCV DNN CPU.")

    def infer(self, input_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        import cv2

        t0 = time.perf_counter()
        self.net.setInput(input_tensor)
        preds = self.net.forward()
        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        return preds, inference_time_ms

