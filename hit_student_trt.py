"""TensorRT inference helpers for the HIT student model."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List

import numpy as np
import pycuda.autoinit  # noqa: F401  # needed to create a CUDA context for TensorRT
import pycuda.driver as cuda
import tensorrt as trt
import torch

from hit_student import _prep_image, _imread_rgb


@dataclass(frozen=True)
class _Binding:
    index: int
    name: str
    dtype: np.dtype


class TensorRTRunner:
    """Minimal TensorRT runtime wrapper."""

    def __init__(self, engine_path: pathlib.Path) -> None:
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        serialized_engine = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_bindings: List[_Binding] = []
        self.output_bindings: List[_Binding] = []

        for idx in range(engine.num_bindings):
            name = engine.get_binding_name(idx)
            dtype = np.dtype(trt.nptype(engine.get_binding_dtype(idx)))
            binding = _Binding(idx, name, dtype)
            if engine.binding_is_input(idx):
                self.input_bindings.append(binding)
            else:
                self.output_bindings.append(binding)

    def __call__(self, *inputs: np.ndarray) -> List[np.ndarray]:
        if len(inputs) != len(self.input_bindings):
            raise ValueError("Number of inputs does not match engine bindings")

        bindings = [0] * self.engine.num_bindings
        device_buffers: List[cuda.DeviceAllocation] = []

        for binding, host_array in zip(self.input_bindings, inputs):
            host_array = np.ascontiguousarray(host_array.astype(binding.dtype, copy=False))
            self.context.set_binding_shape(binding.index, host_array.shape)
            device_mem = cuda.mem_alloc(host_array.nbytes)
            cuda.memcpy_htod_async(device_mem, host_array, self.stream)
            bindings[binding.index] = int(device_mem)
            device_buffers.append(device_mem)

        host_outputs: List[np.ndarray] = []
        for binding in self.output_bindings:
            shape = tuple(self.context.get_binding_shape(binding.index))
            host_output = np.empty(shape, dtype=binding.dtype)
            device_mem = cuda.mem_alloc(host_output.nbytes)
            bindings[binding.index] = int(device_mem)
            device_buffers.append(device_mem)
            host_outputs.append(host_output)

        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        output_offset = 0
        for binding in self.output_bindings:
            host_array = host_outputs[output_offset]
            device_mem = device_buffers[len(self.input_bindings) + output_offset]
            cuda.memcpy_dtoh_async(host_array, device_mem, self.stream)
            output_offset += 1

        self.stream.synchronize()

        for device_mem in device_buffers:
            device_mem.free()

        return host_outputs


def get_encoder(model: torch.nn.Module, device: str = "cuda"):
    patch_embed = model.backbone.body.patch_embed_sat.to(device).eval()

    def encode(path: str | pathlib.Path) -> np.ndarray:
        with torch.no_grad():
            rgb = _imread_rgb(path)
            img_chw = _prep_image(
                rgb,
                out_size=640,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                center_crop=None,
                do_normalize=True,
            )
            tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device=device, dtype=torch.float32)
            features = patch_embed(tensor)
            return features.detach().cpu().numpy()

    return encode


def get_model_forward(engine_path: str | pathlib.Path, tau: float = 1.0):
    runner = TensorRTRunner(pathlib.Path(engine_path))

    input_by_name = {binding.name: binding for binding in runner.input_bindings}
    template_binding = input_by_name.get("template")
    search_binding = input_by_name.get("search_embedding")
    if template_binding is None or search_binding is None:
        raise RuntimeError("Engine inputs must be named 'template' and 'search_embedding'")

    def hit_forward(uav_meta, sat_meta):
        search = torch.from_numpy(uav_meta.image).to(torch.float32)
        if search.dim() == 3:
            search = search.unsqueeze(0)

        template = torch.from_numpy(sat_meta.data).to(torch.float32)
        if template.dim() == 3:
            template = template.unsqueeze(0)

        template_np = np.ascontiguousarray(template.cpu().numpy().astype(template_binding.dtype, copy=False))
        search_np = np.ascontiguousarray(search.cpu().numpy().astype(search_binding.dtype, copy=False))

        ordered_inputs = [template_np if binding.name == template_binding.name else search_np for binding in runner.input_bindings]
        logits = runner(*ordered_inputs)[0]
        logits_tensor = torch.from_numpy(logits).to(torch.float32)
        probs = torch.nn.functional.softmax(tau * logits_tensor, dim=1)
        return probs.view(20, 20).detach().cpu().numpy()

    return hit_forward
