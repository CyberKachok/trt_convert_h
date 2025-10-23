"""Utility script to export the HIT student network to TensorRT.

The script keeps the original PyTorch model untouched: it instantiates the
network, loads the checkpoint, exports an intermediate ONNX representation and
finally builds a TensorRT engine.  Only the bare minimum of options required in
our workflow is exposed in order to keep the script compact and easy to follow.
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Iterable, Tuple

import torch
import tensorrt as trt
from omegaconf import OmegaConf

import hit_student


def _load_cfg(path: pathlib.Path):
    cfg = OmegaConf.load(path)
    if "net" in cfg and "cfg" in cfg.net:
        return cfg.net.cfg
    return cfg


def _select_int(cfg, keys: Iterable[str]) -> int:
    for key in keys:
        value = OmegaConf.select(cfg, key)
        if value is not None:
            return int(value)
    raise AttributeError("Missing configuration value")


def _build_model(cfg, checkpoint: pathlib.Path, device: torch.device) -> torch.nn.Module:
    model = hit_student.VT(cfg)  # type: ignore[attr-defined]
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        current = model.state_dict()
        filtered = {
            k.replace("net.", ""): v
            for k, v in state_dict.items()
            if k.replace("net.", "") in current and current[k.replace("net.", "")].shape == v.shape
        }
        model.load_state_dict(filtered, strict=True)
    model.eval().to(device)
    return model


def _export_onnx(
    model: torch.nn.Module,
    dummy_uav: torch.Tensor,
    dummy_sat_embed: torch.Tensor,
    onnx_path: pathlib.Path,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_uav, dummy_sat_embed),
        onnx_path,
        input_names=["search_image", "template_embedding"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "search_image": {0: "batch"},
            "template_embedding": {0: "batch"},
            "logits": {0: "batch"},
        },
    )


def _build_engine(
    onnx_path: pathlib.Path,
    engine_path: pathlib.Path,
    precision: str,
    workspace_size: int,
    input_shapes: dict[str, Tuple[int, ...]],
) -> None:
    logger = trt.Logger(trt.Logger.WARNING)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(logger) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, logger) as parser:
        with onnx_path.open("rb") as f:
            if not parser.parse(f.read()):
                last_error = parser.get_error(parser.num_errors - 1) if parser.num_errors else "unknown"
                raise RuntimeError(f"Failed to parse ONNX graph: {last_error}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

        requested = precision.lower()
        if requested == "fp16":
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("[WARN] FP16 not supported on this platform, falling back to FP32")
        elif requested == "bf16":
            if hasattr(trt.BuilderFlag, "BF16") and getattr(builder, "platform_has_fast_bf16", False):
                config.set_flag(trt.BuilderFlag.BF16)
            else:
                print("[WARN] BF16 not supported on this platform, falling back to FP32")

        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            provided = input_shapes.get(tensor.name)
            if provided is None:
                raise KeyError(f"Missing shape hint for input '{tensor.name}'")
            sanitized = tuple(
                dim if dim >= 0 else provided[idx]
                for idx, dim in enumerate(tensor.shape)
            )
            profile.set_shape(tensor.name, sanitized, sanitized, sanitized)
        config.add_optimization_profile(profile)

        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT engine build failed")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(engine.serialize())


def _dummy_inputs(
    model: torch.nn.Module,
    device: torch.device,
    uav_size: int,
    sat_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dummy_uav = torch.randn(1, 3, uav_size, uav_size, device=device)
    dummy_sat = torch.randn(1, 3, sat_size, sat_size, device=device)
    with torch.no_grad():
        dummy_sat_embed = model.forward_sat(dummy_sat)
    return dummy_uav, dummy_sat_embed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the HIT student network to TensorRT")
    parser.add_argument("cfg", type=pathlib.Path, help="Path to the hydra config used for the model")
    parser.add_argument("checkpoint", type=pathlib.Path, help="Path to the trained PyTorch checkpoint")
    parser.add_argument("engine", type=pathlib.Path, help="Output path for the TensorRT engine")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16", help="Desired precision mode")
    parser.add_argument("--onnx", type=pathlib.Path, default=None, help="Optional path to store the intermediate ONNX model")
    parser.add_argument("--workspace", type=int, default=1 << 30, help="Workspace size in bytes for TensorRT builder")
    parser.add_argument("--device", default="cuda", help="Device used for export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    cfg = _load_cfg(args.cfg)
    try:
        uav_size = _select_int(cfg, ("DATA.SEARCH.SIZE", "DATA.SEARCH.RES"))
        sat_size = _select_int(cfg, ("DATA.TEMPLATE.SIZE", "DATA.TEMPLATE.RES"))
    except AttributeError as exc:
        raise AttributeError(
            "Config must provide DATA.SEARCH.SIZE/RES and DATA.TEMPLATE.SIZE/RES"
        ) from exc

    model = _build_model(cfg, args.checkpoint, device)
    dummy_uav, dummy_sat_embed = _dummy_inputs(model, device, uav_size, sat_size)

    onnx_path = args.onnx or args.engine.with_suffix(".onnx")
    _export_onnx(model, dummy_uav, dummy_sat_embed, onnx_path)
    shape_hints = {
        "search_image": tuple(dummy_uav.shape),
        "template_embedding": tuple(dummy_sat_embed.shape),
    }
    _build_engine(onnx_path, args.engine, args.precision, args.workspace, shape_hints)

    print(f"TensorRT engine saved to {args.engine}")


if __name__ == "__main__":
    main()
