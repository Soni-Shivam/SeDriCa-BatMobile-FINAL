# File: convert_to_trt.py
"""
Converts a YOLOv8 `.pt` model to ONNX and then TensorRT engine (.trt).
Usage:
    python convert_to_trt.py --pt_path best.pt --onnx_path best.onnx --trt_path best.trt
"""
import os
import argparse
from ultralytics import YOLO
import tensorrt as trt
import pycuda.autoinit

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def export_to_onnx(pt_path: str, onnx_path: str):
    print(f"Exporting {pt_path} to ONNX at {onnx_path}...")
    model = YOLO(pt_path)
    model.export(format='onnx', imgsz=(640, 480), onnx_path=onnx_path)
    print("ONNX export complete.")


def build_trt_engine(onnx_path: str, trt_path: str, fp16: bool = True):
    print(f"Building TensorRT engine from {onnx_path}...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model.")

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1 GiB
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TRT engine.")

    with open(trt_path, 'wb') as f:
        f.write(engine.serialize())
    print(f"TensorRT engine saved to {trt_path}.")


def main():
    parser = argparse.ArgumentParser(description="Convert YOLO PT to TRT engine.")
    parser.add_argument('--pt_path', type=str, required=True, help='Path to input YOLO .pt model')
    parser.add_argument('--onnx_path', type=str, default='best.onnx', help='Output ONNX file path')
    parser.add_argument('--trt_path', type=str, default='best.trt', help='Output TRT engine file path')
    parser.add_argument('--no_fp16', action='store_true', help='Disable FP16 precision')
    args = parser.parse_args()

    export_to_onnx(args.pt_path, args.onnx_path)
    build_trt_engine(args.onnx_path, args.trt_path, fp16=not args.no_fp16)


if __name__ == '__main__':
    main()