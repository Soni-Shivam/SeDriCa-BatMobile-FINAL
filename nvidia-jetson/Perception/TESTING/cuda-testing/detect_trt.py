# ----------------------------------------------------------
# File: detect_trt.py
"""
Loads a TensorRT engine and performs real-time object detection
on RealSense frames, computing 3D pose of detected objects.
Usage:
    python detect_trt.py --trt_path best.trt
"""
import os
import argparse
import numpy as np
import cv2
import torch
from torchvision.ops import nms
import pyrealsense2 as rs
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

# Constants
IMG_W, IMG_H = 640, 480
CONF_THRESH = 0.3#for sigmoid
IOU_THRESH = 0.45

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_engine(trt_path: str):
    with open(trt_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for idx, binding in enumerate(engine):
        # Determine size for input vs output
        if engine.binding_is_input(binding):
            # For input, allocate exactly 3*C*H*W elements (RGB image)
            size = 3 * IMG_H * IMG_W
        else:
            # For output, use engine binding shape
            binding_shape = engine.get_binding_shape(binding)
            size = trt.volume(binding_shape)

        import numpy as np # monkey patch numpy
        if not hasattr(np, 'bool'):
            np.bool = bool
            
        dtype = engine.get_binding_dtype(binding)
       

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    return inputs, outputs, bindings, stream


def trt_infer(context, inputs, outputs, bindings, stream, img: np.ndarray):
    # Preprocess
    img_res = cv2.resize(img, (IMG_W, IMG_H))
    img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(img_rgb, (2, 0, 1)).ravel()

    # Host -> Device
    np.copyto(inputs[0]['host'], tensor)
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    # Inference
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Device -> Host
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()

    # Postprocess: reshape, threshold, xywh->xyxy, NMS
    dets = torch.from_numpy(outputs[0]['host'])
    dets = dets.view(-1, 6)  # [N, 6]
    if dets.numel() == 0:
        return [], [], []

    # Split
    xywh = dets[:, :4]
    scores = dets[:, 4]
    classes = dets[:, 5].long()

    # Filter by confidence
    mask = scores > CONF_THRESH
    xywh = xywh[mask]
    scores = scores[mask]
    classes = classes[mask]
    if xywh.numel() == 0:
        return [], [], []

    # Convert to xyxy
    x = xywh[:, 0]
    y = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    final_boxes, final_scores, final_classes = [], [], []
    # Per-class NMS
    for cls in classes.unique():
        cls_mask = classes == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        keep = nms(cls_boxes, cls_scores, IOU_THRESH)
        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_classes.append(torch.full((len(keep),), cls, dtype=torch.long))

    if final_boxes:
        final_boxes = torch.cat(final_boxes, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_classes = torch.cat(final_classes, dim=0)

        # Convert to Python lists
        return final_boxes.cpu().numpy().tolist(), final_scores.cpu().numpy().tolist(), final_classes.cpu().numpy().tolist()
    else:
        return [], [], []


def main():
    parser = argparse.ArgumentParser(description="Detect with TRT engine and RealSense")
    parser.add_argument('--trt_path', type=str, required=True, help='Path to TRT engine')
    args = parser.parse_args()

    # RealSense setup
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(cfg)
    profile = pipeline.get_active_profile()
    ds = profile.get_device().first_depth_sensor()
    depth_scale = ds.get_depth_scale()
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    # Load TRT engine
    engine = load_engine(args.trt_path)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    print("Starting real-time detection...")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not color_frame or not depth_frame:
                continue

            img = np.asanyarray(color_frame.get_data())
            boxes, scores, class_ids = trt_infer(context, inputs, outputs, bindings, stream, img)

            for box, sc, cid in zip(boxes, scores, class_ids):
                x1, y1, x2, y2 = box
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                depth = depth_frame.get_distance(cx, cy)
                X, Y, Z = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                cv2.putText(img, f"Superman {sc:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                print(f"Pose: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}")

            cv2.imshow("Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()