import pyrealsense2 as rs
import numpy as np
import tensorrt as trt
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from optimized_inference import (
    preprocess_image,
    allocate_buffers,
    do_inference,
    postprocess_outputs,
)

# ——— Config ———
ONNX_INPUT_SHAPE = (1, 3, 640, 640)
ENGINE_PATH       = "best_fp16.trt"
TRT_LOGGER        = trt.Logger(trt.Logger.INFO)
IMG_W, IMG_H      = 640, 480

# ——— Load engine & buffers ———
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        return rt.deserialize_cuda_engine(f.read())

engine  = load_engine(ENGINE_PATH)
context = engine.create_execution_context()
inputs, outputs, bindings, stream = allocate_buffers(engine)

# ——— RealSense setup ———
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, IMG_W, IMG_H, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, IMG_W, IMG_H, rs.format.z16, 30)
align    = rs.align(rs.stream.color)
pipeline.start(cfg)

profile          = pipeline.get_active_profile()
depth_sensor     = profile.get_device().first_depth_sensor()
depth_scale      = depth_sensor.get_depth_scale()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

try:
    while True:
        frames        = pipeline.wait_for_frames()
        aligned       = align.process(frames)
        depth_frame   = aligned.get_depth_frame()
        color_frame   = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # get numpy images
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # — Preprocess & infer —
        in_tensor = preprocess_image(color_image, ONNX_INPUT_SHAPE)
        np.copyto(inputs[0]['host'], in_tensor.ravel())
        trt_outs  = do_inference(context, bindings, inputs, outputs, stream)
        dets      = postprocess_outputs(trt_outs, conf_threshold=0.3)

        # — Post-process: compute and print XYZ for first valid detection —
        found = False
        for d in dets:
            x1, y1, x2, y2 = map(int, d['bbox'])
            xc, yc        = (x1 + x2) // 2, (y1 + y2) // 2

            # bounds check
            if not (0 <= xc < IMG_W and 0 <= yc < IMG_H):
                continue

            depth = depth_frame.get_distance(xc, yc)
            X, Y, Z = rs.rs2_deproject_pixel_to_point(
                color_intrinsics,
                [xc, yc],
                depth
            )

            print(f"Detection: class={d['class_id']}  "
                  f"conf={d['conf']:.2f}  "
                  f"X={X:.3f}m  Y={Y:.3f}m  Z={Z:.3f}m")
            found = True
            break

        if not found:
            print("No object detected this frame.")

except KeyboardInterrupt:
    pass

finally:
    pipeline.stop()
    print("Shutting down.")
