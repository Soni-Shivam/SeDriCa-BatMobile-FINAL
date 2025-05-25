import pyrealsense2 as rs
import numpy as np
import tensorrt as trt
import cv2
import time
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
ENGINE_PATH      = "best_fp16.trt"
TRT_LOGGER       = trt.Logger(trt.Logger.INFO)
IMG_W, IMG_H     = 640, 480
CONF_THRESHOLD   = 3  # adjust as needed
IOU_THRESHOLD   = 1 
CLASS_NAMES      = ["superman"]  # replace with your class names

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
    fps_counter = 0
    start_time = time.time()

    while True:
        frames      = pipeline.wait_for_frames()
        aligned     = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # — Preprocess & infer —
        in_tensor = preprocess_image(color_image, ONNX_INPUT_SHAPE)
        np.copyto(inputs[0]['host'], in_tensor.ravel())
        trt_outs = do_inference(context, bindings, inputs, outputs, stream)
        dets     = postprocess_outputs(trt_outs, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD)


        # — Draw detections —
        for d in dets:
            class_id = d['class_id']
            """ if class_id != 1:  # Only process "superman" class (class_id == 0)
                continue """
            conf = d['conf']
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, d['bbox'])
            xc, yc         = (x1 + x2) // 2, (y1 + y2) // 2

            # bounds check
            if not (0 <= xc < IMG_W and 0 <= yc < IMG_H):
                continue

            """ if (x2 - x1) < 80 and (y2 - y1) < 80:  # Minimum box size
                continue """

            """ if (x2 - x1) > 60 or (y2 - y1) > 60:  # max  check
                continue """
    
            depth = depth_frame.get_distance(xc, yc)
            if depth <= 0 or depth > 5.0:
                continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [xc, yc], depth)

            class_id = d['class_id']
            label = f"{CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'cls'+str(class_id)} {conf:.2f} Z={Z:.2f}m"

            # Draw
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 2, lineType=cv2.LINE_AA)

            print(f"Detection: class={class_id} conf={conf:.2f} X={X:.3f}m Y={Y:.3f}m Z={Z:.3f}m")

        # — FPS calculation —
        fps_counter += 1
        elapsed = time.time() - start_time
        if elapsed >= 1.0:
            fps = fps_counter / elapsed
            fps_counter = 0
            start_time = time.time()
        else:
            fps = 0.0  # placeholder until at least 1 second passes

        cv2.putText(color_image, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show
        cv2.imshow("RealSense Detection", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Shutting down.")
