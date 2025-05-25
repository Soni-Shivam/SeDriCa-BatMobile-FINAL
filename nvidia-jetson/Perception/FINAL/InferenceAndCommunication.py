# This file is responsible for launching three threads, camera capture, object inference, communication
import pyrealsense2 as rs
import socket
import struct
import time
import numpy as np
import cv2
from ultralytics import YOLO
import threading
import collections
import torch
from collections import deque

host = '192.168.0.182'
port = 8001
FRAME_SKIP = 10
COLOR_W, COLOR_H = 1280, 720
RES_W, RES_H = 1280, 720
DEP_W, DEP_H = 1280, 720

frame_queue = collections.deque(maxlen=2)
latest_coord = {"status": False, "x": 0.0, "y": 0.0, "z": 0.0}
coord_lock = threading.Lock()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO("/home/nvidia/freshies/SeDriCa-BatMobile-v2/src/batmobile_perception/best.pt")
model.fuse()
model.to(device)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, COLOR_W, COLOR_H, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, DEP_W, DEP_H, rs.format.z16, 30)

align = rs.align(rs.stream.color)
pipeline.start(config)
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

def camera_thread():
    print("[INFO] Camera thread started.")
    while True:
        try:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            frame_queue.appendleft((color_image, depth_frame))

        except Exception as e:
            print(f"[ERROR] Camera thread: {e}")
        time.sleep(0.01)

frame_counter = 0
def inference_thread():
    global frame_counter
    print("[DEBUG] Inference thread started")
    while True:
        if len(frame_queue) == 0:
            continue

        color_image, depth_frame = frame_queue.pop()
        frame_counter += 1

        if frame_counter % FRAME_SKIP != 0:
            continue

        try:
            start_time = time.time()
            small_img = cv2.resize(color_image, (RES_W, RES_H))
            rgb_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
            depth_image = np.asanyarray(depth_frame.get_data())

            results = model(rgb_img, verbose=False)

            found = False
            for r in results:
                boxes = r.boxes.xywh.cpu().numpy()
                for (xc, yc, w, h) in boxes:
                    xc, yc = int(xc), int(yc)
                    xc_full = int(xc * (COLOR_W / RES_W))
                    yc_full = int(yc * (COLOR_H / RES_H))

                    if 0 <= xc_full < depth_frame.get_width() and 0 <= yc_full < depth_frame.get_height():
                        depth_val = depth_frame.get_distance(xc_full, yc_full)
                        if depth_val > 0:
                            X, Y, Z = rs.rs2_deproject_pixel_to_point(
                                color_intrinsics, [xc_full, yc_full], depth_val
                            )
                            with coord_lock:
                                latest_coord.update({"status": True, "x": X, "y": Y, "z": Z})
                            print(f"[INFO] Object at ({X:.2f}, {Y:.2f}, {Z:.2f})")
                            found = True
                            break
                if found:
                    break

            if not found:
                with coord_lock:
                    latest_coord.update({"status": False, "x": 0.0, "y": 0.0, "z": 0.0})
                print("[INFO] No object detected")

            fps = 1.0 / (time.time() - start_time + 1e-6)
            print(f"[DEBUG] Processed frame #{frame_counter} | FPS: {fps:.2f}")

        except Exception as e:
            print(f"[ERROR] Inference thread: {e}")

def connect_socket():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while True:
        try:
            client_socket.connect((host, port))
            print(f"[INFO] Connected to server at {host}:{port}")
            return client_socket
        except socket.error as e:
            print(f"[WARN] Connection failed: {e}, retrying...")
            time.sleep(2)

def transmission_thread():
    client_socket = connect_socket()
    while True:
        try:
            with coord_lock:
                data = latest_coord.copy()

            data_str = f"({data['x']:.2f},{data['y']:.2f},{data['z']:.2f})"
            header = struct.pack('!I', len(data_str))
            client_socket.sendall(header + data_str.encode())

        except socket.error as e:
            print(f"[ERROR] Send failed: {e}, reconnecting in 1 second...")
            client_socket = connect_socket()
            time.sleep(1)

        except Exception as e:
            print(f"[ERROR] Transmission thread: {e}")
        time.sleep(0.1)  # avoid overloading CPU

try:
    print("[INFO] Starting camera thread...")
    threading.Thread(target=camera_thread, daemon=True).start()

    time.sleep(2)

    print("[INFO] Starting inference thread...")
    threading.Thread(target=inference_thread, daemon=True).start()

    print("[INFO] Starting transmission thread...")
    threading.Thread(target=transmission_thread, daemon=True).start()

    print("[INFO] Entering main monitoring loop. Press Ctrl+C to exit.")
    while True:
        with coord_lock:
            data = latest_coord.copy()
        if data["status"]:
            print(f"[MAIN] Object at ({data['x']:.2f}, {data['y']:.2f}, {data['z']:.2f})")
        else:
            print("[MAIN] No object detected")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("[INFO] Interrupted by user")

finally:
    pipeline.stop()
    print("[INFO] Resources released. Exiting.")
