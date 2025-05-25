"""
this code is for detecting object using model.py , calculating depth and  pose relative to the camera
"""

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

img_w=640
img_h=480
#model = YOLO("../best.pt")
model = YOLO("/home/nvidia/freshies/SeDriCa-BatMobile-v2/src/batmobile_perception/best.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, img_w, img_h, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, img_w, img_h, rs.format.z16, 30)


align_to = rs.stream.color
align = rs.align(align_to)


pipeline.start(config)
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
color_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
def get_location():
    try:
        frames = pipeline.wait_for_frames()
        
        
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame: return (0,0,0,0)
            
       
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

       
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        
        result = model(rgb_image,conf=0.2)

        for r in result:  # result is a list of Results objects

            boxes=r.boxes.xywh.cpu().numpy()
            confidences = r.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = r.boxes.cls.cpu().numpy()  # Class IDs

            
            for vector,conf,clid in zip(boxes,confidences,class_ids):  # Access bounding boxes

                xc,yc,w,h=vector
                xc, yc = int(xc), int(yc)

           
                if 0 <= xc < depth_image.shape[1] and 0 <= yc < depth_image.shape[0]:
                    depth_value = depth_frame.get_distance(xc, yc)

               
                    X,Y,Z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [xc, yc], depth_value)

                    return (1,X,Y,Z)
            return (1.0,0.0,0.0,0.0)
    except:
        return (0.0,0.0,0.0,0.0)

        
        
def main():
    print("Locate function is running!")

if __name__ == "__main__":
    main()
