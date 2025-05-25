import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def preprocess_image(image, input_shape):
    # input_shape is (batch, channels, height, width)
    _, _, H, W = input_shape                   # H=640, W=640
    image_resized = cv2.resize(image, (W, H))  # (width, height)
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_np = image_rgb.astype(np.float32) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))  # CHW
    image_np = np.expand_dims(image_np, axis=0)   # BCHW
    return np.ascontiguousarray(image_np)



def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        import numpy as np # monkey patch numpy
        if not hasattr(np, 'bool'):
            np.bool = bool
        if not hasattr(np, 'bool'):
            trt.nptype[trt.DataType.BOOL] = np.bool_
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()
    return [out['host'] for out in outputs]

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def nms(detections, iou_threshold=0.5):
    detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
    keep = []

    while detections:
        current = detections.pop(0)
        keep.append(current)
        detections = [
            d for d in detections
            if d['class_id'] != current['class_id'] or
               iou(current['bbox'], d['bbox']) < iou_threshold
        ]
    return keep


def postprocess_outputs(output, conf_threshold=0.4, iou_threshold=0.5):
    output = output[0].reshape(-1, 6)  # [x1, y1, x2, y2, conf, class]
    result = []
    for det in output:
        conf = det[4]/100
        if conf >= conf_threshold:
            x1, y1, x2, y2 = det[:4]
            class_id = int(det[5])
            result.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'class_id': class_id
            })
    result = nms(result, iou_threshold)
    return result
