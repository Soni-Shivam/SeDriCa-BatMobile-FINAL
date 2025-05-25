import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX model
with open("best.onnx", "rb") as f:
    if not parser.parse(f.read()):
        print("Failed to parse the ONNX file.")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()

# Builder config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.set_flag(trt.BuilderFlag.FP16)

# Use new API to build and serialize
serialized_engine = builder.build_serialized_network(network, config)

# Save engine
with open("best_fp16.trt", "wb") as f:
    f.write(serialized_engine)
print("Engine saved as best_fp16.trt (serialized directly)")
