import tensorrt as trt
import os

print("构建带TensorRT引擎...")

def build_engine(onnx_path, engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    # 启用FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用FP16加速")

    # 解析ONNX
    print(f"解析ONNX: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("解析失败")
            for i in range(parser.num_errors):
                print(f"错误 {i}: {parser.get_error(i)}")
            return False

    print("✓ ONNX解析成功")

    # 获取网络输出
    num_outputs = network.num_outputs
    print(f"网络输出数量: {num_outputs}")
    print("构建引擎...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("构建失败")
        return False

    # 保存引擎
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    print(f"✓ 引擎保存到: {engine_path}")
    return True

if __name__ == "__main__":
    onnx_file = "yolov8n.onnx"
    engine_file = "yolov8n.engine"

    if not os.path.exists(onnx_file):
        print(f"ONNX文件不存在: {onnx_file}")
        exit(1)

    success = build_engine(onnx_file, engine_file)
    if success:
        print("\n✅ 构建成功!")
        print(f"使用命令: python yolov8_detect_tensorrt.py --engine {engine_file}")
    else:
        print("\n❌ 构建失败")
