#### 在管理后台运行计算任务后，注意看logs，如果出现
```javascript
step000 ------>  size:640x320 fps:24.0 total_frame: 495
Traceback (most recent call last):
  File "yolov8_detect_tensorrt.py", line 381, in <module>
    main(args)
  File "yolov8_detect_tensorrt.py", line 174, in main
    bboxes, scores, labels = det_postprocess(data)
  File "/data/app/yolo/tensorrt_infer/models/torch_utils.py", line 63, in det_postprocess
    assert len(data) == 4
AssertionError
```
或者
```javascript
[02/05/2026-16:48:31] [TRT] [E] 6: The engine plan file is generated on an incompatible device, expecting compute 8.9 got compute 7.5, please rebuild.
[02/05/2026-16:48:31] [TRT] [E] 4: [runtime.cpp::deserializeCudaEngine::66] Error Code 4: Internal Error (Engine deserialization failed.)
Traceback (most recent call last):
  File "yolov8_detect_tensorrt.py", line 417, in <module>
    main(args)
  File "yolov8_detect_tensorrt.py", line 80, in main
    Engine = TRTModule(args.engine, device)
  File "/data/app/yolo/tensorrt_infer/models/engine.py", line 218, in __init__
    self.__init_engine()
  File "/data/app/yolo/tensorrt_infer/models/engine.py", line 227, in __init_engine
    context = model.create_execution_context()
AttributeError: 'NoneType' object has no attribute 'create_execution_context'
```

#### 本质是 TensorRT 引擎的输出格式与代码期望不匹配，需要重新构建，执行 `rebuild_engine.py`，重新上传覆盖模型引擎
