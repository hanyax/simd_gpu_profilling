# simd_gpu_profiling
Here is the code and steps for GPU profiling on Tandom processor paper quantized models.

## Requirement
CUDA=11.4, cuDNN=8.2.4, onnxruntime-gpu=1.11 based on https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

# Profilling Steps:
## Imagenet
```console
cd imagenet
python3 imagenet_gpu_profiling.py ./models/resnet50-fakequant.onnx
python3 imagenet_gpu_profiling.py ./models/efficient-fakequant.onnx
python3 imagenet_gpu_profiling.py ./models/mobilenet-fakequant.onnx 
python3 imagenet_gpu_profiling.py ./models/vgg-fakequant.onnx
```

## Yolo
```console
cd yolo3
unzip val2017.zip
python3 yolo3_gpu_profiling.py
```

## Bert
```console
cd bert
python3 imagenet_gpu_profiling.py
python3 bert_gpu_profiling.py
```

The json files should be under each run directory. Just commit those json files and push to github and we will take it from there.
Thanks!