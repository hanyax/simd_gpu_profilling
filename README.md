# simd_gpu_profiling
Here is the code and steps for GPU profiling on Tandom processor paper quantized models.

## Requirement
CUDA=11.4, cuDNN=8.2.4, onnxruntime-gpu=1.11 based on https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html \
Also need git-lfs, on Ubuntu I believe ```sudo apt-get -y install git-lfs``` should do it

# Profilling Steps:
## Download large files:
After clone the repo, do:
```console
git lfs fetch --all
git lfs pull
```

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
unzip val2017.zip -d ./val2017
python3 yolo3_gpu_profiling.py
```

## Bert
```console
cd bert
python3 bert_gpu_profiling.py
```

## Zip and commit onnxfile
From base directory:
```console
zip runlog_json.zip ./imagenet/onnxruntime*.json ./yolov3/onnxruntime*.json ./bert/onnxruntime*.json
```
