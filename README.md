# PIDNet_TensorRT

This repository provides a step-by-step guide and code for optimizing a state-of-the-art semantic segmentation model using TorchScript, ONNX, and TensorRT.

## Prerequisites
### Device: RTX 3050
* CUDA: 12.0 (driver: 525) 
* cuDNN: 8.9
* TensorRT: 8.6

### Device: NVIDIA Jetson Nano
* Jetpack: 4.6.2
## Usage
### 0. Setup
* Clone this repository and download the pretrained model from the  official [PIDNet](https://github.com/XuJiacong/PIDNet/tree/main) repository. 

### 1. Export the model
For TorchScript:
````bash
python tools/export.py --a pidnet-s --p ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt --f torchscript
````
For ONNX:
````bash
python tools/export.py --a pidnet-s --p ./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt --f onnx
````
For TensorRT (using the above ONNX model):
```bash
trtexec --onnx=path/to/onnx/model --saveEngine=path/to/engine 
```
### 2. Inference

### 3. Speed Measurement
* Measure the inference speed of PIDNet-S for Cityscapes:
````bash
python models/speed/pidnet_speed.py --f all
````
|             | FPS         | % increase |
| :---------- | :---------: |:---------: |
| PyTorch     | 24.72       | -          |
| TorchScript | 27.09       | 9.59       |
| ONNX        | 33.52       | 35.60      |
| TensorRT    | 32.93       | 33.21      |

speed test is performed on a single Nvidia GeForce RTX 3050 GPU

### Acknowledgement
1. [PIDNet](https://github.com/XuJiacong/PIDNet/tree/main)

