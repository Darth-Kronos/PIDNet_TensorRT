# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


TRT_LOGGER = trt.Logger()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument('--p', help='dir for pretrained model', default='../pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.engine', type=str)
    parser.add_argument('--r', help='root or dir for input images', default='../samples/', type=str)
    parser.add_argument('--t', help='the format of input images (.jpg, .png, ...)', default='.png', type=str)     

    args = parser.parse_args()

    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_engine(engine_file_path):
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

if __name__ == '__main__':
    args = parse_args()
    images_list = glob.glob(args.r+'*'+args.t)
    sv_path = args.r+'outputs_trt/'
    print(images_list)
    #Load TensorRT model
    
    engine_file = args.p
    

    for img_path in images_list:
        img_name = img_path.split("\\")[-1]
        img = cv2.imread(os.path.join(args.r, img_name),
                        cv2.IMREAD_COLOR)
        sv_img = np.zeros_like(img).astype(np.uint8)
        img = input_transform(img)
        img = img.transpose((2, 0, 1)).copy()
        img = torch.from_numpy(img).unsqueeze(0)
        img_tens = img.clone()
        print("input image dtype: ", img.numpy().dtype)

        with load_engine(engine_file) as engine:
            with engine.create_execution_context() as context:
                # Set input shape based on image dimensions for inference
                context.set_binding_shape(engine.get_binding_index("input"), (1, 3, 1024, 2048))
                # Allocate host and device buffers
                bindings = []
                count = 0
                for binding in engine:
                    count += 1
                    binding_idx = engine.get_binding_index(binding)
                    size = trt.volume(context.get_binding_shape(binding_idx))
                    dtype = trt.nptype(engine.get_binding_dtype(binding))
                    if engine.binding_is_input(binding):
                        input_buffer = np.ascontiguousarray(img.numpy())
                        input_memory = cuda.mem_alloc(img.numpy().nbytes)
                        bindings.append(int(input_memory))
                    else:
                        print("size and dtype: ", size, dtype)
                        output_buffer = cuda.pagelocked_empty(size, dtype)
                        output_memory = cuda.mem_alloc(output_buffer.nbytes)
                        bindings.append(int(output_memory))
                print("count: ", count)
                stream = cuda.Stream()
                # Transfer input data to the GPU.
                cuda.memcpy_htod_async(input_memory, input_buffer, stream)
                # Run inference
                context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
                # Transfer prediction output from the GPU.
                cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
                # Synchronize the stream
                stream.synchronize()
        pred = torch.from_numpy(np.reshape(output_buffer, (1,19,128,256)))
        # print("pred and its size: ", pred, pred.size())
        pred = F.interpolate(pred, size=(1024,2048), 
                            mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        # print("after argmax: ", pred)
        for i, color in enumerate(color_map):
            for j in range(3):
                sv_img[:,:,j][pred==i] = color_map[i][j]
        print(sv_img)
        sv_img = Image.fromarray(sv_img)
        # print(img_name)
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        sv_img.save(sv_path+img_name.split("/")[-1])
            
            
            
        
        