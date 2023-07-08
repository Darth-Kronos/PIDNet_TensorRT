# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import time
import os
import numpy as np
import _init_paths
import models
import torch
import torch.nn.functional as F
from PIL import Image
import torch_tensorrt

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
    parser = argparse.ArgumentParser(description='demo of video inference')
    parser.add_argument('--model', help='pretrained model path', default='./pretrained_models/cityscapes/PIDNet_S_Cityscapes_test_trt.ts', type=str)
    parser.add_argument('--input', help='input video path', default='/workspace/PIDNet_trt/PIDNet/samples/sample_video3.webm', type=str)
    args = parser.parse_args()
    return args

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

if __name__ == '__main__':
    args = parse_args()
    model = torch.jit.load(args.model)
    video_cap = cv2.VideoCapture(args.input)
    frame_width = int(video_cap.get(3))
    frame_height = int(video_cap.get(4))
    video_fps = int(video_cap.get(5))

    result_path = args.input.split(".")[0] + "_out_PIDNet_S_Cityscapes_test_trt.mp4"
    result = cv2.VideoWriter(result_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frame_width, frame_height))
    print(video_cap.isOpened())
    while(video_cap.isOpened()):
        ret, img = video_cap.read()
        if (ret):
            img = cv2.resize(img, (2048, 1024))
            pred_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            pred = model(img)
            pred = F.interpolate(pred, size=img.size()[-2:], 
                                    mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    pred_img[:,:,j][pred==i] = color_map[i][j]
            pred_img = cv2.resize(pred_img, (frame_width, frame_height))
            result.write(pred_img)
        else:
            break

            
            
            
        
        