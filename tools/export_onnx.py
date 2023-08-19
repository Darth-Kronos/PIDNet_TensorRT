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
# import torch_tensorrt

def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    
    parser.add_argument('--a', help='pidnet-s, pidnet-m or pidnet-l', default='pidnet-s', type=str)
    parser.add_argument('--c', help='cityscapes pretrained or not', type=bool, default=True)
    parser.add_argument('--p', help='pretrained model path', default='../pretrained_models/cityscapes/PIDNet_S_Cityscapes_test.pt', type=str)
    parser.add_argument('--o', help='output file path', default='../pretrained_models/cityscapes/PIDNet_S_Cityscapes_test_onnx.onnx', type=str)
    args = parser.parse_args()
    return args

def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict = False)
    
    return model


if __name__ == '__main__':
    args = parse_args()
    model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)
    model = load_pretrained(model, args.p).cuda().eval()
    input = torch.randn(1, 3, 1024, 2048).cuda()
    torch.onnx.export(model, input, args.o, verbose=False,input_names=["input"],
                      output_names=["output"])
    # trt_model_fp32 = torch_tensorrt.compile(module, inputs=[input], enabled_precisions = torch.float32, workspace_size = 1 << 22)
    # trt_model_fp32 = torch_tensorrt.compile(module, inputs=[torch_tensorrt.Input((1, 3, 1024, 2048))],
    #                                         enabled_precisions = torch.float32, workspace_size = 1 << 22,
    #                                         truncate_long_and_double = True)
    # torch.jit.save(trt_model_fp32, args.o)
    print("model saved!")
