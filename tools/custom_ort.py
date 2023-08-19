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

import onnx
import onnxruntime as ort

print("import done")
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
    
    parser.add_argument('--p', help='dir for pretrained model', default='../pretrained_models/cityscapes/PIDNet_S_Cityscapes_test_onnx.onnx', type=str)
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
    images_list = glob.glob(args.r+'*'+args.t)
    sv_path = args.r+'outputs_ort/'
    
    # model = models.pidnet.get_pred_model(args.a, 19 if args.c else 11)
    # model = load_pretrained(model, args.p).cuda()
    # model.eval()
    print("before warmup")
    image_ortvalue = ort.OrtValue.ortvalue_from_numpy(np.zeros((1,3,1024,2048)), 'cuda', 0)
    # image_ortvalue = ort.OrtValue.ortvalue_from_numpy(np.zeros((1,3,1024,2048)))
    session = ort.InferenceSession(args.p, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    io_binding = session.io_binding()
    io_binding.bind_input(name='input', device_type=image_ortvalue.device_name(),
                          device_id=0, element_type=np.float32,
                          shape=image_ortvalue.shape(), buffer_ptr=image_ortvalue.data_ptr())
    io_binding.bind_output('output')
    print("after init")

    ################################# warm up!!
    torch.cuda.nvtx.range_push("model warm up")
    for i in range(5):
        session.run_with_iobinding(io_binding)
    torch.cuda.nvtx.range_pop()
    print("before inference")
    with torch.no_grad():
        for img_path in images_list:
            # print("img_path: ", img_path)
            img_name = img_path.split("/")[-1]
            img = cv2.imread(os.path.join(args.r, img_name),
                               cv2.IMREAD_COLOR)
            sv_img = np.zeros_like(img).astype(np.uint8)
            torch.cuda.nvtx.range_push("pre-processing")
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0)
            image_ortvalue = ort.OrtValue.ortvalue_from_numpy(img.numpy(), 'cuda', 0)
            io_binding.bind_input(name='input', device_type=image_ortvalue.device_name(),
                          device_id=0, element_type=np.float32,
                          shape=image_ortvalue.shape(), buffer_ptr=image_ortvalue.data_ptr())
            # print("img shape: ", img.size())
            torch.cuda.nvtx.range_pop()

            torch.cuda.nvtx.range_push("inference")
            session.run_with_iobinding(io_binding)
            torch.cuda.nvtx.range_pop()

            
            torch.cuda.nvtx.range_push("post-processing")
            pred = io_binding.copy_outputs_to_cpu()[0]
            pred = F.interpolate(torch.from_numpy(pred), size=img.size()[-2:], 
                                 mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).numpy()
            
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            sv_img = Image.fromarray(sv_img)
            torch.cuda.nvtx.range_pop()

            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            print("save path: ", sv_path, img_name)
            sv_img.save(sv_path+img_name)
            
            
            
        
        