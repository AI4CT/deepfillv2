# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 12:03:52 2018

@author: yzzhao2
"""

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from network import GrayInpaintingNet
import argparse

def forward(size, root, model):
    # pre-processing, let all the images are in RGB color space
    img = Image.open(root)
    img = img.resize((size, size), Image.LANCZOS).convert('RGB')
    img = np.array(img).astype(np.float64)
    
    # convert to grayscale
    img = np.mean(img, axis=2, keepdims=True)
    
    # define a mask
    mask = np.zeros([size, size, 1], dtype = np.float64)
    if size == 144:
        center = np.ones([100, 100, 1], dtype = np.float64)
        mask[22:122, 22:122, :] = center
    elif size == 200:
        center = np.ones([144, 144, 1], dtype = np.float64)
        mask[28:172, 28:172, :] = center
    elif size == 256:
        center = np.ones([200, 200, 1], dtype = np.float64)
        mask[28:228, 28:228, :] = center
    elif size == 128:
        center = np.ones([80, 80, 1], dtype = np.float64)
        mask[24:104, 24:104, :] = center
    
    maskimg = (img * mask) / 255
    maskimg = maskimg.astype(np.float32)
    
    # save masking as image
    mask_img = Image.fromarray((maskimg[...,0] * 255).astype(np.uint8)).convert('L')
    mask_img = mask_img.resize((size, size), Image.NEAREST)
    mask_img.save(root.split('.')[0] + '_mask.jpg')
    
    # save masking as image
    maskimg = transforms.ToTensor()(maskimg)
    maskimg = maskimg.reshape([1, 1, size, size])
    mask = mask.astype(np.float32)
    mask = transforms.ToTensor()(mask)
    mask = mask.reshape([1, 1, size, size])
    
    # move to GPU
    maskimg = maskimg.cuda()
    mask = mask.cuda()
    
    # get the output
    output = model(maskimg, mask)
    
    # transfer to image
    output = output.cpu().detach().numpy().reshape([1, size, size])
    output = output.transpose(1, 2, 0)
    output = output * 255
    output = np.array(output, dtype = np.uint8)
    return output

if __name__ == "__main__":

    size = 128
    root = '/home/yubd/mount/codebase/deepfillv2/images/origin result  images/COCO_test2014_000000000027.jpg'
    
    # create model options
    opt = argparse.Namespace()
    opt.in_channels = 1
    opt.out_channels = 1
    opt.mask_channels = 1
    opt.latent_channels = 64
    opt.pad = 'reflect'
    opt.activ_g = 'lrelu'
    opt.activ_d = 'lrelu'
    opt.norm_g = 'in'
    opt.norm_d = 'bn'
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    
    # create model
    model = GrayInpaintingNet(opt)
    
    # load state dict
    checkpoint = torch.load('/home/yubd/mount/codebase/deepfillv2/deepfillv2-grayscale/models/GrayInpainting_epoch20_batchsize16.pth', weights_only=True)
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()

    output = forward(size, root, model)
    img = Image.fromarray(output.squeeze(), mode='L')
    # img.show()
    
    # save the result
    output_path = '/home/yubd/mount/codebase/deepfillv2/deepfillv2-grayscale/test_output.jpg'
    img.save(output_path)
    print(f"测试完成！结果已保存到: {output_path}")
