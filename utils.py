import os
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv

import network
import dataset

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_generator(opt):
    # Initialize the networks
    generator = network.GrayInpaintingNet(opt)
    print('Generator is created!')
    # Init the networks
    if opt.finetune_path:
        pretrained_net = torch.load(opt.finetune_path, weights_only=True)
        generator = load_dict(generator, pretrained_net)
        print('Load generator with %s' % opt.finetune_path)
    else:
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator

def create_perceptualnet():
    # Pre-trained VGG-16
    vgg16 = torch.load('/home/yubd/mount/codebase/deepfillv2/deepfillv2-grayscale/models/vgg16-397923af.pth', weights_only=True)
    # Get the first 16 layers of vgg16, which is conv3_3
    perceptualnet = network.PerceptualNet()
    # Update the parameters
    load_dict(perceptualnet, vgg16)
    # It does not gradient
    for param in perceptualnet.parameters():
        param.requires_grad = False
    return perceptualnet

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net
    
# ----------------------------------------
#             PATH processing
# ----------------------------------------
def text_readlines(filename):
    # Try to read a txt file and return a list.Return [] if there was a mistake.
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    # This for loop deletes the EOF (like \n)
    for i in range(len(content)):
        content[i] = content[i][:len(content[i])-1]
    file.close()
    return content

def savetxt(name, loss_log):
    np_loss_log = np.array(loss_log)
    np.savetxt(name, np_loss_log)

def get_files(path):
    # read a folder, return the complete path
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(os.path.join(root, filespath))
    return ret

def get_jpgs(path):
    # read a folder, return the image name
    ret = []
    for root, dirs, files in os.walk(path):
        for filespath in files:
            ret.append(filespath)
    return ret

def text_save(content, filename, mode = 'a'):
    # save a list to a txt
    # Try to save a list variable in txt file.
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# ----------------------------------------
#    Validation and Sample at training
# ----------------------------------------
def sample(grayscale, mask, out, save_folder, epoch):
    grayscale = grayscale[0, :, :, :].data.cpu().numpy()
    mask = mask[0, :, :, :].data.cpu().numpy()
    out = out[0, :, :, :].data.cpu().numpy()

    def _to_vis(arr: np.ndarray, repeat_channels: bool = True) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape[2] == 1 and repeat_channels:
            arr = np.repeat(arr, 3, axis=2)
        arr = np.clip(arr, 0.0, 1.0)
        return (arr * 255).astype(np.uint8)

    masked_img = grayscale * (1 - mask) + mask
    grayscale_vis = _to_vis(grayscale)
    mask_vis = _to_vis(mask)
    masked_vis = _to_vis(masked_img)
    out_vis = _to_vis(out)

    img = np.concatenate((grayscale_vis, mask_vis, masked_vis, out_vis), axis=1)
    os.makedirs(save_folder, exist_ok=True)
    imgname = os.path.join(save_folder, f"{epoch}.png")
    cv2.imwrite(imgname, img)


def sample_triplet(blocked, mask, recon, origin, save_folder, epoch, prefix='val'):
    """Save visualization grid for bubble dataset triplets."""

    blocked = blocked[0].numpy()
    mask = mask[0].numpy()
    recon = recon[0].numpy()
    origin = origin[0].numpy()

    def _prepare(arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        elif arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        arr = np.clip(arr, 0.0, 1.0)
        if is_mask:
            return (arr * 255).astype(np.uint8)
        return (arr * 255).astype(np.uint8)

    blocked_vis = _prepare(blocked)
    mask_vis = _prepare(mask, is_mask=True)
    recon_vis = _prepare(recon)
    origin_vis = _prepare(origin)

    grid = np.concatenate((blocked_vis, mask_vis, recon_vis, origin_vis), axis=1)

    save_dir = Path(save_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = save_dir / f"{prefix}_{epoch}.png"
    cv2.imwrite(str(filename), grid)
    
def psnr(pred, target, pixel_max_cnt = 255):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    ssim = skimage.measure.compare_ssim(target, pred, multichannel = True)
    return ssim
