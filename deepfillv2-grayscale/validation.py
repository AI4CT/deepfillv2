import argparse
import os
import torch
import numpy as np
import cv2

import utils
from bubble_dataset import BubbleInpaintDataset

if __name__ == "__main__":
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    # General parameters
    parser.add_argument('--pre_train', type = bool, default = True, help = 'the type of GAN for training')
    parser.add_argument('--finetune_path', type = str, \
        default = "./models/GrayInpainting_epoch20_batchsize8.pth", \
            help = 'the load name of models')
    parser.add_argument('--val_path', type = str, default = "./validation", help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--test_batch_size', type = int, default = 1, help = 'test batch size')
    parser.add_argument('--num_workers', type = int, default = 1, help = 'num of workers')
    # Network parameters
    parser.add_argument('--in_channels', type = int, default = 1, help = 'input RGB image')
    parser.add_argument('--out_channels', type = int, default = 1, help = 'output RGB image')
    parser.add_argument('--mask_channels', type = int, default = 1, help = 'input mask')
    parser.add_argument('--latent_channels', type = int, default = 64, help = 'latent channels')
    parser.add_argument('--pad', type = str, default = 'reflect', help = 'the padding type')
    parser.add_argument('--activ_g', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--activ_d', type = str, default = 'lrelu', help = 'the activation type')
    parser.add_argument('--norm_g', type = str, default = 'in', help = 'the normalization type of generator')
    parser.add_argument('--norm_d', type = str, default = 'bn', help = 'the normalization type of discriminator')
    parser.add_argument('--init_type', type = str, default = 'normal', help = 'the initialization type')
    parser.add_argument('--init_gain', type = float, default = 0.02, help = 'the initialization gain')
    # Dataset parameters for bubble mode
    parser.add_argument('--imgsize', type = int, default = 128, help = 'size of image')
    parser.add_argument('--dataset_mode', type = str, default = 'bubble', help = 'dataset pipeline to use')
    parser.add_argument('--test_root', type = str, required = True, help = 'root directory for test triplets')
    
    opt = parser.parse_args()
    print(opt)

    # ----------------------------------------
    #                   Test
    # ----------------------------------------
    # Initialize
    generator = utils.create_generator(opt).cuda()
    
    # Use bubble dataset for validation
    test_dataset = BubbleInpaintDataset(opt.test_root, opt, phase='test', return_paths=True, strict=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = opt.test_batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    utils.check_path(opt.val_path)

    print(f'Loaded {len(test_dataset)} test samples')

    # forward
    for i, (grayscale, mask, origin, imgname) in enumerate(test_loader):

        # To device
        grayscale = grayscale.cuda()                                        # out: [B, 1, 256, 256]
        mask = mask.cuda()                                                  # out: [B, 1, 256, 256]
        print(i, imgname[0])

        # Forward propagation
        with torch.no_grad():
            fake_target = generator(grayscale, mask)                        # out: [B, 1, 256, 256]
            out_whole = grayscale * (1 - mask) + fake_target * mask

        # Save the repaired image
        fake_target_np = out_whole.clone().data.permute(0, 2, 3, 1)[0, :, :, :].cpu().numpy()
        fake_target_np = (fake_target_np * 255.0).astype(np.uint8)
        fake_target_np = np.squeeze(fake_target_np)  # Remove channel dimension for grayscale
        
        # Extract original filename from path
        original_name = os.path.basename(imgname[0]).replace('_blocked', '_repaired')
        save_img_path = os.path.join(opt.val_path, original_name)
        cv2.imwrite(save_img_path, fake_target_np)
        print(f"Saved repaired image to: {save_img_path}")