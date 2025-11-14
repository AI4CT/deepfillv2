#!/bin/bash

# 优化后的训练脚本 - 针对GPU利用率优化
python train.py \
--baseroot '/mnt/lustre/zhaoyuzhi/dataset/ILSVRC2012_train_256' \
--pre_train True \
--multi_gpu True \
--checkpoint_interval 5 \
--finetune_path './models/GrayInpainting_epoch10_batchsize16.pth' \
--multi_gpu True \
--epochs 500 \
--batch_size 32 \
--lr_g 1e-4 \
--lambda_l1 1 \
--lambda_perceptual 5 \
--lambda_gan 0.1 \
--lr_decrease_epoch 10 \
--lr_decrease_factor 0.5 \
--num_workers 16 \
--imgsize 128 \
--mask_type 'free_form' \
--margin 10 \
--mask_num 20 \
--bbox_shape 30 \
--max_angle 4 \
--max_len 40 \
--max_width 2 \
--gan_param 0.01 \
--dataset_mode 'bubble' \
--train_root '/home/yubd/mount/dataset/dataset_overlap/training_dataset20251111' \
--val_root '/home/yubd/mount/dataset/dataset_overlap/val' \
--test_root '/home/yubd/mount/dataset/dataset_overlap/test' \
--eval_interval 10 \
--cudnn_benchmark True
