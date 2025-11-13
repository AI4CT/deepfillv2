import time
import datetime
import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

import network
import dataset
import utils
from bubble_dataset import BubbleInpaintDataset


def _compute_batch_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute the average PSNR (in dB) for a batch of single-channel images."""

    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    eps = 1e-8
    psnr = 20.0 * torch.log10(1.0 / torch.sqrt(mse + eps))
    return psnr.mean().item()


def _compute_batch_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return mse.mean().item()


def _compute_batch_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    total = 0.0
    count = pred_np.shape[0]
    for p, t in zip(pred_np, target_np):
        p_img = np.clip(p.squeeze(), 0.0, 1.0)
        t_img = np.clip(t.squeeze(), 0.0, 1.0)
        total += structural_similarity(t_img, p_img, data_range=1.0)
    return total / max(count, 1)


def _ensure_dir(path: Sequence[str]) -> None:
    for p in path:
        Path(p).parent.mkdir(parents=True, exist_ok=True)


def _create_timestamp_folders(base_paths: List[str]) -> Dict[str, str]:
    """Create timestamped subfolders for training outputs.
    
    Args:
        base_paths: List of base folder paths (logs, models, samples)
    
    Returns:
        Dictionary mapping base folder names to timestamped paths
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_paths = {}
    
    for base_path in base_paths:
        timestamped_path = os.path.join(base_path, timestamp)
        Path(timestamped_path).mkdir(parents=True, exist_ok=True)
        
        # Extract folder name from path for the dictionary key
        folder_name = os.path.basename(base_path.rstrip('/'))
        timestamped_paths[folder_name] = timestamped_path
        print(f"Created timestamped folder: {timestamped_path}")
    
    return timestamped_paths


def _write_metrics_csv(history: Dict[str, List[Optional[float]]], csv_path: str) -> None:
    csv_file = Path(csv_path)
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        'epoch',
        'train_loss', 'train_psnr', 'train_mse', 'train_ssim',
        'val_loss', 'val_psnr', 'val_mse', 'val_ssim',
        'test_loss', 'test_psnr', 'test_mse', 'test_ssim'
    ]

    with csv_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        num_rows = len(history['epoch'])
        for idx in range(num_rows):
            row = []
            for header in headers:
                datum = history.get(header, [None] * num_rows)[idx]
                if datum is None:
                    row.append('')
                elif header == 'epoch':
                    row.append(int(datum))
                else:
                    row.append(f"{datum:.6f}")
            writer.writerow(row)


def _save_category_samples(
    category_samples: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    base_path: str,
    epoch: int,
    dataset_type: str,
    dataset_mode: str
) -> None:
    """Save visualization samples for each category in organized folders."""
    if not category_samples:
        return
    
    # Create directory structure: base_path/epoch_{epoch}/{dataset_type}/
    epoch_dir = Path(base_path) / f"epoch_{epoch}" / dataset_type
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    for category, samples in category_samples.items():
        blocked, mask, out_whole, origin = samples
        
        # Convert tensors to numpy for visualization
        blocked_np = blocked[0].numpy()
        mask_np = mask[0].numpy()
        out_whole_np = out_whole[0].numpy()
        origin_np = origin[0].numpy()
        
        def _prepare(arr: np.ndarray, is_mask: bool = False) -> np.ndarray:
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.ndim == 2:
                arr = arr[:, :, None]
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            arr = np.clip(arr, 0.0, 1.0)
            return (arr * 255).astype(np.uint8)
        
        blocked_vis = _prepare(blocked_np)
        mask_vis = _prepare(mask_np, is_mask=True)
        out_whole_vis = _prepare(out_whole_np)
        origin_vis = _prepare(origin_np)
        
        # Create grid visualization
        grid = np.concatenate((blocked_vis, mask_vis, out_whole_vis, origin_vis), axis=1)
        
        # Save with category name
        filename = epoch_dir / f"{category}_{dataset_type}.png"
        cv2.imwrite(str(filename), grid)
        print(f"Saved {dataset_type} sample for category {category} at epoch {epoch} to {filename}")


def _plot_metrics(history: Dict[str, List[Optional[float]]], plot_path: str) -> None:
    if not history['epoch']:
        return

    epochs = history['epoch']
    fig, ((ax_loss, ax_mse), (ax_psnr, ax_ssim)) = plt.subplots(2, 2, figsize=(14, 10))

    def _plot_series(ax, values: List[Optional[float]], label: str) -> None:
        xs = [e for e, v in zip(epochs, values) if v is not None]
        ys = [v for v in values if v is not None]
        if xs:
            ax.plot(xs, ys, marker='o', label=label, linewidth=2)

    # L1 Loss subplot
    _plot_series(ax_loss, history['train_loss'], 'Train')
    _plot_series(ax_loss, history['val_loss'], 'Val')
    _plot_series(ax_loss, history['test_loss'], 'Test')
    ax_loss.set_xlabel('Epoch', fontsize=11)
    ax_loss.set_ylabel('L1 Loss', fontsize=11)
    ax_loss.set_title('L1 Loss', fontsize=12, fontweight='bold')
    ax_loss.legend(loc='best', fontsize=10)
    ax_loss.grid(True, linestyle='--', alpha=0.3)

    # MSE subplot
    _plot_series(ax_mse, history['train_mse'], 'Train')
    _plot_series(ax_mse, history['val_mse'], 'Val')
    _plot_series(ax_mse, history['test_mse'], 'Test')
    ax_mse.set_xlabel('Epoch', fontsize=11)
    ax_mse.set_ylabel('MSE', fontsize=11)
    ax_mse.set_title('Mean Squared Error', fontsize=12, fontweight='bold')
    ax_mse.legend(loc='best', fontsize=10)
    ax_mse.grid(True, linestyle='--', alpha=0.3)

    # PSNR subplot
    _plot_series(ax_psnr, history['train_psnr'], 'Train')
    _plot_series(ax_psnr, history['val_psnr'], 'Val')
    _plot_series(ax_psnr, history['test_psnr'], 'Test')
    ax_psnr.set_xlabel('Epoch', fontsize=11)
    ax_psnr.set_ylabel('PSNR (dB)', fontsize=11)
    ax_psnr.set_title('Peak Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
    ax_psnr.legend(loc='best', fontsize=10)
    ax_psnr.grid(True, linestyle='--', alpha=0.3)

    # SSIM subplot
    _plot_series(ax_ssim, history['train_ssim'], 'Train')
    _plot_series(ax_ssim, history['val_ssim'], 'Val')
    _plot_series(ax_ssim, history['test_ssim'], 'Test')
    ax_ssim.set_xlabel('Epoch', fontsize=11)
    ax_ssim.set_ylabel('SSIM', fontsize=11)
    ax_ssim.set_title('Structural Similarity Index', fontsize=12, fontweight='bold')
    ax_ssim.legend(loc='best', fontsize=10)
    ax_ssim.grid(True, linestyle='--', alpha=0.3)

    fig.suptitle('Training Metrics', fontsize=14, fontweight='bold', y=0.995)
    fig.tight_layout()
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)


def _build_datasets(opt) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.Dataset], Optional[torch.utils.data.Dataset]]:
    if opt.dataset_mode == 'bubble':
        trainset = BubbleInpaintDataset(opt.train_root, opt, phase='train', return_paths=False, strict=True)

        def _optional_dataset(root: str, phase: str) -> Optional[torch.utils.data.Dataset]:
            if not root:
                return None
            # Set return_paths=True for val and test to get category information
            ds = BubbleInpaintDataset(root, opt, phase=phase, return_paths=True, strict=False)
            return ds if len(ds) > 0 else None

        valset = _optional_dataset(opt.val_root, 'val')
        testset = _optional_dataset(opt.test_root, 'test')
    else:
        trainset = dataset.InpaintDataset(opt)
        valset = dataset.InpaintDataset_val(opt)
        testset = dataset.ValidationSet_with_Known_Mask(opt)

    return trainset, valset, testset


def _create_dataloader(ds: Optional[torch.utils.data.Dataset], batch_size: int, num_workers: int, shuffle: bool) -> Optional[DataLoader]:
    if ds is None or len(ds) == 0:
        return None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )


def _unpack_batch(batch, dataset_mode: str):
    if dataset_mode == 'bubble':
        if len(batch) == 4:
            blocked, mask, origin, _ = batch
        else:
            blocked, mask, origin = batch
        return blocked, mask, origin

    # Default datasets provide grayscale twice (train) or grayscale with filename (val/test)
    blocked = batch[0]
    mask = batch[1]
    origin = batch[0]
    return blocked, mask, origin


def _evaluate(
    generator: nn.Module,
    dataloader: Optional[DataLoader],
    criterion: nn.Module,
    dataset_mode: str,
    device: torch.device,
    collect_category_samples: bool = False,
) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float],
    Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]
]:
    if dataloader is None:
        return None, None, None, None, None, None

    generator.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_mse = 0.0
    total_ssim = 0.0
    total_count = 0
    sample_tensors = None
    category_samples = {} if collect_category_samples else None

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch and get category info if available
            if dataset_mode == 'bubble':
                if len(batch) == 4:
                    blocked, mask, origin, paths = batch
                    # Extract category from path if available
                    categories = []
                    for path in paths:
                        if isinstance(path, str):
                            # Try to extract category from path, e.g., .../0-10/Blocked/xxx_blocked.png
                            parts = path.split('/')
                            for i, part in enumerate(parts):
                                if part == 'Blocked' and i > 0:
                                    categories.append(parts[i-1])
                                    break
                            else:
                                categories.append(None)
                        else:
                            categories.append(None)
                else:
                    blocked, mask, origin = batch
                    categories = [None] * len(blocked)
            else:
                blocked = batch[0]
                mask = batch[1]
                origin = batch[0]
                categories = [None] * len(blocked)
            
            blocked = blocked.to(device)
            mask = mask.to(device)
            origin = origin.to(device)

            out = generator(blocked, mask)
            out_whole = blocked * (1 - mask) + out * mask

            loss = criterion(out_whole, origin)
            batch_size = blocked.size(0)

            total_loss += loss.item() * batch_size
            total_psnr += _compute_batch_psnr(out_whole, origin) * batch_size
            total_mse += _compute_batch_mse(out_whole, origin) * batch_size
            total_ssim += _compute_batch_ssim(out_whole, origin) * batch_size
            total_count += batch_size

            if sample_tensors is None:
                sample_tensors = (
                    blocked.detach().cpu(),
                    mask.detach().cpu(),
                    out_whole.detach().cpu(),
                    origin.detach().cpu(),
                )
            
            # Collect first sample for each category
            if collect_category_samples and category_samples is not None:
                for i, category in enumerate(categories):
                    if category is not None and category not in category_samples:
                        category_samples[category] = (
                            blocked[i].detach().cpu().unsqueeze(0),
                            mask[i].detach().cpu().unsqueeze(0),
                            out_whole[i].detach().cpu().unsqueeze(0),
                            origin[i].detach().cpu().unsqueeze(0),
                        )

    generator.train()

    if total_count == 0:
        return None, None, None, None, None, None

    avg_loss = total_loss / total_count
    avg_psnr = total_psnr / total_count
    avg_mse = total_mse / total_count
    avg_ssim = total_ssim / total_count
    return avg_loss, avg_psnr, avg_mse, avg_ssim, sample_tensors, category_samples


def _extract_epoch_from_checkpoint(checkpoint_path: str) -> int:
    """从checkpoint文件名中提取训练轮数"""
    filename = os.path.basename(checkpoint_path)
    # 匹配格式: BUDDY_epoch{epoch}_batchsize{batch_size}.pth
    import re
    match = re.search(r'epoch(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"无法从文件名 {filename} 中提取训练轮数")


def _find_previous_log_folder(checkpoint_path: str) -> str:
    """根据checkpoint路径找到对应的日志文件夹"""
    # checkpoint路径: /home/yubd/mount/codebase/deepfillv2/deepfillv2-grayscale/models/20251112_194524/BUDDY_epoch60_batchsize384.pth
    checkpoint_dir = os.path.dirname(checkpoint_path)  # models/20251112_194524
    timestamp = os.path.basename(checkpoint_dir)  # 20251112_194524
    
    # 构建logs路径: /home/yubd/mount/codebase/deepfillv2/deepfillv2-grayscale/logs/20251112_194524
    base_dir = os.path.dirname(os.path.dirname(checkpoint_dir))  # deepfillv2-grayscale
    log_dir = os.path.join(base_dir, 'logs', timestamp)
    
    return log_dir


def _load_previous_metrics_history(log_dir: str, resume_epoch: int) -> Dict[str, List[Optional[float]]]:
    """加载之前的训练历史数据，只保留到resume_epoch轮"""
    metrics_csv = os.path.join(log_dir, 'metrics_log.csv')
    
    if not os.path.exists(metrics_csv):
        print(f"警告: 未找到之前的metrics文件 {metrics_csv}，将从头开始记录")
        return _prepare_history()
    
    print(f"加载之前的训练历史: {metrics_csv}")
    history = _prepare_history()
    
    try:
        with open(metrics_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch_num = int(row['epoch']) if row['epoch'] else None
                # 只保留小于等于resume_epoch的数据
                if epoch_num is not None and epoch_num <= resume_epoch:
                    history['epoch'].append(epoch_num)
                    for key in ['train_loss', 'train_psnr', 'train_mse', 'train_ssim',
                               'val_loss', 'val_psnr', 'val_mse', 'val_ssim',
                               'test_loss', 'test_psnr', 'test_mse', 'test_ssim']:
                        if row[key]:
                            history[key].append(float(row[key]))
                        else:
                            history[key].append(None)
        
        print(f"成功加载 {len(history['epoch'])} 个epoch的历史数据")
        if history['epoch']:
            print(f"历史数据范围: 第1轮 到 第{history['epoch'][-1]}轮 (从第{resume_epoch}轮继续训练)")
        return history
    except Exception as e:
        print(f"加载历史数据失败: {e}，将从头开始记录")
        return _prepare_history()


def _prepare_history() -> Dict[str, List[Optional[float]]]:
    return {
        'epoch': [],
        'train_loss': [],
        'train_psnr': [],
        'train_mse': [],
        'train_ssim': [],
        'val_loss': [],
        'val_psnr': [],
        'val_mse': [],
        'val_ssim': [],
        'test_loss': [],
        'test_psnr': [],
        'test_mse': [],
        'test_ssim': [],
    }


def _append_history(
    history: Dict[str, List[Optional[float]]],
    epoch: int,
    train_loss: float,
    train_psnr: float,
    train_mse: float,
    train_ssim: float,
    val_loss: Optional[float],
    val_psnr: Optional[float],
    val_mse: Optional[float],
    val_ssim: Optional[float],
    test_loss: Optional[float],
    test_psnr: Optional[float],
    test_mse: Optional[float],
    test_ssim: Optional[float],
) -> None:
    history['epoch'].append(epoch)
    history['train_loss'].append(train_loss)
    history['train_psnr'].append(train_psnr)
    history['train_mse'].append(train_mse)
    history['train_ssim'].append(train_ssim)
    history['val_loss'].append(val_loss)
    history['val_psnr'].append(val_psnr)
    history['val_mse'].append(val_mse)
    history['val_ssim'].append(val_ssim)
    history['test_loss'].append(test_loss)
    history['test_psnr'].append(test_psnr)
    history['test_mse'].append(test_mse)
    history['test_ssim'].append(test_ssim)


def _curriculum_categories(available_categories: Sequence[str], epoch: int) -> Sequence[str]:
    if not available_categories:
        return []
    stage = min((epoch - 1) // 10 + 1, len(available_categories))
    return list(available_categories[:stage])


def Trainer(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    cudnn.benchmark = opt.cudnn_benchmark

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    effective_gpus = available_gpus if opt.multi_gpu and available_gpus > 0 else min(1, available_gpus)
    print("There are %d GPUs used" % effective_gpus)

    multiplier = effective_gpus if effective_gpus > 0 else 1
    opt.batch_size *= multiplier
    # Optimize num_workers: use more workers for multi-GPU training
    if effective_gpus > 0:
        # Use 4 workers per GPU, but cap at 32 to avoid diminishing returns
        opt.num_workers = min(opt.num_workers * effective_gpus, 32)
    else:
        opt.num_workers *= multiplier
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)

    # 断点续训相关变量
    resume_epoch = 0
    
    # 如果是断点续训模式
    if opt.resume_training and opt.resume_path:
        if os.path.exists(opt.resume_path):
            resume_epoch = _extract_epoch_from_checkpoint(opt.resume_path)
            print(f"检测到断点续训，将从第 {resume_epoch} 轮继续训练")
            
            # 找到之前的日志文件夹并加载历史数据
            previous_log_dir = _find_previous_log_folder(opt.resume_path)
            history = _load_previous_metrics_history(previous_log_dir, resume_epoch)
        else:
            print(f"警告: 指定的checkpoint文件不存在: {opt.resume_path}")
            print("将从头开始训练")
            opt.resume_training = False
            history = _prepare_history()
    else:
        history = _prepare_history()
    
    # 创建新的时间戳文件夹（即使是断点续训也创建新文件夹）
    base_paths = [opt.save_path, opt.sample_path, os.path.dirname(opt.metrics_log)]
    timestamped_paths = _create_timestamp_folders(base_paths)
    
    # Update opt paths to use timestamped folders
    opt.save_path = timestamped_paths['models']
    opt.sample_path = timestamped_paths['samples']
    opt.metrics_log = os.path.join(timestamped_paths['logs'], os.path.basename(opt.metrics_log))
    opt.metrics_plot = os.path.join(timestamped_paths['logs'], os.path.basename(opt.metrics_plot))

    utils.check_path(opt.save_path)
    utils.check_path(opt.sample_path)
    _ensure_dir([opt.metrics_log, opt.metrics_plot])

    generator = utils.create_generator(opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.multi_gpu and torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator)
    generator = generator.to(device)

    # 如果是断点续训，加载模型权重
    if opt.resume_training and opt.resume_path and os.path.exists(opt.resume_path):
        print(f"加载checkpoint: {opt.resume_path}")
        checkpoint = torch.load(opt.resume_path, map_location=device, weights_only=True)
        if opt.multi_gpu and isinstance(generator, nn.DataParallel):
            generator.module.load_state_dict(checkpoint)
        else:
            generator.load_state_dict(checkpoint)
        print("成功加载模型权重")

    L1Loss = nn.L1Loss()
    optimizer_g = torch.optim.Adam(
        generator.parameters(),
        lr=opt.lr_g,
        betas=(opt.b1, opt.b2),
        weight_decay=opt.weight_decay,
    )

    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_model(net, epoch, opt):
        model_name = 'BUDDY_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu and isinstance(net, nn.DataParallel):
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % epoch)
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % epoch)

    trainset, valset, testset = _build_datasets(opt)
    persistent_loader: Optional[DataLoader] = None

    val_loader = _create_dataloader(valset, max(1, opt.batch_size // 2), opt.num_workers, shuffle=False)
    test_loader = _create_dataloader(testset, max(1, opt.batch_size // 2), opt.num_workers, shuffle=False)

    print('Train samples: %d' % len(trainset))
    if valset is not None:
        print('Validation samples: %d' % len(valset))
    if testset is not None:
        print('Test samples: %d' % len(testset))

    # history变量已经在前面初始化了，不需要重复初始化
    prev_time = time.time()

    # 调整训练轮数范围：如果是断点续训，从resume_epoch开始
    start_epoch = resume_epoch if opt.resume_training else 0
    print(f"训练将从第 {start_epoch + 1} 轮开始，到第 {opt.epochs} 轮结束")

    for epoch in range(start_epoch, opt.epochs):
        generator.train()

        if opt.dataset_mode == 'bubble' and isinstance(trainset, BubbleInpaintDataset):
            categories = _curriculum_categories(trainset.available_categories, epoch + 1)
            epoch_trainset = trainset.filtered_dataset(categories, strict=True)
            train_loader = _create_dataloader(epoch_trainset, opt.batch_size, opt.num_workers, shuffle=True)
            if train_loader is None:
                raise RuntimeError(f"No training data available for categories: {categories}")
            print(f"Epoch {epoch + 1}: using categories {categories}")
        else:
            if persistent_loader is None:
                persistent_loader = _create_dataloader(trainset, opt.batch_size, opt.num_workers, shuffle=True)
                if persistent_loader is None:
                    raise RuntimeError('Training dataset is empty.')
            train_loader = persistent_loader

        train_loss_sum = 0.0
        train_psnr_sum = 0.0
        train_mse_sum = 0.0
        train_ssim_sum = 0.0
        train_count = 0

        for batch_idx, batch in enumerate(train_loader):
            blocked, mask, origin = _unpack_batch(batch, opt.dataset_mode)

            blocked = blocked.to(device)
            mask = mask.to(device)
            origin = origin.to(device)

            optimizer_g.zero_grad()
            out = generator(blocked, mask)
            out_whole = blocked * (1 - mask) + out * mask

            loss = L1Loss(out_whole, origin)
            loss.backward()
            optimizer_g.step()

            batch_size = blocked.size(0)
            train_loss_sum += loss.item() * batch_size
            train_psnr_sum += _compute_batch_psnr(out_whole.detach(), origin) * batch_size
            train_mse_sum += _compute_batch_mse(out_whole.detach(), origin) * batch_size
            train_ssim_sum += _compute_batch_ssim(out_whole.detach(), origin) * batch_size
            train_count += batch_size

            batches_done = epoch * len(train_loader) + batch_idx
            batches_left = opt.epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] time_left: %s" %
                (epoch + 1, opt.epochs, batch_idx, len(train_loader), loss.item(), time_left),
                end=''
            )

        print()

        train_loss_avg = train_loss_sum / max(train_count, 1)
        train_psnr_avg = train_psnr_sum / max(train_count, 1)
        train_mse_avg = train_mse_sum / max(train_count, 1)
        train_ssim_avg = train_ssim_sum / max(train_count, 1)

        # Evaluate validation set with category sample collection (only at specified intervals)
        val_loss, val_psnr, val_mse, val_ssim, val_sample, val_category_samples = None, None, None, None, None, None
        if (epoch + 1) % opt.eval_interval == 0:
            val_loss, val_psnr, val_mse, val_ssim, val_sample, val_category_samples = _evaluate(
                generator, val_loader, L1Loss, opt.dataset_mode, device, collect_category_samples=True
            )
        # Evaluate test set with category sample collection (only at specified intervals)
        test_loss, test_psnr, test_mse, test_ssim, test_sample, test_category_samples = None, None, None, None, None, None
        if (epoch + 1) % opt.eval_interval == 0:
            test_loss, test_psnr, test_mse, test_ssim, test_sample, test_category_samples = _evaluate(
                generator, test_loader, L1Loss, opt.dataset_mode, device, collect_category_samples=True
            )

        adjust_learning_rate(optimizer_g, epoch + 1, opt, opt.lr_g)
        save_model(generator, epoch + 1, opt)

        # Save category samples for validation set
        if val_category_samples is not None:
            _save_category_samples(val_category_samples, opt.sample_path, epoch + 1, 'val', opt.dataset_mode)
        
        # Save category samples for test set
        if test_category_samples is not None:
            _save_category_samples(test_category_samples, opt.sample_path, epoch + 1, 'test', opt.dataset_mode)

        # Keep the original single sample saving for backward compatibility (only at evaluation intervals)
        if val_sample is not None and (epoch + 1) % opt.eval_interval == 0:
            if opt.dataset_mode == 'bubble':
                utils.sample_triplet(*val_sample, opt.sample_path, epoch + 1, prefix='val')
            else:
                blocked_vis, mask_vis, recon_vis, _ = val_sample
                utils.sample(blocked_vis, mask_vis, recon_vis, opt.sample_path, epoch + 1)

        summary = (
            f"Epoch {epoch + 1}/{opt.epochs} | Train L1: {train_loss_avg:.4f}"
            f" | Train PSNR: {train_psnr_avg:.2f} | Train MSE: {train_mse_avg:.4f}"
            f" | Train SSIM: {train_ssim_avg:.3f}"
        )
        if val_loss is not None:
            summary += (
                f" | Val L1: {val_loss:.4f} | Val PSNR: {val_psnr:.2f}"
                f" | Val MSE: {val_mse:.4f} | Val SSIM: {val_ssim:.3f}"
            )
        if test_loss is not None:
            summary += (
                f" | Test L1: {test_loss:.4f} | Test PSNR: {test_psnr:.2f}"
                f" | Test MSE: {test_mse:.4f} | Test SSIM: {test_ssim:.3f}"
            )
        print(summary)

        _append_history(
            history,
            epoch + 1,
            train_loss_avg,
            train_psnr_avg,
            train_mse_avg,
            train_ssim_avg,
            val_loss,
            val_psnr,
            val_mse,
            val_ssim,
            test_loss,
            test_psnr,
            test_mse,
            test_ssim,
        )

        _write_metrics_csv(history, opt.metrics_log)
        _plot_metrics(history, opt.metrics_plot)

def Trainer_GAN(opt):
    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # cudnn benchmark accelerates the network
    cudnn.benchmark = opt.cudnn_benchmark

    # Handle multiple GPUs
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    opt.num_workers *= gpu_num
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)
    
    # Create timestamped folders for this training run
    base_paths = [opt.save_path, opt.sample_path]
    # Check if metrics_log exists in opt, if not create default path
    if hasattr(opt, 'metrics_log') and opt.metrics_log:
        base_paths.append(os.path.dirname(opt.metrics_log))
        has_metrics = True
    else:
        # Create default logs path for GAN training
        default_logs_dir = os.path.join(os.path.dirname(opt.save_path), 'logs')
        base_paths.append(default_logs_dir)
        has_metrics = False
    
    timestamped_paths = _create_timestamp_folders(base_paths)
    
    # Update opt paths to use timestamped folders
    opt.save_path = timestamped_paths['models']
    opt.sample_path = timestamped_paths['samples']
    
    if has_metrics:
        opt.metrics_log = os.path.join(timestamped_paths['logs'], os.path.basename(opt.metrics_log))
        opt.metrics_plot = os.path.join(timestamped_paths['logs'], os.path.basename(opt.metrics_plot))
    else:
        # Create default metrics paths for GAN training
        opt.metrics_log = os.path.join(timestamped_paths['logs'], 'metrics_log.csv')
        opt.metrics_plot = os.path.join(timestamped_paths['logs'], 'metrics_curve.png')

    # Build path folder
    utils.check_path(opt.save_path)
    utils.check_path(opt.sample_path)
    if has_metrics:
        _ensure_dir([opt.metrics_log, opt.metrics_plot])

    # Build networks
    generator = utils.create_generator(opt)
    discriminator = utils.create_discriminator(opt)
    perceptualnet = utils.create_perceptualnet()

    # To device
    if opt.multi_gpu == True:
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
        perceptualnet = nn.DataParallel(perceptualnet)
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()
    else:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        perceptualnet = perceptualnet.cuda()

    # Optimized num_workers for GAN training
    gpu_num = torch.cuda.device_count()
    print("There are %d GPUs used" % gpu_num)
    opt.batch_size *= gpu_num
    # Use 4 workers per GPU, but cap at 32 to avoid diminishing returns
    opt.num_workers = min(opt.num_workers * gpu_num, 32)
    print("Batch size is changed to %d" % opt.batch_size)
    print("Number of workers is changed to %d" % opt.num_workers)

    # Loss functions
    L1Loss = nn.L1Loss()
    MSELoss = nn.MSELoss()

    # Optimizers
    optimizer_g = torch.optim.Adam(generator.parameters(), lr = opt.lr_g, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr = opt.lr_d, betas = (opt.b1, opt.b2), weight_decay = opt.weight_decay)

    # Learning rate decrease
    def adjust_learning_rate(optimizer, epoch, opt, init_lr):
        """Set the learning rate to the initial LR decayed by "lr_decrease_factor" every "lr_decrease_epoch" epochs"""
        lr = init_lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    
    # Save the model if pre_train == True
    def save_model(net, epoch, opt):
        """Save the model at "checkpoint_interval" and its multiple"""
        model_name = 'GrayInpainting_GAN_epoch%d_batchsize%d.pth' % (epoch, opt.batch_size)
        model_path = os.path.join(opt.save_path, model_name)
        if opt.multi_gpu == True:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.module.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
        else:
            if epoch % opt.checkpoint_interval == 0:
                torch.save(net.state_dict(), model_path)
                print('The trained model is successfully saved at epoch %d' % (epoch))
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = True, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Training and Testing
    # ----------------------------------------

    # Initialize start time
    prev_time = time.time()

    # Tensor type
    Tensor = torch.cuda.FloatTensor
    
    # Training loop
    for epoch in range(opt.epochs):
        for batch_idx, (grayscale, mask) in enumerate(dataloader):

            # Load and put to cuda
            grayscale = grayscale.cuda()                                    # out: [B, 1, 256, 256]
            mask = mask.cuda()                                              # out: [B, 1, 256, 256]

            # LSGAN vectors
            valid = Tensor(np.ones((grayscale.shape[0], 1, 8, 8)))
            fake = Tensor(np.zeros((grayscale.shape[0], 1, 8, 8)))

            # ----------------------------------------
            #           Train Discriminator
            # ----------------------------------------
            optimizer_d.zero_grad()

            # forward propagation
            out = generator(grayscale, mask)                                # out: [B, 1, 256, 256]
            out_wholeimg = grayscale * (1 - mask) + out * mask              # in range [0, 1]

            # Fake samples
            fake_scalar = discriminator(out_wholeimg.detach(), mask)
            # True samples
            true_scalar = discriminator(grayscale, mask)
            # Overall Loss and optimize
            loss_fake = MSELoss(fake_scalar, fake)
            loss_true = MSELoss(true_scalar, valid)
            # Overall Loss and optimize
            loss_D = 0.5 * (loss_fake + loss_true)
            loss_D.backward()

            # ----------------------------------------
            #             Train Generator
            # ----------------------------------------
            optimizer_g.zero_grad()

            # forward propagation
            out = generator(grayscale, mask)                                # out: [B, 1, 256, 256]
            out_wholeimg = grayscale * (1 - mask) + out * mask              # in range [0, 1]

            # Mask L1 Loss
            MaskL1Loss = L1Loss(out_wholeimg, grayscale)

            # GAN Loss
            fake_scalar = discriminator(out_wholeimg, mask)
            MaskGAN_Loss = MSELoss(fake_scalar, valid)

            # Get the deep semantic feature maps, and compute Perceptual Loss
            out_3c = torch.cat((out_wholeimg, out_wholeimg, out_wholeimg), 1)
            grayscale_3c = torch.cat((grayscale, grayscale, grayscale), 1)
            out_featuremaps = perceptualnet(out_3c)
            gt_featuremaps = perceptualnet(grayscale_3c)
            PerceptualLoss = L1Loss(out_featuremaps, gt_featuremaps)

            # Compute losses
            loss = opt.lambda_l1 * MaskL1Loss + opt.lambda_perceptual * PerceptualLoss + opt.lambda_gan * MaskGAN_Loss
            loss.backward()
            optimizer_g.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + batch_idx
            batches_left = opt.epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds = batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            print("\r[Epoch %d/%d] [Batch %d/%d] [Mask L1 Loss: %.5f] [Perceptual Loss: %.5f] [D Loss: %.5f] [G Loss: %.5f] time_left: %s" %
                ((epoch + 1), opt.epochs, batch_idx, len(dataloader), MaskL1Loss.item(), PerceptualLoss.item(), loss_D.item(), MaskGAN_Loss.item(), time_left))

        # Learning rate decrease
        adjust_learning_rate(optimizer_g, (epoch + 1), opt, opt.lr_g)
        adjust_learning_rate(optimizer_d, (epoch + 1), opt, opt.lr_d)

        # Save the model
        save_model(generator, (epoch + 1), opt)
        utils.sample(grayscale, mask, out_wholeimg, opt.sample_path, (epoch + 1))
