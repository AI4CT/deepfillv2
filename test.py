"""批量测试脚本：遍历指定数据集目录、保存修复结果并统计重建指标。"""

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from bubble_dataset import BubbleInpaintDataset
from network import GrayInpaintingNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量推理气泡遮挡修复模型")
    parser.add_argument(
        "--test_root",
        type=str,
        default="/home/yubd/mount/dataset/dataset_overlap/test20251117",
        help="测试集根目录（需包含 Blocked/Mask/Origin 结构）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/yubd/mount/codebase/deepfillv2/test_results",
        help="修复结果输出目录",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/yubd/mount/codebase/deepfillv2/models/20251116_194359/BUDDY_GAN_epoch10_batchsize96.pth",
        help="生成器权重路径",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="测试批大小")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers 数量")
    parser.add_argument("--imgsize", type=int, default=128, help="读取并缩放到的图像尺寸")
    parser.add_argument(
        "--metrics_plot",
        type=str,
        default="",
        help="重建指标折线图保存路径（默认位于输出目录内）",
    )
    # 网络结构参数
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--out_channels", type=int, default=1)
    parser.add_argument("--mask_channels", type=int, default=1)
    parser.add_argument("--latent_channels", type=int, default=64)
    parser.add_argument("--pad", type=str, default="reflect")
    parser.add_argument("--activ_g", type=str, default="lrelu")
    parser.add_argument("--norm_g", type=str, default="in")
    return parser.parse_args()


def load_model(opt: argparse.Namespace, device: torch.device) -> GrayInpaintingNet:
    model = GrayInpaintingNet(opt).to(device)
    checkpoint = torch.load(opt.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def _to_uint8_img(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().cpu().numpy()
    arr = np.squeeze(arr)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def save_prediction(
    raw_blocked: np.ndarray,
    pred: torch.Tensor,
    origin_img: torch.Tensor,
    src_path: str,
    output_dir: Path,
) -> None:
    blocked_raw_arr = raw_blocked
    pred_arr = _to_uint8_img(pred)
    origin_arr = _to_uint8_img(origin_img)

    merged = np.concatenate((blocked_raw_arr, pred_arr, origin_arr), axis=1)
    basename = os.path.basename(src_path)
    save_name = basename.replace("_blocked", "_repaired")
    save_path = output_dir / save_name
    cv2.imwrite(str(save_path), merged)


def _load_blocked_image(path: str, imgsize: int) -> np.ndarray:
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"无法读取 Blocked 图像: {path}")
    if arr.shape[0] != imgsize or arr.shape[1] != imgsize:
        arr = cv2.resize(arr, (imgsize, imgsize), interpolation=cv2.INTER_AREA)
    return arr


def _extract_overlap_level(path: str) -> float:
    """从文件名中提取具体的遮挡比例

    文件名格式：{原始气泡名}_{遮挡比例}_blocked.png
    例如：10_frame77_5.7_blocked.png -> 5.7

    Returns:
        float: 遮挡比例（百分比），如果提取失败返回 -1.0
    """
    filename = Path(path).stem  # 获取不含扩展名的文件名

    # 去掉 _blocked 后缀
    if filename.endswith("_blocked"):
        filename = filename[:-8]  # 去掉 "_blocked"

    # 找到最后一个下划线后的内容，应该是遮挡比例
    parts = filename.rsplit("_", 1)
    if len(parts) == 2:
        try:
            overlap_percentage = float(parts[1])
            return overlap_percentage
        except ValueError:
            pass

    return -1.0


def _extract_bubble_id(path: str) -> str:
    """从文件名中提取气泡ID

    文件名格式：{气泡ID}_{遮挡比例}_blocked.png
    例如：1001_frame1816_5.7_blocked.png -> 1001_frame1816

    Returns:
        str: 气泡ID，如果提取失败返回空字符串
    """
    filename = Path(path).stem  # 获取不含扩展名的文件名

    # 去掉 _blocked 后缀
    if filename.endswith("_blocked"):
        filename = filename[:-8]  # 去掉 "_blocked"

    # 去掉最后一个下划线及其后的遮挡比例
    parts = filename.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]  # 返回气泡ID

    return filename  # 如果无法分割，返回整个文件名


def _load_bubble_labels(csv_path: str) -> Dict[str, int]:
    """从CSV文件加载气泡类别标签

    Args:
        csv_path: CSV文件路径，格式为：file_id,cluster

    Returns:
        Dict[str, int]: 气泡ID到类别的映射字典
    """
    bubble_labels = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row['file_id'].strip()
                cluster = int(row['cluster'])
                bubble_labels[file_id] = cluster
        print(f"成功加载 {len(bubble_labels)} 个气泡的类别标签")
    except Exception as e:
        print(f"警告：无法加载气泡类别标签文件 {csv_path}: {e}")
        print("将继续运行但不按类别分组")

    return bubble_labels


def _compute_sample_metrics(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float]:
    diff = pred - target
    l1 = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    eps = 1e-8
    psnr = 20.0 * math.log10(1.0 / math.sqrt(mse + eps))
    pred_np = pred.squeeze().numpy()
    target_np = target.squeeze().numpy()
    ssim = structural_similarity(target_np, pred_np, data_range=1.0)
    return l1, mse, ssim, psnr


def _plot_metric_curves_with_polyfit(sample_metrics: List[Dict[str, Union[float, int, str]]], save_path: Path) -> None:
    """使用散点图+多项式拟合曲线绘制指标曲线（按气泡类别分组）

    Args:
        sample_metrics: 包含每个样本的遮挡比例、类别和指标的列表
        save_path: 图表保存路径
    """
    if not sample_metrics:
        print("No metrics to plot.")
        return

    from collections import defaultdict

    # 按类别分组数据
    category_data = defaultdict(list)
    for sample in sample_metrics:
        category = sample.get("category", -1)
        # 只保留类别有效的样本（类别为 0, 1, 2, 3）
        if category >= 0:
            category_data[category].append(sample)

    # 获取所有类别并排序
    categories = sorted(category_data.keys())
    num_categories = len(categories)

    if num_categories == 0:
        print("No valid categories found in metrics. Falling back to all-data plotting.")
        # 如果没有类别信息，绘制所有数据的单一曲线
        _plot_all_data_fallback(sample_metrics, save_path)
        return

    print(f"按 {num_categories} 个类别分组绘图：{categories}")

    # 为每个类别分配颜色
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    category_colors = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metric_names = ["l1", "mse", "ssim", "psnr"]
    metric_titles = ["L1 Loss", "Mean Squared Error", "SSIM", "PSNR (dB)"]
    metric_ylabels = ["L1", "MSE", "SSIM", "PSNR"]

    # 多项式拟合阶数
    poly_degree = 3

    for ax_idx, (ax, metric_name, title, ylabel) in enumerate(zip(axes.flat, metric_names, metric_titles, metric_ylabels)):
        # 为每个类别绘制散点图和拟合曲线
        for cat_idx, category in enumerate(categories):
            cat_samples = category_data[category]
            if not cat_samples:
                continue

            # 提取当前类别的数据并排序
            data_sorted = sorted(cat_samples, key=lambda x: x["overlap_percentage"])
            overlap_percentages = np.array([d["overlap_percentage"] for d in data_sorted])
            metric_values = np.array([d[metric_name] for d in data_sorted])

            color = category_colors[category]

            # 绘制散点图
            ax.scatter(
                overlap_percentages,
                metric_values,
                alpha=0.5,
                s=25,
                color=color,
                label=f"Category {category}",
                edgecolors="black",
                linewidths=0.5
            )

            # 对当前类别的数据进行分组平均（用于拟合）
            grouped = defaultdict(list)
            for d in data_sorted:
                grouped[d["overlap_percentage"]].append(d[metric_name])

            unique_overlaps = sorted(grouped.keys())
            avg_values = np.array([np.mean(grouped[ovl]) for ovl in unique_overlaps])
            unique_overlaps_arr = np.array(unique_overlaps)

            # 多项式拟合（如果数据点足够）
            if len(unique_overlaps_arr) >= 2:
                try:
                    # 自动选择合适的阶数
                    degree = min(poly_degree, len(unique_overlaps_arr) - 1)
                    coeffs = np.polyfit(unique_overlaps_arr, avg_values, degree)
                    poly = np.poly1d(coeffs)

                    # 生成平滑的x值用于绘制曲线
                    x_smooth = np.linspace(unique_overlaps_arr.min(), unique_overlaps_arr.max(), 300)
                    y_smooth = poly(x_smooth)

                    # 绘制拟合曲线
                    ax.plot(x_smooth, y_smooth, color=color, linewidth=2.5, alpha=0.8, linestyle="-")
                except Exception as e:
                    print(f"Warning: Failed to fit polynomial for {ylabel}, category {category}: {e}")

        ax.set_xlabel("Overlap Percentage (%)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=10, framealpha=0.9)

    fig.suptitle("Reconstruction Metrics vs. Overlap Percentage (By Bubble Category)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"图表已保存，包含 {num_categories} 个类别")


def _plot_all_data_fallback(sample_metrics: List[Dict[str, Union[float, int, str]]], save_path: Path) -> None:
    """回退方案：不分类别，绘制所有数据的曲线

    Args:
        sample_metrics: 包含每个样本的遮挡比例和指标的列表
        save_path: 图表保存路径
    """
    from collections import defaultdict

    # 转换为numpy数组并按遮挡比例排序
    data = sorted(sample_metrics, key=lambda x: x["overlap_percentage"])
    overlap_percentages = np.array([d["overlap_percentage"] for d in data])
    l1_values = np.array([d["l1"] for d in data])
    mse_values = np.array([d["mse"] for d in data])
    ssim_values = np.array([d["ssim"] for d in data])
    psnr_values = np.array([d["psnr"] for d in data])

    # 对相同遮挡比例的样本进行分组平均（用于拟合）
    grouped_data = defaultdict(lambda: {"l1": [], "mse": [], "ssim": [], "psnr": []})
    for d in data:
        ovl = d["overlap_percentage"]
        grouped_data[ovl]["l1"].append(d["l1"])
        grouped_data[ovl]["mse"].append(d["mse"])
        grouped_data[ovl]["ssim"].append(d["ssim"])
        grouped_data[ovl]["psnr"].append(d["psnr"])

    # 计算每个遮挡比例的平均值
    unique_overlaps = sorted(grouped_data.keys())
    avg_l1 = np.array([np.mean(grouped_data[ovl]["l1"]) for ovl in unique_overlaps])
    avg_mse = np.array([np.mean(grouped_data[ovl]["mse"]) for ovl in unique_overlaps])
    avg_ssim = np.array([np.mean(grouped_data[ovl]["ssim"]) for ovl in unique_overlaps])
    avg_psnr = np.array([np.mean(grouped_data[ovl]["psnr"]) for ovl in unique_overlaps])
    unique_overlaps_arr = np.array(unique_overlaps)

    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_specs = [
        (l1_values, avg_l1, "L1 Loss", "L1"),
        (mse_values, avg_mse, "Mean Squared Error", "MSE"),
        (ssim_values, avg_ssim, "SSIM", "SSIM"),
        (psnr_values, avg_psnr, "PSNR (dB)", "PSNR"),
    ]

    # 多项式拟合阶数
    poly_degree = 3

    for ax, (values, avg_values, title, ylabel) in zip(axes.flat, metric_specs):
        # 绘制散点图（所有样本）
        ax.scatter(
            overlap_percentages, values, alpha=0.6, s=30, color="steelblue", label="Samples", edgecolors="navy"
        )

        # 多项式拟合（使用平均值）
        if len(unique_overlaps_arr) >= 2:
            try:
                # 使用多项式拟合，自动选择合适的阶数
                degree = min(poly_degree, len(unique_overlaps_arr) - 1)
                coeffs = np.polyfit(unique_overlaps_arr, avg_values, degree)
                poly = np.poly1d(coeffs)

                # 生成平滑的x值用于绘制曲线
                x_smooth = np.linspace(unique_overlaps_arr.min(), unique_overlaps_arr.max(), 300)
                y_smooth = poly(x_smooth)

                # 绘制拟合曲线
                ax.plot(x_smooth, y_smooth, color="red", linewidth=2, label=f"Polynomial Fit (degree {degree})")
            except Exception as e:
                print(f"Warning: Failed to fit polynomial for {ylabel}: {e}")

        ax.set_xlabel("Overlap Percentage (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle("Reconstruction Metrics vs. Overlap Percentage", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def main() -> None:
    opt = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(opt.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = Path(opt.metrics_plot) if opt.metrics_plot else output_dir / "reconstruction_metrics.png"

    # 加载气泡类别标签
    bubble_labels_path = "/home/yubd/mount/codebase/deepfillv2/bubble_labels_k4_simple.csv"
    bubble_labels = _load_bubble_labels(bubble_labels_path)

    dataset = BubbleInpaintDataset(opt.test_root, opt, phase="test", return_paths=True, strict=False)
    if len(dataset) == 0:
        raise RuntimeError(f"未在 {opt.test_root} 下发现可用的测试样本。")

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    model = load_model(opt, device)
    print(f"共加载 {len(dataset)} 个测试样本，开始推理……")

    # 修改数据结构：存储每个样本的独立数据点（包含类别信息）
    sample_metrics: List[Dict[str, Union[float, int, str]]] = []

    with torch.no_grad():
        for idx, (blocked, mask, origin, paths) in enumerate(dataloader):
            blocked_device = blocked.to(device)
            mask_device = mask.to(device)
            fake_target = model(blocked_device, mask_device)
            out_whole = blocked_device * (1 - mask_device) + fake_target * mask_device

            for b in range(out_whole.size(0)):
                raw_blocked = _load_blocked_image(paths[b], opt.imgsize)
                save_prediction(
                    raw_blocked,
                    out_whole[b],
                    origin[b],
                    paths[b],
                    output_dir,
                )

                pred_cpu = out_whole[b].detach().cpu()
                target_cpu = origin[b].detach().cpu()
                l1, mse, ssim, psnr = _compute_sample_metrics(pred_cpu, target_cpu)

                # 提取具体的遮挡比例（而不是区间）
                overlap_percentage = _extract_overlap_level(paths[b])

                # 提取气泡ID并查找其类别
                bubble_id = _extract_bubble_id(paths[b])
                bubble_category = bubble_labels.get(bubble_id, -1)  # 如果找不到，使用 -1 表示未知类别

                # 只有成功提取遮挡比例的样本才记录
                if overlap_percentage >= 0:
                    sample_metrics.append({
                        "overlap_percentage": overlap_percentage,
                        "bubble_id": bubble_id,
                        "category": bubble_category,
                        "l1": l1,
                        "mse": mse,
                        "ssim": ssim,
                        "psnr": psnr,
                    })

            if (idx + 1) % 50 == 0 or (idx + 1) == len(dataloader):
                print(f"已处理 {min((idx + 1) * opt.batch_size, len(dataset))} / {len(dataset)} 张")

    print(f"推理完成，结果已保存至: {output_dir}")

    if sample_metrics:
        print(f"\n成功收集 {len(sample_metrics)} 个样本的重建指标")
        # 统计每个类别的样本数量
        from collections import Counter
        category_counts = Counter(d["category"] for d in sample_metrics)
        print(f"各类别样本数量：{dict(category_counts)}")

        _plot_metric_curves_with_polyfit(sample_metrics, plot_path)
        print(f"Saved reconstruction metrics plot to: {plot_path}")
    else:
        print("Warning: no metrics were collected; skipping plot generation.")


if __name__ == "__main__":
    main()
