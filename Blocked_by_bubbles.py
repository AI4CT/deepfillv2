import cv2
import os
import random
import numpy as np
import glob
import csv
import time
import json
import signal
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Rich库用于美化控制台输出
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, 
    TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout

# 性能监控
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ==================== 全局变量和配置 ====================
console = Console()
shutdown_flag = False  # 优雅退出标志
processing_lock = Lock()  # 共享资源锁
failed_tasks = []  # 失败任务列表
failed_tasks_lock = Lock()  # 失败任务列表锁

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bubble_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 检查点管理系统 ====================
class CheckpointManager:
    """管理处理进度的检查点，支持断点续传"""
    
    def __init__(self, checkpoint_file: str = 'processing_checkpoint.json'):
        self.checkpoint_file = checkpoint_file
        self.data = self.load()
    
    def load(self) -> Dict:
        """加载检查点数据"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"无法加载检查点文件: {e}")
        return {
            'completed_bubbles': [],
            'completed_levels': {},
            'bubble_samples_per_level': {},  # 新增：跟踪每个气泡在每个级别的样本数
            'metadata_list': [],
            'total_attempts': 0,
            'successful_bubbles': 0,
            'abandoned_bubbles': 0,
            'start_time': datetime.now().isoformat()
        }
    
    def save(self):
        """保存检查点数据"""
        try:
            with processing_lock:
                with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def add_completed_bubble(self, bubble_name: str):
        """添加已完成的气泡"""
        with processing_lock:
            if bubble_name not in self.data['completed_bubbles']:
                self.data['completed_bubbles'].append(bubble_name)
    
    def add_completed_level(self, bubble_name: str, category: str):
        """为指定气泡添加已完成的遮挡级别"""
        with processing_lock:
            if bubble_name not in self.data['completed_levels']:
                self.data['completed_levels'][bubble_name] = []
            if category not in self.data['completed_levels'][bubble_name]:
                self.data['completed_levels'][bubble_name].append(category)
    
    def get_completed_levels(self, bubble_name: str) -> Set[str]:
        """获取指定气泡已完成的遮挡级别"""
        return set(self.data['completed_levels'].get(bubble_name, []))
    
    def is_bubble_completed(self, bubble_name: str) -> bool:
        """检查气泡是否已完成所有级别"""
        return bubble_name in self.data['completed_bubbles']
    
    def add_metadata(self, metadata: Dict):
        """添加元数据"""
        with processing_lock:
            self.data['metadata_list'].append(metadata)
    
    def increment_attempts(self):
        """增加尝试次数"""
        with processing_lock:
            self.data['total_attempts'] += 1
    
    def increment_successful(self):
        """增加成功数"""
        with processing_lock:
            self.data['successful_bubbles'] += 1
    
    def increment_abandoned(self):
        """增加放弃数"""
        with processing_lock:
            self.data['abandoned_bubbles'] += 1
    
    def has_level_sample(self, bubble_name: str, category: str) -> bool:
        """检查某个气泡在某个级别是否已有样本"""
        with processing_lock:
            if bubble_name not in self.data['bubble_samples_per_level']:
                return False
            return category in self.data['bubble_samples_per_level'].get(bubble_name, {})
    
    def add_level_sample(self, bubble_name: str, category: str):
        """记录气泡在某个级别的样本"""
        with processing_lock:
            if bubble_name not in self.data['bubble_samples_per_level']:
                self.data['bubble_samples_per_level'][bubble_name] = {}
            if category not in self.data['bubble_samples_per_level'][bubble_name]:
                self.data['bubble_samples_per_level'][bubble_name][category] = 0
            self.data['bubble_samples_per_level'][bubble_name][category] += 1
    
    def get_bubble_sample_count(self, bubble_name: str) -> int:
        """获取某个气泡的总样本数"""
        return sum(self.data['bubble_samples_per_level'].get(bubble_name, {}).values())
    
    def get_bubble_level_distribution(self, bubble_name: str) -> Dict[str, int]:
        """获取某个气泡的级别分布"""
        return self.data['bubble_samples_per_level'].get(bubble_name, {})

# ==================== 信号处理 ====================
def signal_handler(signum, frame):
    """处理终止信号，实现优雅退出"""
    global shutdown_flag
    signal_name = signal.Signals(signum).name
    console.print(f"\n[yellow]收到信号 {signal_name}，正在优雅退出...[/yellow]")
    shutdown_flag = True

def setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# ==================== 性能监控 ====================
class PerformanceMonitor:
    """监控系统性能"""
    
    def __init__(self):
        self.start_time = time.time()
        self.processed_count = 0
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
    
    def get_stats(self) -> Dict:
        """获取性能统计"""
        elapsed_time = time.time() - self.start_time
        speed = self.processed_count / elapsed_time if elapsed_time > 0 else 0
        
        stats = {
            'elapsed_time': elapsed_time,
            'processed_count': self.processed_count,
            'speed': speed
        }
        
        if self.process and PSUTIL_AVAILABLE:
            try:
                stats['memory_mb'] = self.process.memory_info().rss / 1024 / 1024
                stats['cpu_percent'] = self.process.cpu_percent(interval=0.1)
            except Exception:
                pass
        
        return stats
    
    def increment_processed(self):
        """增加处理计数"""
        self.processed_count += 1

# ==================== 核心处理函数 ====================
def generate_occlusion_intervals(num_intervals: int) -> List[str]:
    """生成遮挡程度区间列表"""
    if num_intervals <= 0:
        return []
    
    # 计算每个区间的大小
    interval_size = 100 // num_intervals
    intervals = []
    
    for i in range(num_intervals):
        start = i * interval_size
        end = (i + 1) * interval_size if i < num_intervals - 1 else 100
        
        intervals.append(f"{start}-{end}")
    
    return intervals

def create_directory_structure(base_path, num_intervals=10):
    """创建分层目录结构（按遮挡比例命名）"""
    categories = generate_occlusion_intervals(num_intervals)
    subdirs = ['Origin', 'Blocked', 'Segmented', 'Mask']
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for category in categories:
        category_path = os.path.join(base_path, category)
        for subdir in subdirs:
            full_path = os.path.join(category_path, subdir)
            if not os.path.exists(full_path):
                os.makedirs(full_path)

def get_bubble_contour(img):
    """获取气泡的外轮廓"""
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 178, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_lengths = [cv2.arcLength(contour, True) for contour in contours]
    contour_lengths_sorted = sorted(contour_lengths, reverse=True)
    second_longest_contour_index = contour_lengths.index(contour_lengths_sorted[1])
    
    return contours[second_longest_contour_index]

def calculate_contour_area(contour):
    """计算轮廓面积"""
    return cv2.contourArea(contour)

def get_category_from_occlusion(occlusion_ratio, num_intervals=10):
    """根据遮挡率确定分类目录（遮挡率 = 1 - 可见率）"""
    occlusion_percentage = occlusion_ratio * 100
    categories = generate_occlusion_intervals(num_intervals)
    
    if not categories:
        return '0-10'  # 默认返回
    
    interval_size = 100 // num_intervals
    
    for i, category in enumerate(categories):
        start, end = map(int, category.split('-'))
        
        # 对于最后一个区间，包含100%
        if i == num_intervals - 1:
            if occlusion_percentage >= start and occlusion_percentage <= end:
                return category
        else:
            if occlusion_percentage >= start and occlusion_percentage < end:
                return category
    
    # 如果没有匹配到任何区间，返回最后一个区间
    return categories[-1]

def determine_occlusion_params(target_occlusion_range, num_intervals=10):
    """根据目标遮挡范围确定遮挡参数"""
    categories = generate_occlusion_intervals(num_intervals)
    
    # 计算区间索引
    try:
        interval_index = categories.index(target_occlusion_range)
    except ValueError:
        # 如果找不到目标区间，使用默认参数
        return ([1, 2, 3, 4], (0.7, 1.2), (0.3, 1.0))
    
    # 根据区间索引动态分配参数
    total_intervals = num_intervals
    
    if interval_index < total_intervals // 3:  # 低遮挡区间
        return ([1], (0.3, 0.6), (0.6, 1.2))
    elif interval_index < 2 * total_intervals // 3:  # 中等遮挡区间
        return ([1, 2], (0.5, 0.8), (0.5, 1.0))
    elif interval_index < 5 * total_intervals // 6:  # 较高遮挡区间
        return ([2, 3], (0.7, 1.0), (0.4, 0.9))
    else:  # 高遮挡区间
        return ([3, 4, 5], (0.9, 1.3), (0.2, 0.7))

def process_bubble_with_target(bubble1_path, bubble_images, output_base_path,
                               target_category=None, attempt_id=0, num_intervals=10, save_files=False):
    """
    处理单个气泡图像，生成四种输出
    新增功能：记录使用的遮挡气泡列表
    新增参数：save_files 控制是否保存文件到磁盘
    
    Returns:
        dict: 包含图像数据和元数据，如果save_files=True则保存文件并返回路径，否则返回图像数据
    """
    try:
        bubble1 = cv2.imread(bubble1_path)
        if bubble1 is None:
            return None
        
        original_contour = get_bubble_contour(bubble1)
        
        original_mask = np.zeros(bubble1.shape[:2], np.uint8)
        cv2.drawContours(original_mask, [original_contour], 0, 255, -1)
        
        # 统一计算方式：使用掩码计算原始面积，而不是轮廓面积
        original_area = np.sum(original_mask > 0)
        
        if target_category:
            blocked_num_range, scale_range, shift_range = determine_occlusion_params(target_category, num_intervals)
        else:
            blocked_num_range = [1, 2, 3, 4]
            scale_range = (0.7, 1.2)
            shift_range = (0.3, 1.0)
        
        blocked_image = bubble1.copy()
        occlusion_mask = np.zeros(bubble1.shape[:2], np.uint8)
        
        blocked_num = random.choice(blocked_num_range)
        
        # 新增：记录使用的遮挡气泡
        occluding_bubbles = []
        
        for i in range(blocked_num):
            bubble2_path = random.choice(bubble_images)
            bubble2 = cv2.imread(bubble2_path)
            
            # 记录遮挡气泡
            occluding_bubble_name = os.path.splitext(os.path.basename(bubble2_path))[0]
            occluding_bubbles.append(occluding_bubble_name)
            if bubble2 is None:
                continue
            
            gray2 = cv2.cvtColor(bubble2, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray2, 180, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) < 2:
                continue
                
            contour_lengths = [cv2.arcLength(contour, True) for contour in contours]
            contour_lengths_sorted = sorted(contour_lengths, reverse=True)
            second_longest_contour_index = contour_lengths.index(contour_lengths_sorted[1])
            x, y, w, h = cv2.boundingRect(contours[second_longest_contour_index])
            
            bubble2_mask = np.zeros(bubble2.shape[:2], np.uint8)
            cv2.drawContours(bubble2_mask, contours, second_longest_contour_index, 255, -1)
            bubble2_extracted = cv2.bitwise_and(bubble2, bubble2, mask=bubble2_mask)
            
            kernel = np.ones((3, 3), np.uint8)
            bubble2_extracted = cv2.erode(bubble2_extracted, kernel, iterations=1)
            bubble2_extracted = cv2.dilate(bubble2_extracted, kernel, 1)
            
            rows, cols, _ = bubble1.shape
            shift_min, shift_max = shift_range
            shift_x1, shift_y1 = random.uniform(-shift_max, -shift_min) * w, random.uniform(-shift_max, -shift_min) * h
            shift_x2, shift_y2 = random.uniform(shift_min, shift_max) * w, random.uniform(shift_min, shift_max) * h
            shift_x, shift_y = random.choice([(shift_x1, shift_y1), (shift_x2, shift_y2)])
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            bubble2_transformed = cv2.warpAffine(bubble2_extracted, M, (cols, rows))
            bubble2_mask_transformed = cv2.warpAffine(bubble2_mask, M, (cols, rows))
            
            rotate_angle = random.randint(-90, 90)
            scale_min, scale_max = scale_range
            rotate_scale = random.uniform(scale_min, scale_max)
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotate_angle, rotate_scale)
            bubble2_transformed = cv2.warpAffine(bubble2_transformed, M, (cols, rows))
            bubble2_mask_transformed = cv2.warpAffine(bubble2_mask_transformed, M, (cols, rows))
            
            scale_x, scale_y = random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)
            M = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
            bubble2_transformed = cv2.warpAffine(bubble2_transformed, M, (cols, rows))
            bubble2_mask_transformed = cv2.warpAffine(bubble2_mask_transformed, M, (cols, rows))
            
            if i == 0:
                blocked_image = cv2.bitwise_and(bubble1, cv2.bitwise_not(cv2.cvtColor(bubble2_mask_transformed, cv2.COLOR_GRAY2BGR)))
                blocked_image = cv2.add(blocked_image, bubble2_transformed)
                occlusion_mask = bubble2_mask_transformed.copy()
            else:
                blocked_image = cv2.bitwise_and(blocked_image, cv2.bitwise_not(cv2.cvtColor(bubble2_mask_transformed, cv2.COLOR_GRAY2BGR)))
                blocked_image = cv2.add(blocked_image, bubble2_transformed)
                occlusion_mask = cv2.bitwise_or(occlusion_mask, bubble2_mask_transformed)
        
        visible_mask = cv2.bitwise_and(original_mask, cv2.bitwise_not(occlusion_mask))
        visible_area = np.sum(visible_mask > 0)
        visibility_ratio = visible_area / original_area if original_area > 0 else 0
        occlusion_ratio = 1.0 - visibility_ratio
        
        # 检查occlusion_ratio是否为负数，如果是则跳过该数据
        if occlusion_ratio < 0:
            # logger.warning(f"跳过负数遮挡率数据 {bubble1_path}: occlusion_ratio={occlusion_ratio}")
            return None
        
        segmented_image = np.ones_like(bubble1) * 255
        visible_part = cv2.bitwise_and(bubble1, bubble1, mask=visible_mask)
        segmented_image[visible_mask > 0] = visible_part[visible_mask > 0]
        
        binary_mask = (visible_mask > 0).astype(np.uint8) * 255
        
        category = get_category_from_occlusion(occlusion_ratio, num_intervals)
        
        # 新的文件命名策略：{原始气泡名}_{具体遮挡比例}_{图像类型}.png
        filename = os.path.splitext(os.path.basename(bubble1_path))[0]
        occlusion_percentage = occlusion_ratio * 100
        unique_filename = f'{filename}_{occlusion_percentage:.1f}'
        
        # 生成文件路径
        origin_path = os.path.join(output_base_path, category, 'Origin', f'{unique_filename}_original.png')
        blocked_path = os.path.join(output_base_path, category, 'Blocked', f'{unique_filename}_blocked.png')
        segmented_path = os.path.join(output_base_path, category, 'Segmented', f'{unique_filename}_segmented.png')
        mask_path = os.path.join(output_base_path, category, 'Mask', f'{unique_filename}_mask.png')
        
        # 根据save_files参数决定是否保存文件
        if save_files:
            cv2.imwrite(origin_path, bubble1, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(blocked_path, blocked_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(segmented_path, segmented_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            cv2.imwrite(mask_path, binary_mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        
        return {
            'filename': unique_filename,
            'original_bubble': filename,
            'bubble_filename': filename,  # 新增：原始气泡文件名
            'occluding_bubbles': ','.join(occluding_bubbles),  # 新增：遮挡气泡列表
            'attempt_id': attempt_id,
            'origin_path': origin_path,
            'blocked_path': blocked_path,
            'segmented_path': segmented_path,
            'mask_path': mask_path,
            'occlusion_ratio': occlusion_ratio,
            'visibility_ratio': visibility_ratio,
            'category': category,
            'images': {  # 新增：图像数据，用于后续保存
                'bubble1': bubble1,
                'blocked_image': blocked_image,
                'segmented_image': segmented_image,
                'binary_mask': binary_mask
            }
        }
    except Exception as e:
        logger.error(f"处理气泡失败 {bubble1_path}: {e}")
        with failed_tasks_lock:
            failed_tasks.append({'path': bubble1_path, 'error': str(e)})
        return None

def process_single_bubble(img_path: str, bubble_images: List[str],
                         output_base_path: str, checkpoint: CheckpointManager,
                         occlusion_categories: List[str], max_attempts: int = 200,
                         max_samples_per_bubble: int = 10, num_intervals: int = 10) -> Dict:
    """
    处理单个气泡的所有遮挡级别
    新的采样策略：每个遮挡级别最多保存一张样本
    
    Args:
        img_path: 气泡图像路径
        bubble_images: 所有气泡图像列表
        output_base_path: 输出基础路径
        checkpoint: 检查点管理器
        occlusion_categories: 遮挡级别列表
        max_attempts: 最大尝试次数
        max_samples_per_bubble: 每个气泡最多样本数（默认10，即每级别1张）
    
    Returns:
        处理结果统计字典
    """
    global shutdown_flag
    
    bubble_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 从检查点获取已完成的遮挡级别
    completed_levels = checkpoint.get_completed_levels(bubble_name)
    attempt_count = 0
    newly_completed = []
    samples_saved = 0
    
    # 目标：为每个遮挡级别生成一张样本，最多max_samples_per_bubble张
    target_levels = min(len(occlusion_categories), max_samples_per_bubble)
    
    while len(completed_levels) < target_levels and attempt_count < max_attempts:
        # 检查退出标志
        if shutdown_flag:
            break
        
        try:
            metadata = process_bubble_with_target(
                img_path,
                bubble_images,
                output_base_path,
                target_category=None,
                attempt_id=attempt_count,
                num_intervals=num_intervals,
                save_files=False  # 先不保存文件，等检查通过后再保存
            )
            
            if metadata:
                category = metadata['category']
                
                # 新策略：检查该级别是否已有样本
                if not checkpoint.has_level_sample(bubble_name, category):
                    # 该级别还没有样本，保存此样本的图像文件
                    images = metadata['images']
                    cv2.imwrite(metadata['origin_path'], images['bubble1'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    cv2.imwrite(metadata['blocked_path'], images['blocked_image'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    cv2.imwrite(metadata['segmented_path'], images['segmented_image'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    cv2.imwrite(metadata['mask_path'], images['binary_mask'], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    
                    # 从metadata中移除图像数据（节省内存）
                    metadata.pop('images', None)
                    
                    completed_levels.add(category)
                    newly_completed.append(category)
                    checkpoint.add_completed_level(bubble_name, category)
                    checkpoint.add_level_sample(bubble_name, category)
                    checkpoint.add_metadata(metadata)
                    samples_saved += 1
                    
                    # 记录采样决策到日志
                    logger.info(f"气泡 {bubble_name}: 在级别 {category} 保存样本 (尝试 {attempt_count})")
                else:
                    # 该级别已有样本，跳过
                    logger.debug(f"气泡 {bubble_name}: 级别 {category} 已有样本，跳过 (尝试 {attempt_count})")
            
            attempt_count += 1
            checkpoint.increment_attempts()
            
        except Exception as e:
            logger.error(f"处理气泡 {bubble_name} 的尝试 {attempt_count} 失败: {e}")
            attempt_count += 1
            continue
    
    # 更新统计
    if len(completed_levels) >= target_levels:
        checkpoint.increment_successful()
        checkpoint.add_completed_bubble(bubble_name)
        status = 'success'
    else:
        checkpoint.increment_abandoned()
        status = 'partial'
    
    return {
        'bubble_name': bubble_name,
        'completed_levels': completed_levels,
        'attempts': attempt_count,
        'newly_completed': newly_completed,
        'samples_saved': samples_saved,
        'status': status
    }

def generate_statistics_and_visualization(metadata_list, output_base_path, checkpoint: CheckpointManager = None, num_intervals: int = 10):
    """
    生成数据集统计和可视化
    新增：按气泡统计的分布图表
    """
    if not metadata_list:
        console.print("[yellow]没有数据可用于统计[/yellow]")
        return

    df = pd.DataFrame(metadata_list)
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 生成动态类别列表
    categories = generate_occlusion_intervals(num_intervals)
    
    # 创建更大的图表（增加一行用于气泡统计）
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Dataset Statistical Analysis', fontsize=16, fontweight='bold')
    
    # 1. 样本数分布
    category_counts = df['category'].value_counts().sort_index()
    counts = [category_counts.get(cat, 0) for cat in categories]
    
    axes[0, 0].bar(categories, counts, color='skyblue', edgecolor='navy')
    axes[0, 0].set_xlabel('Occlusion Ratio Interval (%)', fontsize=12)
    axes[0, 0].set_ylabel('Number of Samples', fontsize=12)
    axes[0, 0].set_title('Sample Distribution by Occlusion Interval', fontsize=13, fontweight='bold')
    axes[0, 0].grid(axis='y', alpha=0.3)
    for i, count in enumerate(counts):
        axes[0, 0].text(i, count, str(count), ha='center', va='bottom')
    
    # 2. 遮挡率直方图
    axes[0, 1].hist(df['occlusion_ratio'] * 100, bins=30, color='lightcoral', edgecolor='darkred', alpha=0.7)
    axes[0, 1].set_xlabel('Occlusion Ratio (%)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Histogram of Occlusion Ratios', fontsize=13, fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].axvline(df['occlusion_ratio'].mean() * 100, color='red', 
                       linestyle='--', linewidth=2, label=f'Mean: {df["occlusion_ratio"].mean()*100:.2f}%')
    axes[0, 1].legend()
    
    # 3. 箱型图
    box_data = [df[df['category'] == cat]['occlusion_ratio'].values * 100 
                for cat in categories if cat in df['category'].values]
    box_labels = [cat for cat in categories if cat in df['category'].values]
    
    bp = axes[1, 0].boxplot(box_data, labels=box_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_edgecolor('darkgreen')
    axes[1, 0].set_xlabel('Occlusion Ratio Interval (%)', fontsize=12)
    axes[1, 0].set_ylabel('Occlusion Ratio (%)', fontsize=12)
    axes[1, 0].set_title('Boxplot of Occlusion Ratios by Interval', fontsize=13, fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 4. 统计摘要
    axes[1, 1].axis('off')
    stats_text = f"""
    Dataset Statistical Summary
    {'='*40}
    Total Samples: {len(df)}
    
    Occlusion Ratio Statistics:
    - Mean: {df['occlusion_ratio'].mean()*100:.2f}%
    - Median: {df['occlusion_ratio'].median()*100:.2f}%
    - Std: {df['occlusion_ratio'].std()*100:.2f}%
    - Min: {df['occlusion_ratio'].min()*100:.2f}%
    - Max: {df['occlusion_ratio'].max()*100:.2f}%
    
    Samples per Occlusion Interval:
    """
    for cat in categories:
        count = category_counts.get(cat, 0)
        percentage = (count / len(df) * 100) if len(df) > 0 else 0
        stats_text += f"  {cat}%: {count} ({percentage:.1f}%)\n"
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5. 按气泡统计的样本数分布（新增）
    if 'bubble_filename' in df.columns:
        bubble_counts = df['bubble_filename'].value_counts().head(20)  # 显示前20个
        axes[2, 0].barh(range(len(bubble_counts)), bubble_counts.values, color='mediumpurple')
        axes[2, 0].set_yticks(range(len(bubble_counts)))
        axes[2, 0].set_yticklabels(bubble_counts.index, fontsize=8)
        axes[2, 0].set_xlabel('Number of Samples', fontsize=12)
        axes[2, 0].set_ylabel('Bubble Name', fontsize=12)
        axes[2, 0].set_title('Top 20 Bubbles by Sample Count', fontsize=13, fontweight='bold')
        axes[2, 0].grid(axis='x', alpha=0.3)
        axes[2, 0].invert_yaxis()
        
        # 添加数值标签
        for i, v in enumerate(bubble_counts.values):
            axes[2, 0].text(v, i, f' {v}', va='center', fontsize=8)
    
    # 6. 气泡样本级别分布热图（新增）
    if checkpoint and 'bubble_samples_per_level' in checkpoint.data:
        # 创建热图数据
        bubble_level_data = checkpoint.data['bubble_samples_per_level']
        if bubble_level_data:
            # 选择前20个气泡
            bubble_names = sorted(bubble_level_data.keys())[:20]
            
            heatmap_data = []
            for bubble in bubble_names:
                row = [bubble_level_data[bubble].get(cat, 0) for cat in categories]
                heatmap_data.append(row)
            
            if heatmap_data:
                sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlOrRd',
                           xticklabels=categories, yticklabels=bubble_names,
                           cbar_kws={'label': 'Sample Count'}, ax=axes[2, 1])
                axes[2, 1].set_xlabel('Occlusion Level (%)', fontsize=12)
                axes[2, 1].set_ylabel('Bubble Name', fontsize=12)
                axes[2, 1].set_title('Sample Distribution Heatmap (Top 20 Bubbles)', fontsize=13, fontweight='bold')
                plt.setp(axes[2, 1].get_yticklabels(), fontsize=8)
                plt.setp(axes[2, 1].get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    
    chart_path = os.path.join(output_base_path, 'dataset_statistics.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    console.print(f"\n[green]统计图表已保存至: {chart_path}[/green]")
    plt.close()

def save_failed_tasks_log(output_base_path: str):
    """保存失败任务日志"""
    if not failed_tasks:
        return
    
    log_path = os.path.join(output_base_path, 'failed_tasks.log')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"失败任务总数: {len(failed_tasks)}\n")
        f.write("="*60 + "\n\n")
        for i, task in enumerate(failed_tasks, 1):
            f.write(f"任务 {i}:\n")
            f.write(f"  路径: {task['path']}\n")
            f.write(f"  错误: {task['error']}\n\n")
    
    console.print(f"[yellow]失败任务日志已保存至: {log_path}[/yellow]")

def generate_validation_report(metadata_list: List[Dict], checkpoint: CheckpointManager, 
                               output_base_path: str, max_samples_per_bubble: int, num_intervals: int = 10):
    """
    生成数据验证报告
    
    验证内容：
    1. 每个气泡的样本数量不超过max_samples_per_bubble
    2. 每个遮挡级别内没有重复使用同一个气泡
    3. 列出所有气泡及其样本分布
    """
    report_path = os.path.join(output_base_path, 'validation_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("数据验证报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本数: {len(metadata_list)}\n")
        f.write(f"每气泡最大样本数限制: {max_samples_per_bubble}\n")
        f.write(f"遮挡区间数量: {num_intervals}\n\n")
        
        # 验证1：检查每个气泡的样本数
        f.write("-"*80 + "\n")
        f.write("验证1: 气泡样本数量检查\n")
        f.write("-"*80 + "\n")
        
        bubble_sample_counts = checkpoint.data.get('bubble_samples_per_level', {})
        violations = []
        
        for bubble_name, level_dist in bubble_sample_counts.items():
            total_samples = sum(level_dist.values())
            if total_samples > max_samples_per_bubble:
                violations.append((bubble_name, total_samples))
        
        if violations:
            f.write(f"❌ 发现 {len(violations)} 个气泡超过样本数限制:\n")
            for bubble, count in violations:
                f.write(f"  - {bubble}: {count} 样本（超出 {count - max_samples_per_bubble}）\n")
        else:
            f.write("✓ 所有气泡样本数量符合限制\n")
        
        f.write("\n")
        
        # 验证2：检查每个级别内是否有重复气泡
        f.write("-"*80 + "\n")
        f.write("验证2: 级别内气泡唯一性检查\n")
        f.write("-"*80 + "\n")
        
        df = pd.DataFrame(metadata_list)
        level_duplicates = []
        
        for category in df['category'].unique():
            level_df = df[df['category'] == category]
            bubble_counts = level_df['bubble_filename'].value_counts()
            duplicates = bubble_counts[bubble_counts > 1]
            
            if len(duplicates) > 0:
                level_duplicates.append((category, duplicates.to_dict()))
        
        if level_duplicates:
            f.write(f"❌ 发现 {len(level_duplicates)} 个级别存在重复气泡:\n")
            for category, dups in level_duplicates:
                f.write(f"  级别 {category}:\n")
                for bubble, count in dups.items():
                    f.write(f"    - {bubble}: {count} 个样本\n")
        else:
            f.write("✓ 所有级别内气泡唯一（每个气泡在同一级别最多出现1次）\n")
        
        f.write("\n")
        
        # 详细气泡分布列表
        f.write("-"*80 + "\n")
        f.write("气泡样本详细分布\n")
        f.write("-"*80 + "\n\n")
        
        categories = generate_occlusion_intervals(num_intervals)
        
        for bubble_name in sorted(bubble_sample_counts.keys()):
            level_dist = bubble_sample_counts[bubble_name]
            total = sum(level_dist.values())
            
            f.write(f"气泡: {bubble_name} (总样本数: {total})\n")
            f.write("  级别分布: ")
            
            dist_str = []
            for cat in categories:
                count = level_dist.get(cat, 0)
                if count > 0:
                    dist_str.append(f"{cat}({count})")
            
            f.write(", ".join(dist_str) if dist_str else "无样本")
            f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("验证报告结束\n")
        f.write("="*80 + "\n")
    
    console.print(f"[green]验证报告已保存至: {report_path}[/green]")
    
    # 返回验证结果摘要
    return {
        'total_violations': len(violations),
        'level_duplicates': len(level_duplicates),
        'valid': len(violations) == 0 and len(level_duplicates) == 0
    }

# ==================== 主处理函数（多线程版本） ====================
def main():
    """主函数 - 支持多线程、进度显示、优雅退出和断点续传"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='气泡图像遮挡处理工具（支持多线程和断点续传）')
    parser.add_argument('--input', type=str, 
                    #    default='C:/DataSet_DOOR/dataset_overlap/bubble_origin_dataset/*.png',
                       default='/home/yubd/mount/dataset/dataset_overlap/bubble_origin_dataset/*.png',
                       help='输入图像路径（支持通配符）')
    parser.add_argument('--output', type=str,
                    #    default='C:/DataSet_DOOR/dataset_overlap/training_dataset',
                       default='/home/yubd/mount/dataset/dataset_overlap/test20251117',
                       help='输出目录路径')
    parser.add_argument('--workers', type=int, default=0,
                       help='工作线程数（0=自动，根据CPU核心数）')
    parser.add_argument('--max-attempts', type=int, default=500,
                       help='每个气泡的最大尝试次数')
    parser.add_argument('--limit', type=int, default=10,
                       help='处理的气泡图像数量限制（0=不限制）')
    parser.add_argument('--resume', action='store_true',
                       help='从上次中断点恢复')
    parser.add_argument('--checkpoint-interval', type=int, default=5,
                       help='检查点保存间隔（处理气泡数）')
    parser.add_argument('--show-performance', action='store_true',
                       help='显示性能监控信息')
    parser.add_argument('--max-samples-per-bubble', type=int, default=10,
                       help='遮挡程度区间数量（默认10，即1-10,10-20...90-100十个区间）')
    
    args = parser.parse_args()
    
    # 设置信号处理
    setup_signal_handlers()
    
    # 确定工作线程数
    if args.workers <= 0:
        workers = max(1, os.cpu_count() - 1)
    else:
        workers = args.workers
    
    console.print(Panel.fit(
        f"[bold cyan]气泡图像遮挡处理工具[/bold cyan]\n\n"
        f"输入路径: {args.input}\n"
        f"输出路径: {args.output}\n"
        f"工作线程数: {workers}\n"
        f"最大尝试次数: {args.max_attempts}\n"
        f"断点续传: {'是' if args.resume else '否'}",
        title="配置信息"
    ))
    
    # 创建输出目录
    create_directory_structure(args.output, args.max_samples_per_bubble)
    
    # 初始化检查点管理器
    checkpoint = CheckpointManager(os.path.join(args.output, 'checkpoint.json'))
    
    # 读取气泡图像
    bubble_images = glob.glob(args.input)
    if args.limit > 0:
        bubble_images = bubble_images[:args.limit]
    
    if len(bubble_images) == 0:
        console.print(f"[red]未找到图像文件: {args.input}[/red]")
        return
    
    # 过滤已完成的气泡（如果是恢复模式）
    if args.resume:
        original_count = len(bubble_images)
        bubble_images = [img for img in bubble_images 
                        if not checkpoint.is_bubble_completed(os.path.splitext(os.path.basename(img))[0])]
        console.print(f"[yellow]恢复模式: 跳过 {original_count - len(bubble_images)} 个已完成的气泡[/yellow]")
    
    if len(bubble_images) == 0:
        console.print("[green]所有气泡已处理完成！[/green]")
        return
    
    console.print(f"\n[bold]找到 {len(bubble_images)} 个待处理的气泡图像[/bold]\n")
    
    # 遮挡级别（动态生成）
    occlusion_categories = generate_occlusion_intervals(args.max_samples_per_bubble)
    
    # 性能监控
    perf_monitor = PerformanceMonitor()
    
    # CSV文件路径
    csv_path = os.path.join(args.output, 'dataset_metadata.csv')
    csv_exists = os.path.exists(csv_path) and args.resume
    
    # 打开CSV文件（更新字段列表）
    csvfile = open(csv_path, 'a' if csv_exists else 'w', newline='', encoding='utf-8')
    fieldnames = ['filename', 'original_bubble', 'bubble_filename', 'occluding_bubbles', 'attempt_id',
                 'origin_path', 'blocked_path', 'segmented_path', 'mask_path',
                 'occlusion_ratio', 'visibility_ratio', 'category']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not csv_exists:
        writer.writeheader()
    
    try:
        # ==================== 使用Rich进度条显示 ====================
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=True
        ) as progress:
            
            # 主进度条：总体气泡处理进度
            main_task = progress.add_task(
                "[cyan]总体进度", 
                total=len(bubble_images)
            )
            
            # 次级任务ID字典（每个线程一个）
            thread_tasks = {}
            
            # 统计变量
            completed_bubbles = 0
            partial_bubbles = 0
            checkpoint_counter = 0
            
            # 使用线程池执行
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # 提交所有任务
                future_to_bubble = {
                    executor.submit(
                        process_single_bubble,
                        img_path,
                        bubble_images,
                        args.output,
                        checkpoint,
                        occlusion_categories,
                        args.max_attempts,
                        args.max_samples_per_bubble,
                        args.max_samples_per_bubble  # num_intervals参数
                    ): img_path
                    for img_path in bubble_images
                }
                
                # 处理完成的任务
                for future in as_completed(future_to_bubble):
                    if shutdown_flag:
                        console.print("\n[yellow]正在取消未完成的任务...[/yellow]")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    img_path = future_to_bubble[future]
                    
                    try:
                        result = future.result()
                        
                        # 更新主进度
                        progress.update(main_task, advance=1)
                        
                        # 根据状态更新统计
                        if result['status'] == 'success':
                            completed_bubbles += 1
                            status_icon = "✓"
                            status_color = "green"
                        else:
                            partial_bubbles += 1
                            status_icon = "⚠"
                            status_color = "yellow"
                        
                        # 显示详细信息（包括采样统计）
                        level_dist = checkpoint.get_bubble_level_distribution(result['bubble_name'])
                        level_str = ','.join([f"{k}({v})" for k, v in sorted(level_dist.items())])
                        
                        progress.console.print(
                            f"[{status_color}]{status_icon}[/{status_color}] "
                            f"{result['bubble_name']}: "
                            f"{len(result['completed_levels'])}/{args.max_samples_per_bubble} 级别 "
                            f"({result['attempts']} 次尝试) "
                            f"[dim]样本: {result.get('samples_saved', 0)}[/dim]"
                        )
                        
                        if level_str and args.show_performance:
                            progress.console.print(f"[dim]  级别分布: {level_str}[/dim]")
                        
                        # 性能监控
                        perf_monitor.increment_processed()
                        
                        # 定期保存检查点
                        checkpoint_counter += 1
                        if checkpoint_counter >= args.checkpoint_interval:
                            checkpoint.save()
                            checkpoint_counter = 0
                        
                        # 显示性能信息（可选）
                        if args.show_performance:
                            stats = perf_monitor.get_stats()
                            perf_info = f"[dim]速度: {stats['speed']:.2f} img/s"
                            if 'memory_mb' in stats:
                                perf_info += f" | 内存: {stats['memory_mb']:.1f}MB"
                            if 'cpu_percent' in stats:
                                perf_info += f" | CPU: {stats['cpu_percent']:.1f}%"
                            perf_info += "[/dim]"
                            progress.console.print(perf_info)
                        
                    except Exception as e:
                        logger.error(f"处理气泡失败 {img_path}: {e}")
                        progress.console.print(f"[red]✗ 处理失败: {os.path.basename(img_path)}[/red]")
                        progress.update(main_task, advance=1)
        
        # 最终保存检查点
        checkpoint.save()
        
        # 写入CSV（从检查点获取元数据）
        for metadata in checkpoint.data['metadata_list']:
            writer.writerow(metadata)
        
        csvfile.close()
        
        # ==================== 显示最终统计 ====================
        total_processed = len(bubble_images)
        stats_table = Table(title="处理统计", show_header=True, header_style="bold magenta")
        stats_table.add_column("项目", style="cyan")
        stats_table.add_column("数值", justify="right", style="green")
        stats_table.add_column("百分比", justify="right", style="yellow")
        
        stats_table.add_row(
            "总气泡数",
            str(total_processed),
            "100.0%"
        )
        stats_table.add_row(
            "完成所有级别",
            str(checkpoint.data['successful_bubbles']),
            f"{checkpoint.data['successful_bubbles']/total_processed*100:.1f}%"
        )
        stats_table.add_row(
            "部分完成",
            str(checkpoint.data['abandoned_bubbles']),
            f"{checkpoint.data['abandoned_bubbles']/total_processed*100:.1f}%"
        )
        stats_table.add_row(
            "生成样本总数",
            str(len(checkpoint.data['metadata_list'])),
            "-"
        )
        stats_table.add_row(
            "总尝试次数",
            str(checkpoint.data['total_attempts']),
            "-"
        )
        stats_table.add_row(
            "平均尝试次数/气泡",
            f"{checkpoint.data['total_attempts']/total_processed:.1f}",
            "-"
        )
        
        # 性能统计
        final_stats = perf_monitor.get_stats()
        stats_table.add_row(
            "总用时",
            f"{final_stats['elapsed_time']:.1f}秒",
            "-"
        )
        stats_table.add_row(
            "平均处理速度",
            f"{final_stats['speed']:.2f} 气泡/秒",
            "-"
        )
        
        console.print("\n")
        console.print(stats_table)
        
        console.print(f"\n[green]元数据已保存至: {csv_path}[/green]")
        
        # 保存失败任务日志
        if failed_tasks:
            save_failed_tasks_log(args.output)
        
        # 生成统计图表
        if checkpoint.data['metadata_list']:
            console.print("\n[cyan]正在生成统计图表...[/cyan]")
            generate_statistics_and_visualization(checkpoint.data['metadata_list'], args.output, checkpoint, args.max_samples_per_bubble)
        
        # 生成验证报告
        console.print("\n[cyan]正在生成验证报告...[/cyan]")
        validation_result = generate_validation_report(
            checkpoint.data['metadata_list'],
            checkpoint,
            args.output,
            args.max_samples_per_bubble,
            args.max_samples_per_bubble
        )
        
        if validation_result['valid']:
            console.print("[green]✓ 数据验证通过[/green]")
        else:
            console.print(f"[yellow]⚠ 数据验证发现问题: "
                         f"{validation_result['total_violations']} 个样本数超限, "
                         f"{validation_result['level_duplicates']} 个级别有重复[/yellow]")
        
        console.print("\n[bold green]所有任务完成！[/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]用户中断，正在保存进度...[/yellow]")
        checkpoint.save()
        csvfile.close()
        console.print("[green]进度已保存，可使用 --resume 参数恢复[/green]")
    except Exception as e:
        logger.error(f"主程序异常: {e}")
        raise
    finally:
        if not csvfile.closed:
            csvfile.close()

if __name__ == "__main__":
    main()
