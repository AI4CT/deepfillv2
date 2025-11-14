#!/usr/bin/env python3
"""
GPU使用率监控脚本
用于实时监控训练过程中的GPU利用率，验证优化效果
"""

import time
import subprocess
import json
from datetime import datetime

def get_gpu_stats():
    """获取GPU统计信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        stats = []
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 8:
                stats.append({
                    'gpu_id': int(parts[0]),
                    'name': parts[1],
                    'gpu_util': int(parts[2]),
                    'mem_util': int(parts[3]),
                    'mem_used': int(parts[4]),
                    'mem_total': int(parts[5]),
                    'temp': int(parts[6]),
                    'power': float(parts[7])
                })
        
        return stats
    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return []

def print_gpu_stats(stats):
    """打印GPU统计信息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== {timestamp} ===")
    
    total_gpu_util = 0
    total_mem_util = 0
    active_gpus = 0
    
    for gpu in stats:
        print(f"GPU {gpu['gpu_id']} ({gpu['name']}):")
        print(f"  GPU利用率: {gpu['gpu_util']:3d}% | 内存利用率: {gpu['mem_util']:3d}%")
        print(f"  显存使用: {gpu['mem_used']:5d}/{gpu['mem_total']:5d} MB ({gpu['mem_used']*100/gpu['mem_total']:5.1f}%)")
        print(f"  温度: {gpu['temp']:2d}°C | 功耗: {gpu['power']:6.1f}W")
        
        total_gpu_util += gpu['gpu_util']
        total_mem_util += gpu['mem_util']
        active_gpus += 1
    
    if active_gpus > 0:
        avg_gpu_util = total_gpu_util / active_gpus
        avg_mem_util = total_mem_util / active_gpus
        print(f"\n平均 GPU利用率: {avg_gpu_util:.1f}% | 平均内存利用率: {avg_mem_util:.1f}%")
        
        # 性能评估
        if avg_gpu_util >= 90:
            print("✅ GPU利用率优秀")
        elif avg_gpu_util >= 75:
            print("⚡ GPU利用率良好")
        elif avg_gpu_util >= 50:
            print("⚠️  GPU利用率一般，建议进一步优化")
        else:
            print("❌ GPU利用率较低，需要优化")

def main():
    """主监控循环"""
    print("开始监控GPU使用率...")
    print("按 Ctrl+C 停止监控")
    
    try:
        while True:
            stats = get_gpu_stats()
            if stats:
                print_gpu_stats(stats)
            
            time.sleep(5)  # 每5秒更新一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    main()
