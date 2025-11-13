#!/usr/bin/env python3
"""
GPU性能监控脚本
用于监控训练过程中的GPU利用率和内存使用情况
"""

import subprocess
import time
import argparse
from datetime import datetime

def get_gpu_stats():
    """获取GPU统计信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip().split('\n')
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def monitor_gpu(interval=1, duration=None):
    """监控GPU状态"""
    print(f"开始监控GPU状态 (间隔: {interval}秒)")
    print("=" * 80)
    print(f"{'时间':<20} {'GPU利用率(%)':<12} {'显存使用(MB)':<15} {'显存总量(MB)':<15} {'温度(°C)':<10}")
    print("-" * 80)
    
    start_time = time.time()
    stats_history = []
    
    try:
        while True:
            current_time = datetime.now().strftime("%H:%M:%S")
            gpu_stats = get_gpu_stats()
            
            if gpu_stats:
                for i, stat in enumerate(gpu_stats):
                    if stat.strip():
                        parts = stat.split(', ')
                        if len(parts) >= 4:
                            util, mem_used, mem_total, temp = parts[:4]
                            print(f"{current_time:<20} {util:<12} {mem_used:<15} {mem_total:<15} {temp:<10}")
                            
                            # 记录统计信息用于分析
                            stats_history.append({
                                'time': current_time,
                                'gpu_id': i,
                                'utilization': int(util),
                                'memory_used': int(mem_used),
                                'memory_total': int(mem_total),
                                'temperature': int(temp)
                            })
            
            time.sleep(interval)
            
            # 检查是否达到持续时间
            if duration and (time.time() - start_time) >= duration:
                break
                
    except KeyboardInterrupt:
        print("\n监控已停止")
    
    # 输出统计摘要
    if stats_history:
        print("\n" + "=" * 80)
        print("统计摘要:")
        
        # 按GPU分组统计
        gpu_groups = {}
        for stat in stats_history:
            gpu_id = stat['gpu_id']
            if gpu_id not in gpu_groups:
                gpu_groups[gpu_id] = []
            gpu_groups[gpu_id].append(stat)
        
        for gpu_id, gpu_stats in gpu_groups.items():
            utils = [s['utilization'] for s in gpu_stats]
            avg_util = sum(utils) / len(utils)
            min_util = min(utils)
            max_util = max(utils)
            
            print(f"GPU {gpu_id}:")
            print(f"  平均利用率: {avg_util:.1f}%")
            print(f"  最小利用率: {min_util}%")
            print(f"  最大利用率: {max_util}%")
            print(f"  波动幅度: {max_util - min_util}%")
            
            # 分析波动情况
            if avg_util >= 85:
                print(f"  状态: ✅ 优秀 (GPU利用率高)")
            elif avg_util >= 70:
                print(f"  状态: ⚠️  良好 (GPU利用率中等，可能仍有优化空间)")
            else:
                print(f"  状态: ❌ 需要优化 (GPU利用率低)")
            
            if max_util - min_util <= 10:
                print(f"  波动: ✅ 稳定 (波动小)")
            elif max_util - min_util <= 20:
                print(f"  波动: ⚠️  中等 (有一定波动)")
            else:
                print(f"  波动: ❌ 不稳定 (波动大)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU性能监控工具")
    parser.add_argument("--interval", type=int, default=1, help="监控间隔(秒)")
    parser.add_argument("--duration", type=int, help="监控持续时间(秒)")
    args = parser.parse_args()
    
    monitor_gpu(args.interval, args.duration)
