#!/usr/bin/env python3
"""
性能监控脚本，用于实时监控训练过程中的GPU和CPU使用情况
"""

import time
import subprocess
import psutil
import GPUtil
from datetime import datetime

def get_gpu_stats():
    """获取GPU统计信息"""
    try:
        gpus = GPUtil.getGPUs()
        stats = []
        for gpu in gpus:
            stats.append({
                'id': gpu.id,
                'name': gpu.name,
                'load': f"{gpu.load*100:.1f}%",
                'memory_used': f"{gpu.memoryUsed}MB",
                'memory_total': f"{gpu.memoryTotal}MB",
                'memory_util': f"{(gpu.memoryUsed/gpu.memoryTotal)*100:.1f}%",
                'temperature': f"{gpu.temperature}°C"
            })
        return stats
    except Exception as e:
        return [{'error': str(e)}]

def get_cpu_stats():
    """获取CPU统计信息"""
    return {
        'usage': f"{psutil.cpu_percent(interval=1):.1f}%",
        'cores': psutil.cpu_count(logical=False),
        'threads': psutil.cpu_count(logical=True),
        'memory_used': f"{psutil.virtual_memory().used / (1024**3):.1f}GB",
        'memory_total': f"{psutil.virtual_memory().total / (1024**3):.1f}GB",
        'memory_percent': f"{psutil.virtual_memory().percent:.1f}%"
    }

def get_process_stats():
    """获取训练进程统计信息"""
    try:
        # 查找python训练进程
        result = subprocess.run(['pgrep', '-f', 'python.*train'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            processes = []
            for pid in pids:
                try:
                    p = psutil.Process(int(pid))
                    processes.append({
                        'pid': pid,
                        'name': p.name(),
                        'cpu_percent': f"{p.cpu_percent():.1f}%",
                        'memory_mb': f"{p.memory_info().rss / (1024**2):.1f}MB",
                        'num_threads': p.num_threads()
                    })
                except:
                    continue
            return processes
    except:
        pass
    return []

def main():
    """主监控循环"""
    print("开始性能监控...")
    print("=" * 80)
    
    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"\n{timestamp}")
            print("-" * 80)
            
            # GPU统计
            gpu_stats = get_gpu_stats()
            print("GPU状态:")
            for gpu in gpu_stats:
                if 'error' in gpu:
                    print(f"  错误: {gpu['error']}")
                else:
                    print(f"  GPU {gpu['id']}: {gpu['name']}")
                    print(f"    利用率: {gpu['load']} | 显存: {gpu['memory_used']}/{gpu['memory_total']} ({gpu['memory_util']}) | 温度: {gpu['temperature']}")
            
            # CPU统计
            cpu_stats = get_cpu_stats()
            print(f"\nCPU状态:")
            print(f"  利用率: {cpu_stats['usage']} | 核心: {cpu_stats['cores']}/{cpu_stats['threads']}")
            print(f"  内存: {cpu_stats['memory_used']}/{cpu_stats['memory_total']} ({cpu_stats['memory_percent']})")
            
            # 进程统计
            process_stats = get_process_stats()
            if process_stats:
                print(f"\n训练进程:")
                for proc in process_stats:
                    print(f"  PID {proc['pid']}: {proc['name']} | CPU: {proc['cpu_percent']} | 内存: {proc['memory_mb']} | 线程: {proc['num_threads']}")
            
            time.sleep(5)  # 每5秒更新一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")

if __name__ == "__main__":
    main()
