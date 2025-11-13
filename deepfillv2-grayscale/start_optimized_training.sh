#!/bin/bash

# GPU优化训练启动脚本
# 自动应用所有性能优化参数

echo "🚀 启动GPU优化训练"
echo "=================="

# 检查GPU是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 未找到nvidia-smi，请确保GPU驱动已安装"
    exit 1
fi

# 显示GPU信息
echo "📊 GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo ""

# 检查CPU核心数
CPU_CORES=$(nproc)
echo "🖥️  CPU核心数: $CPU_CORES"

# 根据CPU核心数推荐num_workers
RECOMMENDED_WORKERS=$((CPU_CORES / 2))
if [ $RECOMMENDED_WORKERS -gt 16 ]; then
    RECOMMENDED_WORKERS=16
fi

echo "💡 推荐num_workers: $RECOMMENDED_WORKERS"
echo ""

# 默认优化参数
BATCH_SIZE=128
NUM_WORKERS=$RECOMMENDED_WORKERS
EVAL_INTERVAL=5
CHECKPOINT_INTERVAL=5

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --eval_interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --batch_size N     批量大小 (默认: 128)"
            echo "  --num_workers N    数据加载进程数 (默认: CPU核心数/2)"
            echo "  --eval_interval N  评估间隔epoch数 (默认: 5)"
            echo "  --help, -h         显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "🔧 训练参数:"
echo "  batch_size: $BATCH_SIZE"
echo "  num_workers: $NUM_WORKERS"
echo "  eval_interval: $EVAL_INTERVAL"
echo "  checkpoint_interval: $CHECKPOINT_INTERVAL"
echo ""

# 启动GPU监控（后台）
echo "📈 启动GPU监控..."
python3 monitor_gpu.py --interval 2 &
MONITOR_PID=$!

# 启动训练
echo "🏃‍♂️ 开始训练..."
echo "按 Ctrl+C 停止训练"
echo ""

python3 train.py \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --eval_interval $EVAL_INTERVAL \
    --checkpoint_interval $CHECKPOINT_INTERVAL \
    --multi_gpu True \
    --cudnn_benchmark True

# 清理：停止GPU监控
echo ""
echo "🛑 停止GPU监控..."
kill $MONITOR_PID 2>/dev/null

echo "✅ 训练完成"
