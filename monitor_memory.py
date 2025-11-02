#!/usr/bin/env python3
"""
CUDA Memory Monitor
监测CUDA内存使用情况并生成报告
"""

import torch
import psutil
import time
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import threading


class CUDAMemoryMonitor:
    def __init__(self, log_file="cuda_memory_log.txt", interval=1.0):
        """
        Args:
            log_file: 内存日志文件
            interval: 监测间隔(秒)
        """
        self.log_file = log_file
        self.interval = interval
        self.monitoring = False
        self.data = []

    def get_memory_info(self):
        """获取详细的CUDA内存信息"""
        try:
            # PyTorch内存统计
            torch.cuda.reset_peak_memory_stats()

            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)    # GB

            # GPU总内存
            props = torch.cuda.get_device_properties(0)
            total_memory = props.total_memory / (1024**3)  # GB

            free = total_memory - allocated

            # 缓存信息
            cache_allocated = torch.cuda.memory_stats()['allocated_bytes.all.current'] / (1024**3)
            cache_reserved = torch.cuda.memory_stats()['reserved_bytes.all.current'] / (1024**3)

            return {
                'timestamp': datetime.now().isoformat(),
                'allocated_gb': round(allocated, 4),
                'reserved_gb': round(reserved, 4),
                'free_gb': round(free, 4),
                'total_gb': round(total_memory, 4),
                'used_percent': round((allocated / total_memory) * 100, 2),
            }
        except Exception as e:
            print(f"Error getting memory info: {e}")
            return None

    def print_memory_status(self):
        """打印当前内存状态"""
        info = self.get_memory_info()
        if info:
            print(f"[{info['timestamp']}] GPU Memory: {info['allocated_gb']:.2f}GB / {info['total_gb']:.2f}GB ({info['used_percent']:.1f}%) | Free: {info['free_gb']:.2f}GB")
            return info
        return None

    def monitor_loop(self):
        """监测循环"""
        while self.monitoring:
            info = self.print_memory_status()
            if info:
                self.data.append(info)
            time.sleep(self.interval)

    def start_monitoring(self):
        """启动后台监测"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"[MONITOR] Started CUDA memory monitoring (interval: {self.interval}s)")

    def stop_monitoring(self):
        """停止监测"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2)

        # 保存日志
        self.save_log()
        print(f"[MONITOR] Stopped monitoring. Logs saved to {self.log_file}")

    def save_log(self):
        """保存监测日志"""
        with open(self.log_file, 'w') as f:
            for data in self.data:
                f.write(json.dumps(data) + '\n')

    def analyze_memory_growth(self):
        """分析内存增长趋势"""
        if len(self.data) < 2:
            print("Not enough data to analyze")
            return

        print("\n" + "="*80)
        print("MEMORY GROWTH ANALYSIS")
        print("="*80)

        start_mem = self.data[0]['allocated_gb']
        peak_mem = max([d['allocated_gb'] for d in self.data])
        end_mem = self.data[-1]['allocated_gb']

        growth = end_mem - start_mem
        growth_rate = growth / len(self.data)  # per second

        print(f"Initial Memory:    {start_mem:.2f}GB")
        print(f"Peak Memory:       {peak_mem:.2f}GB")
        print(f"Final Memory:      {end_mem:.2f}GB")
        print(f"Total Growth:      {growth:.2f}GB")
        print(f"Growth Rate:       {growth_rate:.4f}GB/s")
        print(f"Samples Collected: {len(self.data)}")
        print(f"Duration:          {len(self.data) * self.interval:.1f}s")

        # 找到增长最快的时间段
        if len(self.data) > 10:
            max_segment_growth = 0
            max_segment_idx = 0
            for i in range(len(self.data) - 10):
                segment_growth = self.data[i+10]['allocated_gb'] - self.data[i]['allocated_gb']
                if segment_growth > max_segment_growth:
                    max_segment_growth = segment_growth
                    max_segment_idx = i

            if max_segment_growth > 0:
                print(f"\nFastest Growth Period:")
                print(f"  At: {self.data[max_segment_idx]['timestamp']}")
                print(f"  Growth in 10s: {max_segment_growth:.2f}GB")


def run_training_with_monitoring():
    """运行训练并监测内存"""
    monitor = CUDAMemoryMonitor()
    monitor.start_monitoring()

    try:
        print("\n" + "="*80)
        print("STARTING TRAINING WITH MEMORY MONITORING")
        print("="*80 + "\n")

        # 运行训练脚本
        result = subprocess.run(
            [sys.executable, 'train.py'],
            cwd=Path(__file__).parent
        )

        return result.returncode

    except KeyboardInterrupt:
        print("\n\n[MONITOR] Training interrupted by user")

    except Exception as e:
        print(f"\n\n[MONITOR] Error during training: {e}")

    finally:
        monitor.stop_monitoring()
        monitor.analyze_memory_growth()


if __name__ == '__main__':
    exit_code = run_training_with_monitoring()
    sys.exit(exit_code)
