import torch
import numpy as np
import time
import psutil
import os
import matplotlib.pyplot as plt
from functools import wraps

class GPUMonitor:
    """
    Monitor GPU usage during simulation runs.
    Provides utilities for tracking memory usage and optimizing GPU resources.
    """
    def __init__(self, log_interval=10, log_dir=None):
        self.log_interval = log_interval  # seconds
        self.log_dir = log_dir
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.memory_usage = []
        self.timestamps = []
        self.utilization = []
        self.last_log_time = time.time()
        self.start_time = time.time()
        self.enabled = torch.cuda.is_available()
        
        if self.enabled:
            print(f"GPU Monitoring initialized on {torch.cuda.get_device_name(0)}")
            self.memory_allocated_start = torch.cuda.memory_allocated() / 1e6  # MB
            self.memory_reserved_start = torch.cuda.memory_reserved() / 1e6    # MB
            print(f"Initial GPU memory: {self.memory_allocated_start:.2f}MB allocated, "
                  f"{self.memory_reserved_start:.2f}MB reserved")
        else:
            print("GPU monitoring disabled - no CUDA device available")
    
    def update(self):
        """Record current GPU metrics"""
        if not self.enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            memory_allocated = torch.cuda.memory_allocated() / 1e6  # MB
            memory_reserved = torch.cuda.memory_reserved() / 1e6    # MB
            
            # Get GPU utilization (requires pynvml or similar)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
            except:
                gpu_util = -1  # Unable to get utilization
            
            self.memory_usage.append((memory_allocated, memory_reserved))
            self.utilization.append(gpu_util)
            self.timestamps.append(current_time - self.start_time)
            self.last_log_time = current_time
            
            print(f"GPU Memory: {memory_allocated:.2f}MB allocated, {memory_reserved:.2f}MB reserved, "
                  f"Utilization: {gpu_util}%")
    
    def plot_usage(self):
        """Plot GPU usage statistics"""
        if not self.enabled or len(self.timestamps) == 0:
            print("No GPU data to plot")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Memory plot
        plt.subplot(2, 1, 1)
        mem_allocated = [m[0] for m in self.memory_usage]
        mem_reserved = [m[1] for m in self.memory_usage]
        plt.plot(self.timestamps, mem_allocated, label='Allocated Memory')
        plt.plot(self.timestamps, mem_reserved, label='Reserved Memory')
        plt.title('GPU Memory Usage Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory (MB)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Utilization plot
        plt.subplot(2, 1, 2)
        plt.plot(self.timestamps, self.utilization, color='green')
        plt.title('GPU Utilization Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Utilization (%)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.log_dir:
            plt.savefig(os.path.join(self.log_dir, 'gpu_usage.png'))
            print(f"GPU usage plot saved to {os.path.join(self.log_dir, 'gpu_usage.png')}")
        else:
            plt.show()
    
    def summary(self):
        """Print summary of GPU usage"""
        if not self.enabled or len(self.timestamps) == 0:
            print("No GPU data available")
            return
        
        max_allocated = max([m[0] for m in self.memory_usage])
        avg_allocated = sum([m[0] for m in self.memory_usage]) / len(self.memory_usage)
        max_reserved = max([m[1] for m in self.memory_usage])
        avg_reserved = sum([m[1] for m in self.memory_usage]) / len(self.memory_usage)
        
        if -1 not in self.utilization:
            max_util = max(self.utilization)
            avg_util = sum(self.utilization) / len(self.utilization)
            util_summary = f"GPU Utilization: Max {max_util:.2f}%, Avg {avg_util:.2f}%"
        else:
            util_summary = "GPU Utilization data not available"
        
        summary = f"""
        === GPU USAGE SUMMARY ===
        Runtime: {self.timestamps[-1]:.2f} seconds
        Max Memory Allocated: {max_allocated:.2f}MB
        Avg Memory Allocated: {avg_allocated:.2f}MB
        Max Memory Reserved: {max_reserved:.2f}MB
        Avg Memory Reserved: {avg_reserved:.2f}MB
        {util_summary}
        """
        
        print(summary)
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, 'gpu_summary.txt'), 'w') as f:
                f.write(summary)


def gpu_optimized(func):
    """Decorator to optimize and profile GPU usage for functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
        
        # Clear cache before execution
        torch.cuda.empty_cache()
        
        # Record starting memory
        start_mem = torch.cuda.memory_allocated() / 1e6
        start_time = time.time()
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Record ending memory and time
        end_mem = torch.cuda.memory_allocated() / 1e6
        end_time = time.time()
        
        # Print usage information
        print(f"Function {func.__name__} - Memory: {end_mem-start_mem:.2f}MB, Time: {end_time-start_time:.4f}s")
        
        return result
    return wrapper

def optimize_tensor_layout(tensor):
    """Optimize tensor memory layout for GPU operations"""
    if not torch.cuda.is_available() or not tensor.is_cuda:
        return tensor
    
    # Ensure tensor is contiguous in memory
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Apply memory format optimizations based on tensor dimensions
    if tensor.dim() == 4:  # For 4D tensors (e.g., images, feature maps)
        tensor = tensor.to(memory_format=torch.channels_last)
    
    return tensor

def batch_process(data_list, batch_size, process_func, device=None):
    """Process data in batches to optimize GPU usage"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    results = []
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i+batch_size]
        batch_tensor = torch.tensor(batch, device=device)
        batch_results = process_func(batch_tensor)
        
        # Move results back to CPU if needed
        if isinstance(batch_results, torch.Tensor) and batch_results.is_cuda:
            batch_results = batch_results.cpu()
        
        results.extend(batch_results.numpy() if isinstance(batch_results, torch.Tensor) else batch_results)
    
    return results
