import time
import pynvml
import os

# 初始化NVML库
pynvml.nvmlInit()

gpu_idxs = [0, 1, 2, 3]


def check_gpu_usage():
    gpu_usage_below_threshold = True
    for i in gpu_idxs:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_usage = utilization.gpu
        if gpu_usage > 5:
            gpu_usage_below_threshold = False
            break
    return gpu_usage_below_threshold


while True:
    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                 time.localtime())  # 获取当前时间并格式化
    if check_gpu_usage():
        print(
            f"{current_time}: All GPUs are below 5% usage. Executing command..."
        )
        os.system(
            "nohup bash scripts/eval/all.sh Meta-Llama-3-8B-alpaca-level-1 > eval.log &"
        )  # 这里的"ls"是示例命令，你可以替换为你需要执行的命令
        break
    else:
        print(f"{current_time}: At least one GPU is above 5% usage.")
    time.sleep(5 * 60)  # 每5分钟检查一次
