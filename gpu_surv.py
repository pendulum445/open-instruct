import argparse
import os
import time

import pynvml

pynvml.nvmlInit()


def check_gpu_usage(gpu_idxs, threshold=5):
    gpu_usage_below_threshold = True
    for i in gpu_idxs:
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_usage = utilization.gpu
        if gpu_usage > threshold:
            gpu_usage_below_threshold = False
            break
    return gpu_usage_below_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_idxs",
                        type=str,
                        required=True,
                        help="Comma-separated list of GPU indices")
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="Command to execute when GPU usage is below the threshold")
    args = parser.parse_args()

    gpu_idxs = [int(idx) for idx in args.gpu_idxs.split(",")]
    command = args.command

    while True:
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        if check_gpu_usage(gpu_idxs):
            print(
                f"{current_time}: All GPUs are below 5% usage. Executing command..."
            )
            os.system(f"nohup {command} > eval.log &")
            break
        else:
            print(f"{current_time}: At least one GPU is above 5% usage.")
        time.sleep(5 * 60)


if __name__ == "__main__":
    main()
