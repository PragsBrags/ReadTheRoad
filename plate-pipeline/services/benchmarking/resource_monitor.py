import psutil
import time
import os
import threading 

from pynvml import *
from services.database.schema import ResourceLoggingCreate

class ResourceMonitor:
    def __init__(self,job_id, interval, enabled, persistence=None):
        self.interval = interval
        self.enabled = enabled
        self.job_id = job_id
        self.persistence = persistence
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.enabled is not True:
            return
        
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            cpu_usage = self.logging_cpu()
            mem_usage = self.logging_mem()
            gpu_usage, vram_usage, gpu_name, power_usage = self.logging_gpu()

            cpu_avg = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
            mem_avg = sum(mem_usage) / len(mem_usage) if mem_usage else 0

            gpu_avg = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0
            vram_avg = sum(vram_usage) / len(vram_usage) if vram_usage else 0
            power_avg = sum(power_usage) / len(power_usage) if power_usage else 0

            if self.persistence:
                    self.persistence.save_resource_log(
                        ResourceLoggingCreate(
                            job_id=self.job_id,
                            cpu_percent=cpu_avg,
                            ram_usage=mem_avg,
                            gpu_percent=gpu_avg,
                            gpu_ram_usage=vram_avg,
                            gpu_power_usage=power_avg,
                            gpu_name=gpu_name,
                        )
                    )
    
    def logging_cpu(self):
        cpu_avg = []

        cpu_percent = psutil.cpu_percent(interval=self.interval)
        cpu_avg.append(cpu_percent)

        return cpu_avg
    
    def logging_mem(self):
        mem_usage = []

        mem = psutil.virtual_memory()
        mem_usage.append(mem.percent)

        return mem_usage
    
    def logging_gpu(self):
    
        try:
            nvmlInit()
            count = nvmlDeviceGetCount()
            if count == 0:
                return [], [], None, []

            handle = nvmlDeviceGetHandleByIndex(0)

            gpu_percentages = []
            mem_usage = []
            power_usage = []

            name = nvmlDeviceGetName(handle)
                
            # Query memory layout
            info = nvmlDeviceGetMemoryInfo(handle)
            utilization = nvmlDeviceGetUtilizationRates(handle)
            power_mw = nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0  # Convert to Watts

            used_mem = info.used / 1024**2
            mem_usage.append(used_mem)
            gpu_percentages.append(utilization.gpu)
            power_usage.append(power_w)

            return gpu_percentages, mem_usage, name, power_usage
        
        except:
            return [], [], None, []
    
    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval + 2)