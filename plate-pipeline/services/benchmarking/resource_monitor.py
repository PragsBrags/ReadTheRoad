from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:  # pragma: no cover - exercised in deployments without psutil
    psutil = None

try:
    import pynvml
except ImportError:  # pragma: no cover - GPU metrics are optional
    pynvml = None

logger = logging.getLogger(__name__)


class ResourceMonitor:
    def __init__(
        self,
        job_id,
        interval,
        enabled,
        persistence=None,
        pid: int | None = None,
        app_root: str | os.PathLike[str] | None = None,
    ):
        self.interval = max(float(interval), 0.0)
        self.enabled = enabled
        self.job_id = job_id
        self.persistence = persistence
        self.root_pid = pid or os.getpid()
        self.app_root = (
            Path(app_root).resolve()
            if app_root
            else Path(__file__).parents[2].resolve()
        )
        self.stop_event = threading.Event()
        self.thread = None
        self._samples: list[dict[str, Any]] = []
        self._samples_lock = threading.Lock()
        self._flushed = False
        self._last_gpu_sample_time = 0

    def start(self):
        if self.enabled is not True:
            return

        self._prime_cpu_counters()
        self._last_gpu_sample_time = int(time.time() * 1_000_000)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop_event.is_set():
            cpu_usage = self.logging_cpu()
            if self.stop_event.is_set():
                break

            mem_usage = self.logging_mem()
            gpu_usage, vram_usage, gpu_name, power_usage = self.logging_gpu()

            self._record_sample(
                cpu_percent=self._average(cpu_usage),
                ram_usage=self._average(mem_usage),
                gpu_percent=self._average(gpu_usage),
                gpu_ram_usage=self._average(vram_usage),
                gpu_power_usage=self._average(power_usage),
                gpu_name=gpu_name,
            )

    def logging_cpu(self):
        if psutil is None:
            self._wait_for_interval()
            return []

        processes = self._app_processes()
        for process in processes:
            self._process_cpu_percent(process)

        if self._wait_for_interval():
            return []

        cpu_percent = sum(
            self._process_cpu_percent(process)
            for process in processes
        )
        cpu_capacity = psutil.cpu_count() or 1
        return [cpu_percent / cpu_capacity]

    def logging_mem(self):
        if psutil is None:
            return []

        mem_percent = sum(
            self._process_memory_percent(process)
            for process in self._app_processes()
        )
        return [mem_percent]

    def logging_gpu(self):
        if pynvml is None:
            return [], [], None, []

        app_pids = {process.pid for process in self._app_processes()}
        if not app_pids:
            return [], [], None, []

        gpu_names = []
        gpu_percentages = []
        mem_usage = []
        power_usage = []
        newest_sample_time = self._last_gpu_sample_time

        try:
            pynvml.nvmlInit()
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                gpu_name = self._decode_gpu_name(pynvml.nvmlDeviceGetName(handle))
                gpu_names.append(gpu_name)

                app_gpu_processes = self._app_gpu_processes(handle, app_pids)
                if not app_gpu_processes:
                    continue

                used_gpu_memory = sum(
                    self._used_gpu_memory(process)
                    for process in app_gpu_processes
                )
                mem_usage.append(used_gpu_memory / 1024**2)

                gpu_utilization, sample_time = self._process_gpu_utilization(
                    handle,
                    app_pids,
                    self._last_gpu_sample_time,
                )
                if sample_time is not None:
                    newest_sample_time = max(newest_sample_time, sample_time)
                if gpu_utilization is not None:
                    gpu_percentages.append(gpu_utilization)
                else:
                    device_utilization = self._device_gpu_utilization(handle)
                    if device_utilization is not None:
                        gpu_percentages.append(device_utilization)

                device_power = self._device_power_usage(handle)
                if device_power is not None:
                    power_usage.append(device_power)

            self._last_gpu_sample_time = newest_sample_time
            return gpu_percentages, mem_usage, ", ".join(gpu_names) or None, power_usage

        except Exception as exc:
            logger.debug("Unable to collect app GPU metrics: %s", exc)
            return [], [], None, []
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval + 2)
        if not self._has_samples():
            self._record_instant_sample()
        self._flush_resource_log()

    def _wait_for_interval(self) -> bool:
        return self.stop_event.wait(self.interval)

    @staticmethod
    def _resource_log_row(**values):
        from services.database.schema import ResourceLoggingCreate

        return ResourceLoggingCreate(**values)

    def _record_sample(self, **values) -> None:
        with self._samples_lock:
            self._samples.append(values)

    def _has_samples(self) -> bool:
        with self._samples_lock:
            return bool(self._samples)

    def _record_instant_sample(self) -> None:
        cpu_usage = self._instant_cpu()
        mem_usage = self.logging_mem()
        gpu_usage, vram_usage, gpu_name, power_usage = self.logging_gpu()

        self._record_sample(
            cpu_percent=self._average(cpu_usage),
            ram_usage=self._average(mem_usage),
            gpu_percent=self._average(gpu_usage),
            gpu_ram_usage=self._average(vram_usage),
            gpu_power_usage=self._average(power_usage),
            gpu_name=gpu_name,
        )

    def _flush_resource_log(self) -> None:
        if self.persistence is None or self._flushed:
            return

        self._flushed = True
        aggregate = self._aggregate_samples()
        if aggregate is None:
            return

        self.persistence.save_resource_log(
            self._resource_log_row(job_id=self.job_id, **aggregate)
        )

    def _aggregate_samples(self) -> dict[str, Any] | None:
        with self._samples_lock:
            samples = list(self._samples)

        if not samples:
            return None

        return {
            "cpu_percent": self._average_present(samples, "cpu_percent"),
            "ram_usage": self._average_present(samples, "ram_usage"),
            "gpu_percent": self._average_present(samples, "gpu_percent"),
            "gpu_ram_usage": self._average_present(samples, "gpu_ram_usage"),
            "gpu_power_usage": self._average_present(samples, "gpu_power_usage"),
            "gpu_name": self._aggregate_gpu_names(samples),
        }

    @staticmethod
    def _average_present(samples: list[dict[str, Any]], key: str):
        values = [
            sample[key]
            for sample in samples
            if sample.get(key) is not None
        ]
        return sum(values) / len(values) if values else None

    @staticmethod
    def _aggregate_gpu_names(samples: list[dict[str, Any]]) -> str | None:
        names = []
        for sample in samples:
            name = sample.get("gpu_name")
            if name and name not in names:
                names.append(name)
        return ", ".join(names) or None

    def _prime_cpu_counters(self) -> None:
        if psutil is None:
            return

        for process in self._app_processes():
            self._process_cpu_percent(process)

    def _instant_cpu(self):
        if psutil is None:
            return []

        cpu_percent = sum(
            self._process_cpu_percent(process)
            for process in self._app_processes()
        )
        cpu_capacity = psutil.cpu_count() or 1
        return [cpu_percent / cpu_capacity]

    def _app_processes(self):
        if psutil is None:
            return []

        process_by_pid = {}
        try:
            root = psutil.Process(self.root_pid)
        except self._psutil_exceptions():
            root = None

        if root is not None:
            process_by_pid[root.pid] = root
            try:
                for child in root.children(recursive=True):
                    process_by_pid[child.pid] = child
            except self._psutil_exceptions():
                pass

        for process in self._iter_app_root_processes():
            process_by_pid[process.pid] = process

        active_processes = []
        for process in process_by_pid.values():
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    active_processes.append(process)
            except self._psutil_exceptions():
                continue
        return active_processes

    @staticmethod
    def _process_cpu_percent(process) -> float:
        try:
            return max(float(process.cpu_percent(interval=None)), 0.0)
        except Exception:
            return 0.0

    @staticmethod
    def _process_memory_percent(process) -> float:
        try:
            return max(float(process.memory_percent()), 0.0)
        except Exception:
            return 0.0

    def _iter_app_root_processes(self):
        if psutil is None:
            return []

        matches = []
        attrs = ["pid", "cwd", "cmdline"]
        for process in psutil.process_iter(attrs=attrs):
            try:
                info = process.info
                cwd = info.get("cwd")
                if cwd and self._is_under_app_root(cwd):
                    matches.append(process)
                    continue

                cmdline = info.get("cmdline") or []
                if any(self._command_part_matches_app_root(part) for part in cmdline):
                    matches.append(process)
            except self._psutil_exceptions():
                continue
        return matches

    def _is_under_app_root(self, path: str) -> bool:
        try:
            Path(path).resolve().relative_to(self.app_root)
            return True
        except (OSError, ValueError):
            return False

    def _command_part_matches_app_root(self, value: str) -> bool:
        try:
            path = Path(value).resolve()
        except (OSError, ValueError):
            return False

        return path == self.app_root or self.app_root in path.parents

    def _app_gpu_processes(self, handle, app_pids: set[int]):
        by_pid: dict[int, Any] = {}
        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
        ):
            getter = getattr(pynvml, getter_name, None)
            if getter is None:
                continue
            try:
                for process in getter(handle):
                    pid = int(getattr(process, "pid", 0))
                    if pid not in app_pids:
                        continue

                    existing = by_pid.get(pid)
                    if existing is None or (
                        self._used_gpu_memory(process) > self._used_gpu_memory(existing)
                    ):
                        by_pid[pid] = process
            except Exception:
                continue
        return list(by_pid.values())

    def _process_gpu_utilization(
        self,
        handle,
        app_pids: set[int],
        last_sample_time: int,
    ) -> tuple[float | None, int | None]:
        getter = getattr(pynvml, "nvmlDeviceGetProcessUtilization", None)
        if getter is None:
            return None, None

        try:
            samples = getter(handle, last_sample_time)
        except Exception:
            return None, None

        if not samples:
            return None, None

        newest_sample_time = max(
            int(getattr(sample, "timeStamp", last_sample_time))
            for sample in samples
        )
        app_samples = [
            sample
            for sample in samples
            if int(getattr(sample, "pid", 0)) in app_pids
        ]
        if not app_samples:
            return None, newest_sample_time

        return (
            sum(float(getattr(sample, "smUtil", 0.0)) for sample in app_samples)
            / len(app_samples),
            newest_sample_time,
        )

    @staticmethod
    def _device_gpu_utilization(handle) -> float | None:
        getter = getattr(pynvml, "nvmlDeviceGetUtilizationRates", None)
        if getter is None:
            return None

        try:
            return max(float(getter(handle).gpu), 0.0)
        except Exception:
            return None

    @staticmethod
    def _device_power_usage(handle) -> float | None:
        getter = getattr(pynvml, "nvmlDeviceGetPowerUsage", None)
        if getter is None:
            return None

        try:
            return max(float(getter(handle)) / 1000.0, 0.0)
        except Exception:
            return None

    @staticmethod
    def _used_gpu_memory(process) -> float:
        value = getattr(process, "usedGpuMemory", 0)
        unavailable = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE", None)
        if value is None or value == unavailable:
            return 0.0

        try:
            return max(float(value), 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _decode_gpu_name(name) -> str:
        if isinstance(name, bytes):
            return name.decode("utf-8", errors="replace")
        return str(name)

    @staticmethod
    def _average(values):
        return sum(values) / len(values) if values else None

    @staticmethod
    def _psutil_exceptions():
        if psutil is None:
            return (Exception,)
        return (
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess,
        )
