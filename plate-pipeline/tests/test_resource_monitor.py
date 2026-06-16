from types import SimpleNamespace

from services.benchmarking import resource_monitor
from services.benchmarking.resource_monitor import ResourceMonitor


class FakePsutilError(Exception):
    pass


class FakeProcess:
    def __init__(self, pid, cpu_samples=None, memory_percent=0.0, children=None):
        self.pid = pid
        self._cpu_samples = list(cpu_samples or [0.0])
        self._memory_percent = memory_percent
        self._children = children or []

    def children(self, recursive=True):
        return self._children

    def cpu_percent(self, interval=None):
        if len(self._cpu_samples) > 1:
            return self._cpu_samples.pop(0)
        return self._cpu_samples[0]

    def is_running(self):
        return True

    def memory_percent(self):
        return self._memory_percent

    def status(self):
        return "running"


class FakePsutil:
    STATUS_ZOMBIE = "zombie"
    NoSuchProcess = FakePsutilError
    AccessDenied = FakePsutilError
    ZombieProcess = FakePsutilError

    def __init__(self, root_process, cpu_count=4):
        self.root_process = root_process
        self._cpu_count = cpu_count

    def Process(self, pid):
        if pid != self.root_process.pid:
            raise self.NoSuchProcess(pid)
        return self.root_process

    def cpu_count(self):
        return self._cpu_count

    def process_iter(self, attrs=None):
        return []


class FakeNvml:
    NVML_VALUE_NOT_AVAILABLE = -1

    def __init__(self):
        self.shutdown_called = False

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlDeviceGetCount(self):
        return 1

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"gpu-{index}"

    def nvmlDeviceGetName(self, handle):
        return b"Fake GPU"

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return [
            SimpleNamespace(pid=101, usedGpuMemory=256 * 1024**2),
            SimpleNamespace(pid=999, usedGpuMemory=1024 * 1024**2),
        ]

    def nvmlDeviceGetGraphicsRunningProcesses(self, handle):
        return []

    def nvmlDeviceGetProcessUtilization(self, handle, last_sample_time):
        return [
            SimpleNamespace(pid=101, smUtil=35, timeStamp=last_sample_time + 10),
            SimpleNamespace(pid=999, smUtil=90, timeStamp=last_sample_time + 10),
        ]

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=75)

    def nvmlDeviceGetPowerUsage(self, handle):
        return 50_000


class FakeNvmlWithoutProcessUtilization(FakeNvml):
    nvmlDeviceGetProcessUtilization = None


class FakePersistence:
    def __init__(self):
        self.rows = []

    def save_resource_log(self, row):
        self.rows.append(row)


def test_cpu_and_memory_metrics_are_limited_to_app_process_tree(monkeypatch):
    child_process = FakeProcess(
        pid=101,
        cpu_samples=[0.0, 20.0],
        memory_percent=1.5,
    )
    root_process = FakeProcess(
        pid=100,
        cpu_samples=[0.0, 40.0],
        memory_percent=2.5,
        children=[child_process],
    )

    monkeypatch.setattr(resource_monitor, "psutil", FakePsutil(root_process))

    monitor = ResourceMonitor("job-1", interval=0, enabled=True, pid=100)

    assert monitor.logging_cpu() == [15.0]
    assert monitor.logging_mem() == [4.0]


def test_gpu_metrics_are_limited_to_app_process_tree(monkeypatch):
    child_process = FakeProcess(pid=101)
    root_process = FakeProcess(pid=100, children=[child_process])
    fake_nvml = FakeNvml()

    monkeypatch.setattr(resource_monitor, "psutil", FakePsutil(root_process))
    monkeypatch.setattr(resource_monitor, "pynvml", fake_nvml)

    monitor = ResourceMonitor("job-1", interval=0, enabled=True, pid=100)
    monitor._last_gpu_sample_time = 1_000

    gpu_usage, vram_usage, gpu_name, power_usage = monitor.logging_gpu()

    assert gpu_usage == [35.0]
    assert vram_usage == [256.0]
    assert gpu_name == "Fake GPU"
    assert power_usage == [50.0]


def test_resource_log_is_saved_once_with_aggregated_readings():
    persistence = FakePersistence()
    monitor = ResourceMonitor(
        "job-1",
        interval=0,
        enabled=True,
        persistence=persistence,
    )
    monitor._resource_log_row = lambda **values: SimpleNamespace(**values)

    monitor._record_sample(
        cpu_percent=10.0,
        ram_usage=20.0,
        gpu_percent=30.0,
        gpu_ram_usage=100.0,
        gpu_power_usage=40.0,
        gpu_name="GPU A",
    )
    monitor._record_sample(
        cpu_percent=30.0,
        ram_usage=40.0,
        gpu_percent=50.0,
        gpu_ram_usage=300.0,
        gpu_power_usage=60.0,
        gpu_name="GPU A",
    )

    monitor.stop()
    monitor.stop()

    assert len(persistence.rows) == 1
    row = persistence.rows[0]
    assert row.job_id == "job-1"
    assert row.cpu_percent == 20.0
    assert row.ram_usage == 30.0
    assert row.gpu_percent == 40.0
    assert row.gpu_ram_usage == 200.0
    assert row.gpu_power_usage == 50.0
    assert row.gpu_name == "GPU A"
    assert fake_nvml.shutdown_called is True


def test_gpu_utilization_falls_back_when_process_samples_are_unavailable(monkeypatch):
    child_process = FakeProcess(pid=101)
    root_process = FakeProcess(pid=100, children=[child_process])
    fake_nvml = FakeNvmlWithoutProcessUtilization()

    monkeypatch.setattr(resource_monitor, "psutil", FakePsutil(root_process))
    monkeypatch.setattr(resource_monitor, "pynvml", fake_nvml)

    monitor = ResourceMonitor("job-1", interval=0, enabled=True, pid=100)

    gpu_usage, vram_usage, gpu_name, power_usage = monitor.logging_gpu()

    assert gpu_usage == [75.0]
    assert vram_usage == [256.0]
    assert gpu_name == "Fake GPU"
    assert power_usage == [50.0]
