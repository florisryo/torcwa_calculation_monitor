from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil
import torch

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover
    pynvml = None  # type: ignore


def _mb(value: float) -> float:
    return value / (1024.0 * 1024.0)


class ResourceMonitor:
    """Periodic sampler for CPU / memory / GPU metrics with lightweight event marks."""

    def __init__(
        self,
        interval_s: float = 0.5,
        enable_gpu: Optional[bool] = None,
        gpu_index: int = 0,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be positive")
        self.interval_s = float(interval_s)

        self._process = psutil.Process()
        self._process.cpu_percent(None)  # prime counters
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ts: Optional[float] = None
        self._records: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        self._sample_idx = 0
        self._lock = threading.Lock()

        if enable_gpu is None:
            enable_gpu = torch.cuda.is_available()
        self._gpu_enabled = bool(enable_gpu)
        self._gpu_index = gpu_index
        self._gpu_handle = None
        self._cuda_present = torch.cuda.is_available()

        if self._gpu_enabled and pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._gpu_enabled = False
                self._gpu_handle = None

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def gpu_enabled(self) -> bool:
        return self._gpu_enabled

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_evt.clear()
        self._start_ts = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="ResourceMonitor", daemon=True)
        self.mark("monitor_start")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self.mark("monitor_stop")
        self._stop_evt.set()
        self._thread.join()
        self._thread = None
        if self._gpu_handle is not None and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def mark(self, label: str, *, metadata: Optional[Dict[str, Any]] = None, sync_cuda: bool = False) -> None:
        if self._start_ts is None:
            raise RuntimeError("ResourceMonitor must be started before mark().")
        if sync_cuda and self._cuda_present:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_ts
        with self._lock:
            self._events.append({"time_s": elapsed, "label": label, "metadata": metadata or {}})

    def records(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def events(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def to_dataframe(self):  # type: ignore[override]
        if pd is None:
            raise ImportError("pandas is required to convert records into a DataFrame")
        return pd.DataFrame(self.records())

    def export(self, path: Path) -> None:
        df = self.to_dataframe()
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df.to_csv(path, index=False)
        elif suffix in {".parquet", ".pq"}:
            df.to_parquet(path, index=False)
        else:
            raise ValueError("Unsupported file suffix for export")

    def _run(self) -> None:
        next_tick = time.perf_counter()
        while not self._stop_evt.is_set():
            self._capture_sample()
            next_tick += self.interval_s
            remaining = next_tick - time.perf_counter()
            if remaining > 0:
                self._stop_evt.wait(remaining)

    def _capture_sample(self) -> None:
        if self._start_ts is None:
            return
        elapsed = time.perf_counter() - self._start_ts

        cpu_percent = psutil.cpu_percent(interval=None)
        proc_cpu_percent = self._process.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        proc_mem = self._process.memory_info()
        system_mem_used = vm.total - vm.available
        process_mem_percent = (float(proc_mem.rss) / vm.total * 100.0) if vm.total else None

        try:
            process_threads = self._process.num_threads()
        except psutil.Error:
            process_threads = None

        process_cpu_num = None
        if hasattr(self._process, "cpu_num"):
            try:
                process_cpu_num = self._process.cpu_num()
            except (psutil.Error, AttributeError):
                process_cpu_num = None

        sample: Dict[str, Any] = {
            "sample": self._sample_idx,
            "elapsed_s": elapsed,
            "cpu_percent": cpu_percent,
            "process_cpu_percent": proc_cpu_percent,
            "system_memory_percent": vm.percent,
            "system_memory_used_mb": _mb(system_mem_used),
            "process_memory_mb": _mb(proc_mem.rss),
            "process_memory_percent": process_mem_percent,
            "process_threads": process_threads,
            "process_cpu_num": process_cpu_num,
        }

        if hasattr(self._process, "cpu_affinity"):
            try:
                affinity = self._process.cpu_affinity()
            except (psutil.Error, NotImplementedError):
                affinity = None
        else:
            affinity = None
        sample["process_cpu_affinity"] = list(affinity) if affinity is not None else None

        if self._gpu_enabled and self._gpu_handle is not None and pynvml is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                total_mb = _mb(mem_info.total) if mem_info.total else None
                sample["gpu_util_percent"] = float(util.gpu)
                sample["gpu_memory_mb"] = _mb(mem_info.used)
                sample["gpu_memory_percent"] = (float(mem_info.used) / mem_info.total * 100.0) if mem_info.total else None
                sample["gpu_memory_total_mb"] = total_mb
            except Exception:
                sample["gpu_util_percent"] = None
                sample["gpu_memory_mb"] = None
                sample["gpu_memory_percent"] = None
                sample["gpu_memory_total_mb"] = None
        else:
            sample["gpu_util_percent"] = None
            sample["gpu_memory_mb"] = None
            sample["gpu_memory_percent"] = None
            sample["gpu_memory_total_mb"] = None

        if self._cuda_present:
            sample["cuda_allocated_mb"] = _mb(torch.cuda.memory_allocated())
            sample["cuda_reserved_mb"] = _mb(torch.cuda.memory_reserved())
        else:
            sample["cuda_allocated_mb"] = None
            sample["cuda_reserved_mb"] = None

        with self._lock:
            self._records.append(sample)
            self._sample_idx += 1