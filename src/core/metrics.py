from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import psutil
import torch


@dataclass
class SystemMetrics:
    """Snapshot of hardware utilization tailored for RTX 3050 4GB VRAM constraints."""
    cpu_percent: float
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    cuda_available: bool
    gpu_name: str = "N/A"
    vram_allocated_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    vram_peak_mb: float = 0.0
    vram_total_mb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "ram_used_mb": round(self.ram_used_mb, 1),
            "ram_total_mb": round(self.ram_total_mb, 1),
            "ram_percent": self.ram_percent,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "vram_allocated_mb": round(self.vram_allocated_mb, 1),
            "vram_reserved_mb": round(self.vram_reserved_mb, 1),
            "vram_peak_mb": round(self.vram_peak_mb, 1),
            "vram_total_mb": round(self.vram_total_mb, 1),
        }


def get_system_metrics() -> SystemMetrics:
    """Captures instantaneous CPU, RAM, and GPU/VRAM hardware metrics."""
    # CPU & RAM
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    ram_used = (mem.total - mem.available) / (1024 ** 2)
    ram_total = mem.total / (1024 ** 2)

    cuda_avail = torch.cuda.is_available()
    gpu_name = "N/A"
    vram_alloc = 0.0
    vram_res = 0.0
    vram_peak = 0.0
    vram_total = 0.0

    if cuda_avail:
        try:
            dev_idx = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(dev_idx)
            vram_alloc = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
            vram_res = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
            vram_peak = torch.cuda.max_memory_allocated(dev_idx) / (1024 ** 2)
            props = torch.cuda.get_device_properties(dev_idx)
            vram_total = props.total_memory / (1024 ** 2)
        except Exception:
            pass

    return SystemMetrics(
        cpu_percent=cpu,
        ram_used_mb=ram_used,
        ram_total_mb=ram_total,
        ram_percent=mem.percent,
        cuda_available=cuda_avail,
        gpu_name=gpu_name,
        vram_allocated_mb=vram_alloc,
        vram_reserved_mb=vram_res,
        vram_peak_mb=vram_peak,
        vram_total_mb=vram_total,
    )


class PipelinePerformanceTracker:
    """
    Lightweight telemetry tracker measuring processing FPS, inference latency,
    and dropped frames without overhead.
    """

    def __init__(self, camera_id: str = "CAM_01"):
        self.camera_id = camera_id
        self.start_time = time.time()
        self.total_frames: int = 0
        self.total_latency_ms: float = 0.0
        self.min_latency_ms: float = float("inf")
        self.max_latency_ms: float = 0.0
        self.dropped_frames: int = 0
        self._last_frame_time: Optional[float] = None
        self._fps_window: list[float] = []

    def record_frame(self, latency_ms: float = 0.0) -> None:
        """Records a successfully processed frame and its inference/tracking latency."""
        now = time.time()
        self.total_frames += 1
        self.total_latency_ms += latency_ms
        if latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms

        if self._last_frame_time is not None:
            delta = now - self._last_frame_time
            if delta > 0:
                self._fps_window.append(1.0 / delta)
                if len(self._fps_window) > 30:
                    self._fps_window.pop(0)

        self._last_frame_time = now

    def record_drop(self, count: int = 1) -> None:
        """Records dropped or stale frames."""
        self.dropped_frames += count

    @property
    def fps(self) -> float:
        """Calculates current rolling or overall average processing FPS."""
        if self._fps_window:
            return sum(self._fps_window) / len(self._fps_window)
        elapsed = time.time() - self.start_time
        return self.total_frames / elapsed if elapsed > 0 else 0.0

    @property
    def average_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_frames if self.total_frames > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        overall_fps = self.total_frames / elapsed if elapsed > 0 else 0.0
        sys_metrics = get_system_metrics()

        return {
            "camera_id": self.camera_id,
            "elapsed_sec": round(elapsed, 2),
            "processed_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "overall_fps": round(overall_fps, 2),
            "rolling_fps": round(self.fps, 2),
            "avg_latency_ms": round(self.average_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float("inf") else 0.0,
            "max_latency_ms": round(self.max_latency_ms, 2),
            "system": sys_metrics.to_dict(),
        }

    def format_summary(self) -> str:
        s = self.summary()
        sys_m = s["system"]
        lines = [
            f"--- Performance Report [{s['camera_id']}] ---",
            f" Frames Processed : {s['processed_frames']}",
            f" Frames Dropped   : {s['dropped_frames']}",
            f" Processing Speed : {s['overall_fps']} FPS (rolling: {s['rolling_fps']} FPS)",
            f" Average Latency  : {s['avg_latency_ms']} ms (min: {s['min_latency_ms']} ms, max: {s['max_latency_ms']} ms)",
            f" Elapsed Time     : {s['elapsed_sec']}s",
            f" Host Memory      : {sys_m['ram_used_mb']:.1f} MB / {sys_m['ram_total_mb']:.1f} MB ({sys_m['ram_percent']}%)",
            f" CPU Utilization  : {sys_m['cpu_percent']}%",
        ]
        if sys_m["cuda_available"]:
            lines.extend([
                f" GPU Device       : {sys_m['gpu_name']}",
                f" VRAM Allocated   : {sys_m['vram_allocated_mb']:.1f} MB / {sys_m['vram_total_mb']:.1f} MB",
                f" VRAM Peak        : {sys_m['vram_peak_mb']:.1f} MB",
            ])
        else:
            lines.append(" GPU Acceleration : CPU Mode (CUDA unavailable)")
        lines.append("-" * 45)
        return "\n".join(lines)
