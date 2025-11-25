import threading
from typing import Dict, Optional, Tuple

import numpy as np
import pyrealsense2 as rs

import logging_mp


logger_mp = logging_mp.get_logger(__name__)


class RealSenseStream:
    def __init__(
        self,
        serial: str,
        resolution: Tuple[int, int],
        fps: int,
        enable_depth: bool = False,
    ) -> None:
        self.serial = serial
        self.height, self.width = resolution
        self.fps = fps
        self.enable_depth = enable_depth

        self.color_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth_frame = (
            np.zeros((self.height, self.width), dtype=np.uint16) if self.enable_depth else None
        )

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._pipeline: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None

    def start(self) -> None:
        if self._running:
            return

        logger_mp.info(
            f"Starting RealSense stream serial={self.serial}, res=({self.height},{self.width}), depth={self.enable_depth}"
        )
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self._align = rs.align(rs.stream.color)

        profile = self._pipeline.start(config)
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name) if device else "Unknown"
        logger_mp.info(f"RealSense device started: {device_name} ({self.serial})")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as exc:  # noqa: BLE001
                logger_mp.warning(f"Failed to stop pipeline for {self.serial}: {exc}")

    def get_frames(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        with self._lock:
            color = self.color_frame.copy()
            depth = self.depth_frame.copy() if self.depth_frame is not None else None
        return color, depth

    def _capture_loop(self) -> None:
        assert self._pipeline is not None
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames()
                if self.enable_depth and self._align is not None:
                    frames = self._align.process(frames)

                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None
                else:
                    depth_image = None

                with self._lock:
                    self.color_frame = color_image
                    if self.enable_depth and depth_image is not None:
                        self.depth_frame = depth_image
            except Exception as exc:  # noqa: BLE001
                logger_mp.warning(f"RealSense stream {self.serial} error: {exc}")
                break


class RealSenseManager:
    def __init__(
        self,
        wrist_serials: Optional[Tuple[str, str]] = None,
        front_serial: Optional[str] = None,
        resolution: Tuple[int, int] = (480, 640),
        fps: int = 30,
    ) -> None:
        self.streams: Dict[str, RealSenseStream] = {}

        if wrist_serials and len(wrist_serials) == 2:
            left_serial, right_serial = wrist_serials
            self.streams["wrist_left"] = RealSenseStream(
                serial=left_serial, resolution=resolution, fps=fps, enable_depth=False
            )
            self.streams["wrist_right"] = RealSenseStream(
                serial=right_serial, resolution=resolution, fps=fps, enable_depth=False
            )
        if front_serial:
            self.streams["front"] = RealSenseStream(
                serial=front_serial, resolution=resolution, fps=fps, enable_depth=True
            )

    def start(self) -> None:
        for name, stream in self.streams.items():
            try:
                stream.start()
            except Exception as exc:  # noqa: BLE001
                logger_mp.warning(f"Failed to start RealSense stream {name}: {exc}")

    def stop(self) -> None:
        for stream in self.streams.values():
            stream.stop()

    def get_frames(self) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        frames: Dict[str, Dict[str, Optional[np.ndarray]]] = {}
        for name, stream in self.streams.items():
            color, depth = stream.get_frames()
            frames[name] = {"color": color, "depth": depth}
        return frames