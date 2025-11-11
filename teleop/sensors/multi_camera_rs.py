# teleop/sensors/multi_camera_rs.py
import time
import cv2
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from dataclasses import dataclass
try:
    import pyrealsense2 as rs
except ImportError:
    rs = None  # 개발/테스트용(리얼센스 미연결 시)

@dataclass
class RSSpec:
    name: str           # "wrist_left" | "wrist_right" | "front"
    serial: str         # RealSense device serial, or "SYNTH*"
    width: int = 640
    height: int = 480
    fps: int = 30
    need_depth: bool = False

def _synthetic_colorbars_rgb(w, h, label, t):
    img = np.zeros((h, w, 3), np.uint8)
    cols = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
    for i, c in enumerate(cols):
        x0, x1 = i*w//4, (i+1)*w//4 - 1
        cv2.rectangle(img, (x0,0), (x1,h-1), c, -1)
    cv2.putText(img, f"{label} NO-CAM {t:.1f}s", (10,h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _synthetic_depth(w, h, t):
    x = np.tile(np.linspace(0,65535,w, dtype=np.uint16), (h,1))
    return (x + int((np.sin(t)*0.5+0.5)*5000)).astype(np.uint16)

def _rs_worker(spec: RSSpec, shm_rgb_name: str, shm_depth_name: str, flag: mp.Event):
    shm_c = shared_memory.SharedMemory(name=shm_rgb_name)
    color_rgb = np.ndarray((spec.height, spec.width, 3), np.uint8, buffer=shm_c.buf)

    depth = None
    if spec.need_depth and shm_depth_name:
        shm_d = shared_memory.SharedMemory(name=shm_depth_name)
        depth = np.ndarray((spec.height, spec.width), np.uint16, buffer=shm_d.buf)

    have_device = (rs is not None) and (not spec.serial.startswith("SYNTH"))
    pipe = None; align = None

    if have_device:
        try:
            pipe = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(spec.serial)
            cfg.enable_stream(rs.stream.color, spec.width, spec.height, rs.format.bgr8, spec.fps)
            if spec.need_depth:
                cfg.enable_stream(rs.stream.depth, spec.width, spec.height, rs.format.z16, spec.fps)
                align = rs.align(rs.stream.color)
            pipe.start(cfg)
            print(f"[{spec.name}] RS started (serial={spec.serial}, depth={spec.need_depth})")
        except Exception as e:
            print(f"[{spec.name}] RS start failed: {e}")
            have_device = False
    else:
        print(f"[{spec.name}] SYNTH or RS missing")

    t0, cnt = time.time(), 0
    while True:
        now = time.time()
        if have_device and pipe is not None:
            try:
                frames = pipe.wait_for_frames()
                if spec.need_depth and align is not None:
                    frames = align.process(frames)
                c = frames.get_color_frame()
                if not c:
                    raise RuntimeError("No color frame")
                c_bgr = np.asanyarray(c.get_data())
                c_rgb = cv2.cvtColor(c_bgr, cv2.COLOR_BGR2RGB)
                color_rgb[:] = c_rgb
                if spec.need_depth and depth is not None:
                    d = frames.get_depth_frame()
                    if not d:
                        raise RuntimeError("No depth frame")
                    depth[:] = np.asanyarray(d.get_data())
                cnt += 1
                if now - t0 >= 1.0:
                    print(f"[{spec.name}] FPS ~ {cnt/(now-t0):.1f}")
                    t0, cnt = now, 0
                flag.set()
            except Exception as e:
                print(f"[{spec.name}] RS stream err: {e}")
                have_device = False
        else:
            color_rgb[:] = _synthetic_colorbars_rgb(spec.width, spec.height, spec.name, now - t0)
            if spec.need_depth and depth is not None:
                depth[:] = _synthetic_depth(spec.width, spec.height, now - t0)
            flag.set()
            if now - t0 >= 1.0:
                print(f"[{spec.name}] synthetic streaming (depth={spec.need_depth})")
                t0 = now
        time.sleep(0.001)

class MultiCameraRS:
    """3대 RealSense 관리 (front: RGB+Depth, wrists: RGB-only)"""
    def __init__(self, specs: list[RSSpec]):
        self.specs = specs
        self.shms = {}   # name -> (shm_c, shm_d or None)
        self.flags = {}
        self.procs = {}
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        for s in specs:
            shm_c = shared_memory.SharedMemory(create=True, size=s.width*s.height*3)
            if s.need_depth:
                shm_d = shared_memory.SharedMemory(create=True, size=s.width*s.height*2)
                flag = mp.Event()
                p = mp.Process(target=_rs_worker, args=(s, shm_c.name, shm_d.name, flag), daemon=True)
                p.start()
                self.shms[s.name] = (shm_c, shm_d)
            else:
                flag = mp.Event()
                p = mp.Process(target=_rs_worker, args=(s, shm_c.name, "", flag), daemon=True)
                p.start()
                self.shms[s.name] = (shm_c, None)
            self.flags[s.name] = flag
            self.procs[s.name] = p
            print(f"Started RS process: {s.name} (serial={s.serial}, depth={s.need_depth})")

    def ready(self, name: str) -> bool:
        return self.flags[name].is_set()

    def get_rgb(self, name: str) -> np.ndarray:
        s = next(sp for sp in self.specs if sp.name == name)
        shm_c, _ = self.shms[name]
        return np.ndarray((s.height, s.width, 3), np.uint8, buffer=shm_c.buf)

    def get_depth(self, name: str) -> np.ndarray | None:
        s = next(sp for sp in self.specs if sp.name == name)
        _, shm_d = self.shms[name]
        if shm_d is None:
            return None
        return np.ndarray((s.height, s.width), np.uint16, buffer=shm_d.buf)

    def close(self):
        for name, (a, b) in self.shms.items():
            try: a.close(); a.unlink()
            except Exception: pass
            if b is not None:
                try: b.close(); b.unlink()
                except Exception: pass

def list_serials():
    if rs is None:
        return []
    ctx = rs.context()
    serials = []
    for dev in ctx.query_devices():
        try:
            serials.append(dev.get_info(rs.camera_info.serial_number))
        except Exception:
            pass
    return serials
