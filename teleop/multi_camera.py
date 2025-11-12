"""
3× Intel RealSense demo with EXACT 4 views:
  - front_rgb, front_depth, wrist_left, wrist_right
Front = RGB+Depth, Wrists = RGB-only
- Each camera pinned by serial (stable multi-device)
- Multiprocessing + SharedMemory (zero-copy)
- Rerun Viewer spawn->connect fallback
- Layout auto-applied (new/old SDK), else instruct manual pin
- BGR -> RGB conversion for correct colors
- Synthetic fallback if device missing

Install:
  pip install rerun-sdk opencv-python pyrealsense2
Run:
  python (filename).py
"""

# depth 색상 고정 범위 (미터 단위 가정: D435/D455 depth_scale ~= 0.001)
DEPTH_MIN_M = 0.25
DEPTH_MAX_M = 2.00
DEPTH_SCALE  = 0.001

import time
import cv2
import numpy as np
import rerun as rr
import multiprocessing as mp
from multiprocessing import shared_memory
import pyrealsense2 as rs

# ========================= Rerun helpers =========================

FOUR_STREAMS = [
    "/cams/front_rgb",
    "/cams/front_depth",
    "/cams/wrist_left",
    "/cams/wrist_right",
]

def rr_init_or_connect(app_name: str = "multi-cam-demo"):
    # Spawn → connect fallback
    try:
        rr.init(app_name, spawn=True)
        print("[RERUN] Viewer spawned successfully.")
    except Exception as e:
        print(f"[RERUN] spawn failed: {e}")
        print("[RERUN] Falling back to connect() mode. Start viewer with `rerun` if needed.")
        rr.init(app_name, connect=True)

def rr_apply_layout_four():
    """
    Try multiple layout APIs so that ONLY the 4 requested views appear.
    Order: wrist_left, wrist_right, front_rgb, front_depth (2x2 grid)
    """
    # Newer SDK (>=0.23): Viewport(layout="grid")
    if hasattr(rr, "send_blueprint") and hasattr(rr, "Viewport"):
        try:
            rr.send_blueprint(
                rr.Viewport(
                    contents=[
                        rr.ImageView("/cams/wrist_left",  name="wrist_left"),
                        rr.ImageView("/cams/wrist_right", name="wrist_right"),
                        rr.ImageView("/cams/front_rgb",   name="front_rgb"),
                        rr.DepthImageView("/cams/front_depth", name="front_depth"),
                    ],
                    layout="grid",
                )
            )
            print("[RERUN] Applied layout via Viewport(grid) with the 4 views.")
            return
        except Exception as e:
            print(f"[RERUN] Viewport layout failed: {e}")

    # Older SDK: Blueprint + Horizontal/Vertical
    if hasattr(rr, "send_blueprint") and hasattr(rr, "Blueprint") and hasattr(rr, "Horizontal") and hasattr(rr, "Vertical"):
        try:
            rr.send_blueprint(
                rr.Blueprint(
                    rr.Horizontal(
                        rr.Vertical(
                            rr.ImageView("/cams/wrist_left",  name="wrist_left"),
                            rr.ImageView("/cams/wrist_right", name="wrist_right"),
                        ),
                        rr.Vertical(
                            rr.ImageView("/cams/front_rgb", name="front_rgb"),
                            rr.DepthImageView("/cams/front_depth", name="front_depth"),
                        ),
                    )
                )
            )
            print("[RERUN] Applied layout via Blueprint/Horizontal/Vertical with the 4 views.")
            return
        except Exception as e:
            print(f"[RERUN] Blueprint layout failed: {e}")

    print("[RERUN] Layout helper not available; pin these 4 streams manually (drag to views):")
    for s in FOUR_STREAMS:
        print(f"  - {s}")

def rr_set_time(seconds: float):
    if hasattr(rr, "set_time"):
        try:
            rr.set_time("demo_time", seconds)
            return
        except TypeError:
            pass
    # fallback
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("demo_time", seconds)

def rr_log_rgb(path: str, img_rgb: np.ndarray):
    if hasattr(rr, "Image"):
        rr.log(path, rr.Image(img_rgb))
    elif hasattr(rr, "log_image"):
        rr.log_image(path, img_rgb)
    else:
        print("[RERUN] No compatible image logging API found.")

def rr_log_depth(path: str, depth_u16: np.ndarray, meter: float = 0.001):
    if hasattr(rr, "DepthImage"):
        rr.log(path, rr.DepthImage(depth_u16, meter=meter))
    elif hasattr(rr, "log_depth_image"):
        rr.log_depth_image(path, depth_u16, meter=meter)
    else:
        print("[RERUN] No compatible depth logging API found.")

# ========================= Utils (synthetic fallback) =========================

def synthetic_colorbars_rgb(w, h, label: str, t):
    img = np.zeros((h, w, 3), np.uint8)
    cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for i, c in enumerate(cols):
        x0, x1 = i * w // 4, (i + 1) * w // 4 - 1
        cv2.rectangle(img, (x0, 0), (x1, h - 1), c, -1)
    txt = f"{label} - NO CAMERA [{t:.1f}s]"
    cv2.putText(img, txt, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def synthetic_depth(w, h, t):
    x = np.tile(np.linspace(0, 65535, w, dtype=np.uint16), (h, 1))
    return (x + int((np.sin(t) * 0.5 + 0.5) * 5000)).astype(np.uint16)

# ========================= Camera spec & worker =========================

class RSSpec:
    def __init__(self, name, serial, width=640, height=480, fps=30, need_depth=False):
        self.name = name              # "wrist_left", "wrist_right", "front"
        self.serial = serial          # serial string, or "SYNTH*"
        self.width = width
        self.height = height
        self.fps = fps
        self.need_depth = need_depth  # True only for front

def _rs_worker(spec: RSSpec, shm_rgb_name, shm_depth_name, flag):
    """RealSense worker pinned to a specific serial. Depth only if need_depth=True."""
    shm_c = shared_memory.SharedMemory(name=shm_rgb_name)
    color_rgb = np.ndarray((spec.height, spec.width, 3), np.uint8, buffer=shm_c.buf)

    depth = None
    if spec.need_depth and shm_depth_name:
        shm_d = shared_memory.SharedMemory(name=shm_depth_name)
        depth = np.ndarray((spec.height, spec.width), np.uint16, buffer=shm_d.buf)

    have_device = not spec.serial.startswith("SYNTH")
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
            print(f"[{spec.name}] Failed to start RS serial={spec.serial}: {e}")
            have_device = False
    else:
        print(f"[{spec.name}] SYNTH source (no real device)")

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
                    print(f"[{spec.name}] RS FPS ~ {cnt / (now - t0):.1f}")
                    t0, cnt = now, 0
                flag.set()
            except Exception as e:
                print(f"[{spec.name}] RS stream error (serial={spec.serial}): {e}")
                have_device = False
        else:
            color_rgb[:] = synthetic_colorbars_rgb(spec.width, spec.height, spec.name, now - t0)
            if spec.need_depth and depth is not None:
                depth[:] = synthetic_depth(spec.width, spec.height, now - t0)
            flag.set()
            if now - t0 >= 1.0:
                print(f"[{spec.name}] synthetic streaming (depth={spec.need_depth})")
                t0 = now
        time.sleep(0.001)

# ========================= Manager =========================

class RSManager:
    def __init__(self, specs):
        self.specs = specs
        self.shms = {}  
        self.flags = {}
        self.procs = {}

        for s in specs:
            shm_c = shared_memory.SharedMemory(create=True, size=s.width * s.height * 3)
            if s.need_depth:
                shm_d = shared_memory.SharedMemory(create=True, size=s.width * s.height * 2)
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

    def ready(self, name):
        return self.flags[name].is_set()

    def get_rgb(self, name):
        s = next(sp for sp in self.specs if sp.name == name)
        shm_c, _ = self.shms[name]
        return np.ndarray((s.height, s.width, 3), np.uint8, buffer=shm_c.buf)

    def get_depth(self, name):
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

# ========================= Device enumeration =========================

def list_realsense_serials():
    ctx = rs.context()
    devs = ctx.query_devices()
    serials = []
    for d in devs:
        try:
            serials.append(d.get_info(rs.camera_info.serial_number))
        except Exception:
            pass
    return serials

# ========================= Main =========================

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    serials = list_realsense_serials()
    print(f"[MAIN] Found RealSense devices (serials): {serials}")

    # 원하는 3개를 직접 지정하려면 아래와 같이
    # serials = ["<LEFT_SERIAL>", "<RIGHT_SERIAL>", "<FRONT_SERIAL>"]

    if len(serials) == 0:
        serials = ["SYNTH0", "SYNTH1", "SYNTH2"]
    elif len(serials) == 1:
        serials = [serials[0], "SYNTH1", "SYNTH2"]
    elif len(serials) == 2:
        serials = [serials[0], serials[1], "SYNTH2"]
    else:
        serials = serials[:3]

    # name 매핑: wrist_left, wrist_right, front (front만 depth)
    specs = [
        RSSpec("wrist_left",  serials[0], width=640, height=480, fps=30, need_depth=False),
        RSSpec("wrist_right", serials[1], width=640, height=480, fps=30, need_depth=False),
        RSSpec("front",       serials[2], width=640, height=480, fps=30, need_depth=True),
    ]

    mgr = RSManager(specs)
    rr_init_or_connect("multi-cam-demo")
    rr_apply_layout_four()
    print("Starting main capture loop... (Ctrl+C to exit)")

    t0 = time.time()
    try:
        while True:
            rr_set_time(time.time() - t0)

            # wrist RGB-only
            if mgr.ready("wrist_left"):
                rr_log_rgb("/cams/wrist_left", mgr.get_rgb("wrist_left"))

            if mgr.ready("wrist_right"):
                rr_log_rgb("/cams/wrist_right", mgr.get_rgb("wrist_right"))

            # front RGB + Depth
            if mgr.ready("front"):
                rr_log_rgb("/cams/front_rgb", mgr.get_rgb("front"))
                d = mgr.get_depth("front")
                if d is not None:
                    # rr_log_depth("/cams/front_depth", d, meter=0.001)
                    dm = d.astype(np.float32) * DEPTH_SCALE
                    dm = np.clip(dm, DEPTH_MIN_M, DEPTH_MAX_M)
                    norm = ((dm - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M) * 255.0).astype(np.uint8)
                    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)      # BGR
                    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)     # RGB 변환
                    rr_log_rgb("/cams/front_depth", colored_rgb)

            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n[MAIN] Interrupted by user. Cleaning up...")
    finally:
        mgr.close()
        print("[MAIN] Bye.")

if __name__ == "__main__":
    main()
