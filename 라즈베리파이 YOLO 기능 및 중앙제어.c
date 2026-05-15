#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pathlib
from pathlib import Path
from dataclasses import dataclass
import time
import faulthandler

# ===== (0) 안전모드 =====
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# ===== (토치 세이프 모드: segfault 회피) =====
os.environ.setdefault("PYTORCH_DISABLE_XNNPACK", "1")
os.environ.setdefault("ATEN_CPU_CAPABILITY", "default")
os.environ.setdefault("MALLOC_ARENA_MAX", "1")

import cv2
import numpy as np

faulthandler.enable()
try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

# ===== (직렬 통신) =====
try:
    import serial
except Exception:
    serial = None

SERIAL_PORT_CANDIDATES = ("/dev/ttyAMA0","/dev/ttyS0","/dev/ttyACM0","/dev/ttyUSB0")
SERIAL_BAUD = 38400
SERIAL_TIMEOUT = 0.05

STM32_PORT_CANDIDATES = ("/dev/ttyAMA4", "/dev/ttyS4")

def _open_serial_from_candidates(cands, baud, timeout=SERIAL_TIMEOUT, label="SERIAL"):
    if serial is None:
        print("[WARN] pyserial 미설치 (pip install pyserial)")
        return None
    for dev in cands:
        try:
            ser = serial.Serial(dev, baud, timeout=timeout)
            time.sleep(0.2)
            if ser.is_open:
                print(f"[{label}] Connected: {dev} @ {baud}")
                return ser
        except Exception:
            continue
    print(f"[WARN] {label} 포트를 찾지 못했습니다. 후보={cands}")
    return None

def open_arduino_serial():
    return _open_serial_from_candidates(SERIAL_PORT_CANDIDATES, SERIAL_BAUD, label="ARDUINO")

def open_stm32_serial():
    return _open_serial_from_candidates(STM32_PORT_CANDIDATES, SERIAL_BAUD, label="STM32")

def send_xyz_to_arduino(ser, x_mm, y_mm, z_mm):
    if ser is None: return
    try:
        x_out = int(round(x_mm / 10.0))
        y_out = int(round(y_mm / 10.0))
        z_out = int(round(z_mm / 10.0))
        line = f"x={x_out} y={y_out} z={z_out}\n"
        ser.write(line.encode("ascii", errors="ignore"))
    except Exception as e:
        print(f"[WARN] serial write failed: {e}")

def send_stop_to_stm32(ser_stm):
    if ser_stm is None: return False
    try:
        ser_stm.write(b"stop\n")
        return True
    except Exception as e:
        print(f"[WARN] STM32 write failed: {e}")
        return False

def send_g_to_stm32(ser_stm):
    if ser_stm is None: return False
    try:
        ser_stm.write(b"g\n")
        return True
    except Exception as e:
        print(f"[WARN] STM32 write failed: {e}")
        return False

def try_read_stm32_line(ser_stm):
    if ser_stm is None: return ""
    try:
        line = ser_stm.readline()
        if not line:
            return ""
        try:
            s = line.decode("utf-8", errors="ignore").strip()
        except Exception:
            s = str(line).strip()
        return s
    except Exception:
        return ""

# ===== (1) 호환 & PyTorch 패치 =====
import torch
try:
    pathlib.WindowsPath = pathlib.PosixPath
except Exception:
    pass

_ORIG_TORCH_LOAD = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIG_TORCH_LOAD(*args, **kwargs)
torch.load = _patched_torch_load

from torch.serialization import add_safe_globals
from torch.nn.modules.container import Sequential
add_safe_globals([Sequential])

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

from openni import openni2

# ===== (1-1) 클래스 이름 표준화 =====
CLASS_ALIASES = {'tomato': 'red', 'orenge': 'orange'}  # 철자 보정
def normalize_class_name(name: str) -> str:
    n = (name or '').strip().lower()
    return CLASS_ALIASES.get(n, n)

# ===== (1-2) 전역 색상 =====
BOX_COLOR = (0, 0, 255)  # RED (BGR)

# ===== (2) 설정 =====
@dataclass
class Config:
    HINTS: tuple = (
        "/home/pi/Downloads/OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d/Redist",
        "/home/pi/Downloads/OpenNI_2.3.0.86_202210111155_4c8f5aa4_beta6_a311d/tools/NiViewer",
        "/home/pi/Downloads/orbbec_sdk/Redist",
    )
    YOLOV5_DIR: str = "/home/pi/Desktop/yolo_project/yolov5"
    WEIGHTS: str = "/home/pi/Desktop/yolo_project/yolov5/runs/train/tomato_model_v59/weights/best.pt"

    COLOR_IDX_CANDIDATES: tuple = (0, 1, 2, 3)
    RGB_W: int = 640
    RGB_H: int = 480
    RGB_FPS: int = 30
    MIRROR_RGB: bool = False
    DEPTH_MAX_MM: int = 3000

    IMG_SIZE: int = 320
    CONF_THRES: float = 0.45
    IOU_THRES: float = 0.45

    # 오직 red 클래스만 허용
    ALLOWED_CLASS_NAMES: frozenset = frozenset({'red'})

    # 검출 허용치(완화값 유지)
    MIN_BOX_AREA: int = 1200
    MIN_AR: float = 0.40
    MAX_AR: float = 2.00

    MIN_COLOR_RATIO_RED: float = 0.10
    PRINT_EVERY_N: int = 1
    HIGH_CONF_BYPASS_THRES: float = 0.60

    # === 주황색 완전 차단 게이트 ===
    ORANGE_VETO_ENABLE: bool = True
    ORANGE_VETO_RATIO_MIN: float = 0.20

    # stop 전송 방식
    ALWAYS_SEND_STOP_ON_DET: bool = False
    STOP_COOLDOWN_FRAMES: int = 30

    # 깊이 보정/정렬/샘플링/홀드
    DEPTH_SCALE_A: float = 1.00
    DEPTH_BIAS_B:  int   = 0
    PIXEL_OFFSET_X: int = 0
    PIXEL_OFFSET_Y: int = 0
    DEPTH_RADIUS_MIN: int = 3
    DEPTH_RADIUS_MAX: int = 9
    HOLD_VALID_FRAMES: int = 10

    # 중앙 게이트
    CENTER_GATE_ENABLE: bool = True
    CENTER_GATE_THIRDS: int = 3

    # 얼굴 근처 필터(느슨)
    FACE_GATE_ENABLE: bool = True
    FACE_IOU_THRESH: float = 0.45
    FACE_DET_EVERY_N: int = 5
    FACE_MIN_SIZE: int = 60

    # 피부/형상/크기/지속성
    SKIN_GATE_ENABLE: bool = True
    SKIN_RATIO_MAX: float = 0.55
    ROUNDNESS_GATE_ENABLE: bool = True
    ROUNDNESS_MIN: float = 0.35

    SIZE_GATE_ENABLE: bool = True
    TOMATO_DIAM_MM_MIN: int = 18
    TOMATO_DIAM_MM_MAX: int = 60

    PERSIST_GATE_ENABLE: bool = True
    PERSIST_MIN_FRAMES: int = 2

    # === (추가) 중앙 (0,0) 점 표시 ===
    DRAW_ORIGIN_DOT: bool = True
    ORIGIN_DOT_RADIUS: int = 4
    ORIGIN_DOT_COLOR: tuple = (255, 255, 255)  # 흰색

cfg = Config()

# ===== (2-1) OpenNI Redist 탐색 =====
def _has_openni2_layout(p: Path) -> bool:
    return (p / "OpenNI2").exists() and (p / "OpenNI2" / "Drivers").exists()

def _iter_depth_limited(root: Path, max_depth: int = 3):
    try:
        stack = [(root, 0)]
        while stack:
            cur, d = stack.pop()
            yield cur
            if d < max_depth:
                for child in cur.iterdir():
                    if child.is_dir():
                        stack.append((child, d + 1))
    except Exception:
        return

def resolve_openni_redist():
    tried = []
    try:
        openni2.initialize()
        print("[OPENNI] initialize() without path -> OK")
        return "", "system"
    except Exception:
        pass
    env = os.environ.get("OPENNI2_REDIST")
    if env:
        p = Path(env); tried.append(str(p))
        if p.exists() and (p.name.lower() == "redist" or _has_openni2_layout(p)):
            return str(p), "redist"
    for h in cfg.HINTS:
        p = Path(h); tried.append(str(p))
        if not p.exists(): continue
        if p.name.lower() == "niviewer":
            cand = p.parent.parent / "Redist"
            tried.append(str(cand))
            if cand.exists(): return str(cand), "redist"
            continue
        if p.name.lower() == "redist":
            return str(p), "redist"
        if _has_openni2_layout(p):
            return str(p), "root"
    for root in ("/usr/lib/arm-linux-gnueabihf", "/usr/lib/aarch64-linux-gnu",
                 "/usr/local/lib", "/usr/lib", "/opt/OpenNI2", "/opt/ni"):
        p = Path(root); tried.append(str(p))
        if p.exists():
            if _has_openni2_layout(p): return str(p), "root"
            for sub in ("OpenNI2","openni2","OpenNI","orbbec","astra"):
                sp = p / sub; tried.append(str(sp))
                if _has_openni2_layout(sp): return str(p), "root"
    home = Path.home()
    for root in (home / "Downloads", home):
        if not root.exists(): continue
        for m in _iter_depth_limited(root, max_depth=3):
            if m.name.lower() == "redist": return str(m), "redist"
            if _has_openni2_layout(m): return str(m), "root"
    print("[OPENNI] Tried these paths for Redist (none worked):")
    for t in tried: print(" -", t)
    return None, "missing"

print("[CHECK] Resolving OpenNI Redist...")
REDIST_PATH, REDIST_KIND = resolve_openni_redist()
kind_msg = "OK (already initialized)" if REDIST_PATH == "" else ("OK: " + REDIST_PATH) if REDIST_PATH else "MISSING"
print(f"[CHECK] OpenNI Redist -> {kind_msg}")

for p_str in [cfg.YOLOV5_DIR, cfg.WEIGHTS]:
    p = Path(p_str)
    print(f"[CHECK] {p} -> {'OK' if p.exists() else 'MISSING'}")
YOLOV5_DIR_RESOLVED = str(Path(cfg.YOLOV5_DIR).resolve())
if YOLOV5_DIR_RESOLVED not in sys.path:
    sys.path.insert(0, YOLOV5_DIR_RESOLVED)

# ===== (3) YOLO 유틸 =====
from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator
from utils.augmentations import letterbox

# ===== (4) 카메라/깊이 유틸 =====
def open_color_camera(indices, width, height, fps):
    for idx in indices:
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release(); continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and frame is not None and frame.size and frame.shape[0] > 0 and frame.shape[1] > 0:
            print(f"[OK] Color via V4L2: /dev/video{idx} ({frame.shape[1]}x{frame.shape[0]})")
            return cap
        cap.release()
    print("[ERR] No valid V4L2 color camera found.")
    return None

def pick_depth_mode(device):
    try:
        sinfo = device.get_sensor_info(openni2.SENSOR_DEPTH)
        modes = sinfo.videoModes
        fmt_map = {
            openni2.PIXEL_FORMAT_DEPTH_1_MM: "PIXEL_FORMAT_DEPTH_1_MM",
            openni2.PIXEL_FORMAT_DEPTH_100_UM: "PIXEL_FORMAT_DEPTH_100_UM",
        }
        preferred = (
            (640, 480, "PIXEL_FORMAT_DEPTH_1_MM",    30),
            (640, 480, "PIXEL_FORMAT_DEPTH_100_UM", 30),
            (640, 400, "PIXEL_FORMAT_DEPTH_1_MM",    30),
            (640, 400, "PIXEL_FORMAT_DEPTH_100_UM", 30),
        )
        for (w, h, fmt_name, fps) in preferred:
            for vm in modes:
                if (vm.resolutionX, vm.resolutionY, vm.fps) == (w, h, fps) and fmt_map.get(vm.pixelFormat, "") == fmt_name:
                    return vm
    except Exception:
        pass
    return None

def depth_frame_to_numpy_mm(frame):
    if frame is None: return None
    buf = frame.get_buffer_as_uint16()
    if buf is None: return None
    arr = np.frombuffer(buf, dtype=np.uint16)
    h, w = frame.height, frame.width
    if h <= 0 or w <= 0 or arr.size != h * w: return None
    return arr.reshape(h, w)

def depth_mm_to_colormap(depth_mm, vmax=3000):
    if depth_mm is None or depth_mm.size == 0: return None
    vis = np.clip(depth_mm.astype(np.float32), 0, float(vmax))
    vis = (vis / float(vmax) * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

# ===== (5) 색상/오검출 저감 유틸 =====
def red_color_ratio(bgr_roi):
    if bgr_roi is None or bgr_roi.size == 0: return 0.0
    if bgr_roi.shape[0] <= 1 or bgr_roi.shape[1] <= 1: return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    mask = (
        cv2.inRange(hsv, np.array([0, 80, 60],  np.uint8), np.array([10, 255, 255], np.uint8)) |
        cv2.inRange(hsv, np.array([170, 80, 60], np.uint8), np.array([180, 255, 255], np.uint8))
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    sz = float(mask.size) if mask is not None else 1.0
    return float(np.count_nonzero(mask)) / sz

def orange_ratio_hsv(bgr_roi):
    if bgr_roi is None or bgr_roi.size == 0: return 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([10,  80, 60], np.uint8), np.array([20, 255, 255], np.uint8))
    m2 = cv2.inRange(hsv, np.array([20,  80, 60], np.uint8), np.array([25, 255, 255], np.uint8))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return float(np.count_nonzero(mask)) / float(mask.size)

def skin_ratio_ycrcb(bgr_roi):
    if bgr_roi is None or bgr_roi.size == 0: return 0.0
    ycrcb = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(
        ycrcb,
        np.array([  0, 133,  77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8)
    )
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return float(np.count_nonzero(skin)) / float(skin.size)

def red_mask(bgr_roi):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 80, 60],  np.uint8), np.array([10, 255, 255], np.uint8))
    m2 = cv2.inRange(hsv, np.array([170, 80, 60], np.uint8), np.array([180, 255, 255], np.uint8))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=1)
    return mask

def largest_contour_circularity(mask):
    if mask is None or mask.size == 0: return 0.0
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    if P <= 0 or A <= 1: return 0.0
    return float(4.0 * np.pi * A / (P * P))

def bbox_px_to_mm(w_px, h_px, Z_mm, intrinsics):
    if intrinsics is None or Z_mm is None or Z_mm <= 0: return None, None, None
    fx = intrinsics['fx']; fy = intrinsics['fy']
    if fx <= 0 or fy <= 0: return None, None, None
    width_mm  = (w_px * float(Z_mm)) / float(fx)
    height_mm = (h_px * float(Z_mm)) / float(fy)
    diam_mm   = 0.5 * (width_mm + height_mm)
    return width_mm, height_mm, diam_mm

# ===== (6) OpenNI 초기화 =====
def initialize_openni():
    if REDIST_PATH is None:
        raise RuntimeError("OpenNI2 Redist 경로를 찾을 수 없습니다.")
    if REDIST_PATH != "":
        openni2.initialize(REDIST_PATH)
    dev = openni2.Device.open_any()
    try:
        dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
        print("[OK] Image registration: DEPTH_TO_COLOR")
    except Exception as e:
        print(f"[WARN] Registration not supported (3D 좌표가 부정확할 수 있음): {e}")
    if not dev.has_sensor(openni2.SENSOR_DEPTH):
        raise RuntimeError("No depth sensor via OpenNI")
    depth_stream = openni2.VideoStream(dev, openni2.SENSOR_DEPTH)
    try:
        depth_stream.set_mirroring_enabled(False)
    except Exception:
        pass
    vm = pick_depth_mode(dev)
    if vm:
        try:
            depth_stream.set_video_mode(vm)
        except Exception as e:
            print(f"[WARN] set_video_mode failed, default used: {e}")
    depth_stream.start()
    print("[OK] Depth started")

    intrinsics = None
    try:
        intrinsics_raw = depth_stream.get_intrinsics()
        if isinstance(intrinsics_raw, (tuple, list)) and len(intrinsics_raw) >= 4:
            intrinsics = {'fx': float(intrinsics_raw[0]), 'fy': float(intrinsics_raw[1]),
                          'cx': float(intrinsics_raw[2]), 'cy': float(intrinsics_raw[3])}
        elif hasattr(intrinsics_raw, 'fx'):
            intrinsics = {'fx': float(intrinsics_raw.fx), 'fy': float(intrinsics_raw.fy),
                          'cx': float(intrinsics_raw.cx), 'cy': float(intrinsics_raw.cy)}
        else:
            raise RuntimeError("Unknown intrinsics format")
        print(f"[OK] Intrinsics loaded: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}, cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
    except Exception as e:
        print(f"[ERR] Failed to get depth intrinsics: {e}")
        print("[WARN] Using fallback (W/2, H/2). 3D accuracy will be LOW.")
        try:
            vm_check = depth_stream.get_video_mode()
            W, H = vm_check.resolutionX, vm_check.resolutionY
            HFOV_FALLBACK_RAD = 1.047
            fx_fallback = float(W) / (2.0 * np.tan(HFOV_FALLBACK_RAD / 2.0))
            intrinsics = {'fx': fx_fallback, 'fy': fx_fallback, 'cx': float(W)/2.0, 'cy': float(H)/2.0}
            print(f"[WARN] Fallback intrinsics (approx): {intrinsics}")
        except Exception as e2:
            print(f"[FATAL] Could not get VideoMode for fallback: {e2}")
            raise RuntimeError("FATAL: Cannot calculate Intrinsics.")
    return dev, depth_stream, True, intrinsics

# ===== (7) YOLO 모델 =====
def load_yolo_model():
    try:
        import torch.backends.mkldnn as mkldnn
        mkldnn.enabled = False
    except Exception:
        pass
    try:
        torch.backends.quantized.engine = 'none'
    except Exception:
        pass
    device = select_device('')
    model = DetectMultiBackend(cfg.WEIGHTS, device=device, fuse=False)
    stride, names = model.stride, model.names
    imgsz = check_img_size(cfg.IMG_SIZE, s=stride)
    model.eval()
    try:
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        with torch.inference_mode():
            _ = model(dummy)
    except Exception as e:
        print(f"[WARN] warmup failed (safe to ignore): {e}")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    names_fixed = {i: normalize_class_name(n) for i, n in names.items()}
    print("[OK] YOLO model loaded")
    print(f"[INFO] classes(original): {names}")
    print(f"[INFO] classes(normalized): {names_fixed}")
    return model, device, names_fixed, stride, imgsz

# ===== (8) 전/후처리 =====
def letterbox_and_tensor(rgb_frame, imgsz, device):
    lb_img, ratio, pad = letterbox(rgb_frame, new_shape=imgsz, auto=False, scaleFill=False, scaleup=True)
    img = lb_img[:, :, ::-1]
    img = np.ascontiguousarray(img)
    img_t = torch.from_numpy(img).to(device).permute(2,0,1).float() / 255.0
    img_t = img_t.unsqueeze(0)
    return img_t, (ratio, pad)

# ===== (8-1) 3D 좌표 변환 =====
def convert_2d_to_3d(u, v, depth_value, intrinsics):
    """영상 좌표(u,v) + 깊이 -> 카메라 좌표계(mm)
       변경: Y 축 부호를 반전하여 '위=+' / '아래=-' 로 출력."""
    if depth_value <= 0 or intrinsics is None:
        return None, None, None
    fx = intrinsics['fx']; fy = intrinsics['fy']
    cx_cam = intrinsics['cx']; cy_cam = intrinsics['cy']
    if fx == 0 or fy == 0:
        return None, None, None
    Z_cam = float(depth_value)
    X_cam = (u - cx_cam) * Z_cam / fx
    Y_cam = -(v - cy_cam) * Z_cam / fy  # Y 부호 반전
    return int(round(X_cam)), int(round(Y_cam)), int(round(Z_cam))

# ----- (깊이 안정화) -----
_LAST_VALID_D_MM = 0
_LAST_VALID_FRAME = -999999

def get_depth_at_center_smart(
    depth_mm, depth_shape, rgb_shape,
    cx_rgb, cy_rgb, max_depth_mm, bbox_w=None, bbox_h=None, frame_idx=0
):
    global _LAST_VALID_D_MM, _LAST_VALID_FRAME
    if depth_mm is None or depth_shape[0] <= 0 or depth_shape[1] <= 0:
        return _LAST_VALID_D_MM if (frame_idx - _LAST_VALID_FRAME) <= cfg.HOLD_VALID_FRAMES else 0
    DEPTH_H, DEPTH_W = depth_shape
    RGB_H, RGB_W = rgb_shape[:2]
    sx = DEPTH_W / float(RGB_W) if RGB_W > 0 else 1.0
    sy = DEPTH_H / float(RGB_H) if RGB_H > 0 else 1.0
    cx_d = int(round(cx_rgb * sx)) + cfg.PIXEL_OFFSET_X
    cy_d = int(round(cy_rgb * sy)) + cfg.PIXEL_OFFSET_Y
    if bbox_w is None or bbox_h is None:
        r = cfg.DEPTH_RADIUS_MIN
    else:
        longer = max(1, int(max(bbox_w, bbox_h) * 0.02))
        r = int(np.clip(longer, cfg.DEPTH_RADIUS_MIN, cfg.DEPTH_RADIUS_MAX))
    xs1 = max(0, cx_d - r); xs2 = min(DEPTH_W - 1, cx_d + r)
    ys1 = max(0, cy_d - r); ys2 = min(DEPTH_H - 1, cy_d + r)
    patch = depth_mm[ys1:ys2+1, xs1:xs2+1]
    if patch.size == 0:
        return _LAST_VALID_D_MM if (frame_idx - _LAST_VALID_FRAME) <= cfg.HOLD_VALID_FRAMES else 0
    valid = patch[(patch > 0) & (patch <= max_depth_mm)]
    if valid.size == 0:
        return _LAST_VALID_D_MM if (frame_idx - _LAST_VALID_FRAME) <= cfg.HOLD_VALID_FRAMES else 0
    lo = np.percentile(valid, 20)
    hi = np.percentile(valid, 80)
    core = valid[(valid >= lo) & (valid <= hi)]
    if core.size == 0:
        core = valid
    d_mm = int(np.median(core))
    d_mm = int(round(cfg.DEPTH_SCALE_A * d_mm + cfg.DEPTH_BIAS_B))
    if d_mm > 0:
        _LAST_VALID_D_MM = d_mm
        _LAST_VALID_FRAME = frame_idx
        return d_mm
    return _LAST_VALID_D_MM if (frame_idx - _LAST_VALID_FRAME) <= cfg.HOLD_VALID_FRAMES else 0

def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def _safe_hconcat(left, right, target_h):
    if left is None or left.size == 0 or left.shape[0] <= 0 or left.shape[1] <= 0:
        return None
    if right is None or right.size == 0 or right.shape[0] <= 0 or right.shape[1] <= 0:
        return left
    try:
        r_h = target_h
        r_w = int(round(right.shape[1] * (r_h / float(right.shape[0]))))
        if r_h <= 0 or r_w <= 0: return left
        right_r = cv2.resize(right, (r_w, r_h))
        left_r  = cv2.resize(left,  (left.shape[1], r_h))
        return cv2.hconcat([left_r, right_r])
    except Exception as e:
        print(f"[WARN] hconcat failed: {e}")
        return left

# ===== (추가) 중앙 게이트 & 가이드선 =====
def in_center_gate(cx: int, cy: int, img_w: int, img_h: int, thirds: int = 3) -> bool:
    if img_w <= 0 or thirds <= 1:
        return True
    x_step = img_w // thirds
    x_min, x_max = x_step, img_w - x_step
    return (x_min <= cx <= x_max)

def draw_vertical_thirds_lines(img: np.ndarray, thirds: int = 3,
                               color=(0, 255, 255), thickness: int = 2):
    if img is None or img.size == 0 or thirds <= 1:
        return
    h, w = img.shape[:2]
    x_step = w // thirds
    for i in range(1, thirds):
        x = i * x_step
        cv2.line(img, (x, 0), (x, h - 1), color, thickness, cv2.LINE_AA)

def draw_origin_dot(img: np.ndarray):
    """영상 중앙을 (0,0) 기준점으로 표시하는 작은 점."""
    if not cfg.DRAW_ORIGIN_DOT or img is None or img.size == 0:
        return
    h, w = img.shape[:2]
    cx0, cy0 = w // 2, h // 2
    cv2.circle(img, (cx0, cy0), cfg.ORIGIN_DOT_RADIUS, cfg.ORIGIN_DOT_COLOR, -1, lineType=cv2.LINE_AA)

# ===== (얼굴 검출 구성: 필요 시만 사용) =====
_FACE_CASCADE = None
_LAST_FACES = []
_LAST_FACE_FRAME = -999

def init_face_cascade():
    global _FACE_CASCADE
    if not cfg.FACE_GATE_ENABLE:
        _FACE_CASCADE = None
        return
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
        if _FACE_CASCADE.empty():
            print("[WARN] Haar cascade load failed:", cascade_path)
            _FACE_CASCADE = None
        else:
            print("[OK] Face cascade loaded")
    except Exception as e:
        print("[WARN] init_face_cascade:", e)
        _FACE_CASCADE = None

def detect_faces_lazy(bgr_img: np.ndarray, frame_idx: int):
    global _LAST_FACES, _LAST_FACE_FRAME
    if not cfg.FACE_GATE_ENABLE or _FACE_CASCADE is None:
        return []
    if (frame_idx - _LAST_FACE_FRAME) < cfg.FACE_DET_EVERY_N:
        return _LAST_FACES
    faces_out = []
    try:
        h, w = bgr_img.shape[:2]
        scale = 0.5 if max(h, w) > 640 else 1.0
        small = cv2.resize(bgr_img, (int(w*scale), int(h*scale))) if scale != 1.0 else bgr_img
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        minSize = (max(24, int(cfg.FACE_MIN_SIZE*scale)), max(24, int(cfg.FACE_MIN_SIZE*scale)))
        faces = _FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=minSize
        )
        for (x, y, fw, fh) in faces:
            x1 = int(x/scale); y1 = int(y/scale)
            x2 = int((x+fw)/scale); y2 = int((y+fh)/scale)
            faces_out.append((x1, y1, x2, y2))
    except Exception:
        faces_out = []
    _LAST_FACES = faces_out
    _LAST_FACE_FRAME = frame_idx
    return faces_out

def _iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2) -> float:
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0

def _center_in_box(cx, cy, box):
    x1,y1,x2,y2 = box
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)

# ===== (핵심) 검출 처리 =====
_frame_counter = 0
LAST_DET = {"X_mm": None, "Y_mm": None, "Z_mm": None, "valid": False}

_prev_had_det = False
_last_stop_sent_frame = -999999
STM32_STATUS_TEXT = "STM32: (대기)"
_persist_ok_frames = 0

def process_detections(annotator, pred, img_tensor_shape, rgb_frame,
                       depth_mm, depth_shape, names, ratio_pad,
                       on_first_detect=lambda: None,
                       faces=None, intrinsics=None):
    global _frame_counter, LAST_DET, _prev_had_det, _persist_ok_frames
    _frame_counter += 1
    rgb_h, rgb_w = rgb_frame.shape[:2]
    allow_set = {s.lower() for s in cfg.ALLOWED_CLASS_NAMES}

    had_det_this_frame = False
    LAST_DET["valid"] = False

    best = None
    def better(a, b):
        if b is None: return True
        za = a["Z"] if a["Z"] is not None and a["Z"] > 0 else 1e9
        zb = b["Z"] if b["Z"] is not None and b["Z"] > 0 else 1e9
        if za != zb: return za < zb
        return a["conf"] > b["conf"]

    for det in pred:
        if det is None or not len(det):
            continue
        coords = det[:, :4].clone().detach()
        coords = scale_coords(img_tensor_shape[2:], coords, rgb_frame.shape, ratio_pad=ratio_pad).round()
        confs = det[:, 4].detach().cpu().numpy() if torch.is_tensor(det[:,4]) else det[:, 4]
        clss  = det[:, 5]

        for i in range(det.shape[0]):
            x1, y1, x2, y2 = coords[i].tolist()
            x1, y1, x2, y2 = _clamp_box(x1, y1, x2, y2, rgb_w, rgb_h)
            conf = float(confs[i])
            cls_idx = int(clss[i])
            raw_name = names.get(cls_idx, "obj")
            cls_name = normalize_class_name(raw_name)

            # 1) 클래스 필터 (오직 'red'만)
            if cls_name not in allow_set:
                continue

            w, h = max(0, x2 - x1), max(0, y2 - y1)
            if w * h < cfg.MIN_BOX_AREA:
                continue

            ar = (h / w) if w > 0 else 999.0
            if not (cfg.MIN_AR <= ar <= cfg.MAX_AR):
                continue

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cfg.CENTER_GATE_ENABLE and not in_center_gate(cx, cy, rgb_w, rgb_h, cfg.CENTER_GATE_THIRDS):
                continue

            roi = rgb_frame[max(0,y1):min(rgb_h,y2), max(0,x1):min(rgb_w,x2)]

            # === 주황색 완전 차단 ===
            if cfg.ORANGE_VETO_ENABLE:
                orange_ratio = orange_ratio_hsv(roi)
                if orange_ratio >= cfg.ORANGE_VETO_RATIO_MIN:
                    continue

            high_conf = (conf >= cfg.HIGH_CONF_BYPASS_THRES)

            rr = red_color_ratio(roi)
            if not high_conf and rr < cfg.MIN_COLOR_RATIO_RED:
                continue

            if cfg.SKIN_GATE_ENABLE and not high_conf:
                sr = skin_ratio_ycrcb(roi)
                if sr >= cfg.SKIN_RATIO_MAX:
                    if rr <= 0.35:
                        continue

            if cfg.ROUNDNESS_GATE_ENABLE and not high_conf:
                rm = red_mask(roi)
                circ = largest_contour_circularity(rm)
                if circ < cfg.ROUNDNESS_MIN:
                    continue

            d_mm = get_depth_at_center_smart(
                depth_mm, depth_shape, rgb_frame.shape,
                cx, cy, cfg.DEPTH_MAX_MM, bbox_w=w, bbox_h=h, frame_idx=_frame_counter
            )
            X_cam, Y_cam, Z_cam = convert_2d_to_3d(cx, cy, d_mm, intrinsics)

            # 실물 크기 게이트
            if cfg.SIZE_GATE_ENABLE and (X_cam is not None) and (Z_cam and Z_cam > 0):
                _, _, diam_mm = bbox_px_to_mm(w, h, Z_cam, intrinsics)
                if diam_mm is None or not (cfg.TOMATO_DIAM_MM_MIN <= diam_mm <= cfg.TOMATO_DIAM_MM_MAX):
                    had_det_this_frame = True
                    continue
            else:
                diam_mm = None

            cand = {
                "x1":x1,"y1":y1,"x2":x2,"y2":y2,
                "cx":cx,"cy":cy,
                "X":X_cam,"Y":Y_cam,"Z":Z_cam,
                "conf":conf,"diam_mm":diam_mm
            }
            if X_cam is not None:
                if better(cand, best):
                    best = cand
            had_det_this_frame = True

    if best is not None and best["Z"] is not None:
        x1,y1,x2,y2 = best["x1"],best["y1"],best["x2"],best["y2"]
        cx,cy = best["cx"],best["cy"]
        dist_txt = f"{(best['Z']/10.0):.1f}cm"
        annotator.box_label((x1, y1, x2, y2), f"red {best['conf']:.2f} | {dist_txt}", color=BOX_COLOR)
        cv2.circle(annotator.im, (int(cx), int(cy)), 5, (255,255,255), -1)
        cv2.putText(annotator.im, f"(X,Y,Z)=({best['X']},{best['Y']},{best['Z']}) mm",
                    (x1, min(rgb_h - 25, y2 + 36)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2, cv2.LINE_AA)
        if cfg.SIZE_GATE_ENABLE and best["diam_mm"] is not None:
            cv2.putText(annotator.im, f"D≈{best['diam_mm']:.0f}mm",
                        (x1, min(rgb_h - 5, y2 + 54)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2, cv2.LINE_AA)

        LAST_DET["X_mm"] = best["X"]
        LAST_DET["Y_mm"] = best["Y"]
        LAST_DET["Z_mm"] = best["Z"]
        LAST_DET["valid"] = True

        if (_frame_counter % cfg.PRINT_EVERY_N) == 0:
            print(f"[DET] best (X,Y,Z)=({best['X']},{best['Y']},{best['Z']}) mm")

    trigger_this_frame = False
    if cfg.PERSIST_GATE_ENABLE:
        if best is not None and best["Z"] is not None:
            _persist_ok_frames += 1
        else:
            _persist_ok_frames = 0
        trigger_this_frame = (_persist_ok_frames >= cfg.PERSIST_MIN_FRAMES)
    else:
        trigger_this_frame = (best is not None and best["Z"] is not None)

    if trigger_this_frame and not _prev_had_det:
        on_first_detect()
    _prev_had_det = (best is not None)

    return had_det_this_frame

# ===== (9) 안전 정리 =====
def cleanup(color_cap, depth_stream, dev, depth_started, ser=None, ser_stm=None):
    print("[INFO] Cleaning up resources...")
    try: time.sleep(0.05)
    except Exception: pass
    try:
        if cv2.getWindowProperty("Tomato + Depth", 0) >= 0:
            cv2.destroyWindow("Tomato + Depth")
        cv2.waitKey(1); time.sleep(0.05)
        cv2.destroyAllWindows(); time.sleep(0.05)
    except Exception as e:
        print(f"[WARN] destroyAllWindows: {e}")
    try:
        if color_cap: color_cap.release(); time.sleep(0.05)
    except Exception as e:
        print(f"[WARN] color_cap.release: {e}")
    try:
        if depth_stream and depth_started:
            try: depth_stream.stop()
            except Exception: pass
            time.sleep(0.05)
    except Exception as e:
        print(f"[WARN] depth_stream.stop: {e}")
    try:
        if dev:
            try: dev.close()
            except Exception: pass
            time.sleep(0.05)
    except Exception as e:
        print(f"[WARN] dev.close: {e}")
    try:
        if ser and getattr(ser, "is_open", False): ser.close()
    except Exception as e:
        print(f"[WARN] serial.close (arduino): {e}")
    try:
        if ser_stm and getattr(ser_stm, "is_open", False): ser_stm.close()
    except Exception as e:
        print(f"[WARN] serial.close (stm32): {e}")
    print("[DONE] Clean exit")

# ===== (10) 메인 =====
def main():
    global _last_stop_sent_frame, STM32_STATUS_TEXT
    init_face_cascade()

    print("[STAGE] Loading YOLO model first...")
    try:
        model, device, names, stride, imgsz = load_yolo_model()
        print("[STAGE] YOLO ready")
    except Exception as e:
        print(f"[ERR] YOLO load failed (before any camera/depth init): {e}")
        return

    print("[STAGE] Initializing OpenNI depth...")
    try:
        dev, depth_stream, depth_started, depth_intrinsics = initialize_openni()
        print("[STAGE] Depth ready")
    except Exception as e:
        print(f"[ERR] Depth start failed: {e}")
        return

    print("[STAGE] Opening RGB camera...")
    color_cap = open_color_camera(cfg.COLOR_IDX_CANDIDATES, cfg.RGB_W, cfg.RGB_H, cfg.RGB_FPS)
    if not color_cap:
        cleanup(None, depth_stream, dev, True)
        return

    ser_arduino = open_arduino_serial()
    ser_stm32   = open_stm32_serial()

    cv2.namedWindow("Tomato + Depth", cv2.WINDOW_NORMAL)
    print("[INFO] Press 'q' or ESC to exit.")

    last_det_ts = time.monotonic()
    g_sent_for_gap = False
    NO_DET_TIMEOUT_SEC = 5.0

    try:
        while True:
            ok, rgb = color_cap.read()
            if not ok or rgb is None:
                print("[WARN] RGB read failed"); break
            if rgb.size == 0 or rgb.shape[0] <= 0 or rgb.shape[1] <= 0:
                print("[WARN] RGB invalid shape"); continue

            rgb = cv2.resize(rgb, (cfg.RGB_W, cfg.RGB_H))
            if cfg.MIRROR_RGB: rgb = cv2.flip(rgb, 1)

            # 카메라 180° 회전(거꾸로 장착 대응)
            rgb = cv2.flip(rgb, -1)

            rgb_annot = rgb.copy()

            depth_mm = None
            depth_vis = None
            depth_shape = (0, 0)
            try:
                dframe = depth_stream.read_frame()
                if dframe is not None:
                    depth_mm = depth_frame_to_numpy_mm(dframe)
                    if depth_mm is not None:
                        # 깊이도 180° 회전
                        depth_mm = cv2.flip(depth_mm, -1)
                        depth_shape = depth_mm.shape
                        depth_vis = depth_mm_to_colormap(depth_mm, vmax=cfg.DEPTH_MAX_MM)
            except Exception as e:
                print(f"[WARN] depth read skipped: {e}")
                depth_vis = None

            try:
                img_t, (ratio, pad) = letterbox_and_tensor(rgb, imgsz, device)
                with torch.inference_mode():
                    pred = model(img_t)
            except Exception as e:
                print(f"[WARN] inference exception: {e} -> retry with tiny warmup")
                try:
                    with torch.inference_mode():
                        _ = model(torch.zeros_like(img_t)[:, :, :32, :32])
                        pred = model(img_t)
                except Exception as e2:
                    print(f"[FATAL] inference segfault/exception suspected: {e2}")
                    break

            try:
                pred = non_max_suppression(pred, conf_thres=cfg.CONF_THRES, iou_thres=cfg.IOU_THRES)
                pred = [d.clone().detach() if (d is not None and isinstance(d, torch.Tensor)) else d for d in pred]
            except Exception as e:
                print(f"[ERR] NMS failed: {e}")
                break

            annot = Annotator(rgb_annot, line_width=2)
            ratio_pad = ((ratio[0], ratio[1]), (pad[0], pad[1]))

            faces = detect_faces_lazy(rgb, _frame_counter)

            def _on_first_detect():
                global _last_stop_sent_frame, STM32_STATUS_TEXT
                ok_send = send_stop_to_stm32(ser_stm32)
                _last_stop_sent_frame = _frame_counter
                if ok_send:
                    print("[ACT] DETECTED -> sent 'stop' to STM32 (UART4)")
                    STM32_STATUS_TEXT = "STM32: stop (검출)"

            had_det = process_detections(
                annot, pred, img_t.shape, rgb,
                depth_mm, depth_shape, names, ratio_pad,
                on_first_detect=_on_first_detect,
                faces=faces,
                intrinsics=depth_intrinsics
            )

            now_ts = time.monotonic()
            if had_det:
                last_det_ts = now_ts
                g_sent_for_gap = False
            else:
                if (not g_sent_for_gap) and (now_ts - last_det_ts >= NO_DET_TIMEOUT_SEC):
                    if send_g_to_stm32(ser_stm32):
                        print("[ACT] NO DETECT >= 5s -> sent 'g' to STM32")
                        STM32_STATUS_TEXT = "STM32: g (무검출 5s)"
                        g_sent_for_gap = True

            if LAST_DET["valid"]:
                send_xyz_to_arduino(
                    ser_arduino,
                    LAST_DET["X_mm"],
                    LAST_DET["Y_mm"],
                    LAST_DET["Z_mm"]
                )
                LAST_DET["valid"] = False

            stm_line = try_read_stm32_line(ser_stm32)
            if stm_line:
                print(f"[STM32] {stm_line}")
                if "pwm=0" in stm_line.replace(" ", ""):
                    STM32_STATUS_TEXT = "STM32: pwm=0 (수신)"
                else:
                    STM32_STATUS_TEXT = f"STM32: {stm_line[:40]}"

            left = annot.result()

            # 중앙 (0,0) 점 찍기
            draw_origin_dot(left)

            # 가이드 라인
            draw_vertical_thirds_lines(left, thirds=cfg.CENTER_GATE_THIRDS,
                                       color=(0, 255, 255), thickness=2)

            canvas = _safe_hconcat(cv2.resize(left, (cfg.RGB_W, cfg.RGB_H)), depth_vis, cfg.RGB_H)
            try:
                overlay = canvas if canvas is not None else left
                cv2.putText(overlay, STM32_STATUS_TEXT, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Tomato + Depth", overlay)
            except Exception as e:
                print(f"[WARN] imshow failed: {e}")
                break

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

    finally:
        cleanup(color_cap, depth_stream, dev, True, ser=ser_arduino, ser_stm=ser_stm32)

if __name__ == "__main__":
    main()
