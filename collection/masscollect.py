import warnings

warnings.filterwarnings("ignore", message=r".*[Mm]INGW-[Ww]64.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy._core.getlimits")
warnings.filterwarnings("ignore", category=UserWarning)

import threading
import queue
import cv2
import time
import os
import sys
import platform
import zipfile
import random
import string
import subprocess
import shutil
import webbrowser
import numpy as np
import serial
import serial.tools.list_ports
from enum import Enum
import tkinter as tk
from tkinter import ttk

try:
    import sv_ttk
    HAS_SV_TTK = True
except ImportError:
    HAS_SV_TTK = False

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

speech_lock = threading.Lock()


def speak(text):
    done_event = threading.Event()
    threading.Thread(
        target=_speak_platform_specific,
        args=(text, done_event),
        daemon=True,
    ).start()
    return done_event


def _speak_platform_specific(text, done_event):
    try:
        with speech_lock:
            system = platform.system().lower()
            if system == "darwin":
                subprocess.run(["say", text], check=False, timeout=10)
            elif system == "windows":
                ps_cmd = (
                    f'Add-Type -AssemblyName System.Speech; '
                    f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}");'
                )
                subprocess.run(["powershell", "-Command", ps_cmd], check=False, timeout=20)
            elif system == "linux":
                try:
                    subprocess.run(["espeak", text], check=False, timeout=20)
                except FileNotFoundError:
                    try:
                        proc = subprocess.Popen(["festival", "--tts"], stdin=subprocess.PIPE)
                        proc.communicate(input=text.encode())
                    except FileNotFoundError:
                        pass
    except Exception:
        pass
    finally:
        done_event.set()


WAIT_TIME = 0.1
ETVR_HEADER = b"\xff\xa0"
ETVR_HEADER_FRAME = b"\xff\xa1"
ETVR_HEADER_LEN = 6

PROMPTS = [
    "Look left", "Look left and squint", "Look right", "Look right and squint",
    "Look up", "Look up and squint", "Look down", "Look down and squint",
    "Look top-left", "Look top-right", "Look bottom-left", "Look bottom-right",
    "Look straight", "Look straight and squint", "Close your eyes",
    "Squeeze your eyes shut",
    "Widen your eyes and look straight", "Widen your eyes and look left",
    "Widen your eyes and look right",
    "Raise eyebrows fully and look forward",
    "Raise eyebrows halfway and look forward",
    "Lower eyebrows fully and look forward",
    "Lower eyebrows halfway and look forward",
    "Raise eyebrows fully and look in random direction",
    "Lower eyebrows fully and look in random direction",
    "Close your eyes and look in random direction",
    "Look in a full circle starting now",
]


class CameraState(Enum):
    CONNECTING = 0
    CONNECTED = 1
    DISCONNECTED = 2


def is_serial_capture_source(addr):
    if isinstance(addr, str):
        return (
            addr.lower().startswith("com")
            or addr.startswith("/dev/cu")
            or addr.startswith("/dev/tty")
        )
    return False


class SplitUVCCamera:
    """Single UVC camera split down the middle into left / right eye feeds (BigScreen Beyond)."""

    def __init__(self, capture_source, cancellation_event, status_queue,
                 left_output_queue, right_output_queue):
        self.capture_source = capture_source
        self.cancellation_event = cancellation_event
        self.status_queue = status_queue
        self.left_output_queue = left_output_queue
        self.right_output_queue = right_output_queue
        self.fps = 0
        self.cap = None
        self.total_frames = 0

    def run(self):
        try:
            if isinstance(self.capture_source, str) and self.capture_source.isdigit():
                self.capture_source = int(self.capture_source)

            backends = [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
            for backend in backends:
                try:
                    tmp = cv2.VideoCapture(self.capture_source, backend)
                    if tmp.isOpened():
                        ret, frame = tmp.read()
                        if ret and frame is not None:
                            self.cap = tmp
                            self.status_queue.put(("INFO", f"Opened with backend {backend}"))
                            break
                    tmp.release()
                except Exception as e:
                    self.status_queue.put(("WARNING", f"Backend {backend} failed: {e}"))

            if self.cap is None:
                self.status_queue.put(("ERROR", "Failed to open camera"))
                return

            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.status_queue.put(("INFO", f"Resolution: {w}x{h}"))
            except Exception:
                pass

            self.status_queue.put(("INFO", "Split UVC camera ready"))
            failed = 0
            fps_count = 0
            fps_start = time.time()

            while not self.cancellation_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    failed += 1
                    if failed >= 10:
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(self.capture_source)
                        failed = 0
                    time.sleep(0.1)
                    continue

                failed = 0
                h, w = frame.shape[:2]
                mid = w // 2
                lf = cv2.resize(frame[:, :mid], (240, 240))
                rf = cv2.resize(frame[:, mid:], (240, 240))
                if len(lf.shape) > 2:
                    lf = cv2.cvtColor(lf, cv2.COLOR_BGR2GRAY)
                if len(rf.shape) > 2:
                    rf = cv2.cvtColor(rf, cv2.COLOR_BGR2GRAY)

                self.total_frames += 1
                fps_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    self.fps = fps_count / elapsed
                    fps_count = 0
                    fps_start = time.time()

                self._clear(self.left_output_queue)
                self._clear(self.right_output_queue)
                try:
                    self.left_output_queue.put((lf, self.total_frames, self.fps), timeout=0.01)
                except queue.Full:
                    pass
                try:
                    self.right_output_queue.put((rf, self.total_frames, self.fps), timeout=0.01)
                except queue.Full:
                    pass
                time.sleep(0.001)

        except Exception as e:
            self.status_queue.put(("ERROR", str(e)))
        finally:
            if self.cap is not None:
                self.cap.release()

    def _clear(self, q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


class SerialCamera:
    """ESP/ETVR tracker over a serial COM port."""

    def __init__(self, capture_source, cancellation_event, status_queue, output_queue):
        self.camera_status = CameraState.CONNECTING
        self.capture_source = capture_source
        self.camera_status_outgoing = status_queue
        self.camera_output_outgoing = output_queue
        self.cancellation_event = cancellation_event
        self.serial_connection = None
        self.fps = 0
        self.buffer = b""
        self.fl = [0]
        self.prevft = time.time()
        self.total_frames = 0

    def __del__(self):
        self._cleanup()

    def _cleanup(self):
        if self.serial_connection is not None:
            self.serial_connection.close()

    def run(self):
        try:
            while not self.cancellation_event.is_set():
                if (self.serial_connection is None
                        or not self.serial_connection.is_open
                        or self.camera_status == CameraState.DISCONNECTED):
                    self._start_connection(self.capture_source)
                if self.camera_status == CameraState.CONNECTED:
                    self._get_picture()
                else:
                    if self.cancellation_event.wait(WAIT_TIME):
                        break
        except Exception as e:
            self.camera_status_outgoing.put(("ERROR", str(e)))
        finally:
            self._cleanup()

    def _start_connection(self, port):
        if self.serial_connection is not None and self.serial_connection.is_open:
            if self.serial_connection.port == port:
                return
            self.serial_connection.close()

        com_ports = [tuple(p) for p in serial.tools.list_ports.comports()]
        if not any(port in p for p in com_ports):
            self.camera_status_outgoing.put(("WARNING", f"Port {port} not found, retrying..."))
            return

        try:
            rate = 115200 if sys.platform == "darwin" else 3000000
            conn = serial.Serial(baudrate=rate, port=port, xonxoff=False, dsrdtr=False, rtscts=False)
            if sys.platform == "win32":
                conn.set_buffer_size(rx_size=32768, tx_size=32768)
            self.camera_status_outgoing.put(("INFO", f"Serial device connected on {port}"))
            self.serial_connection = conn
            self.camera_status = CameraState.CONNECTED
        except serial.SerialException as e:
            self.camera_status_outgoing.put(("ERROR", f"Failed on {port}: {e}"))
            self.camera_status = CameraState.DISCONNECTED

    def _get_next_packet_bounds(self):
        beg = -1
        while beg == -1:
            if self.cancellation_event.is_set():
                return None, None
            self.buffer += self.serial_connection.read(2048)
            beg = self.buffer.find(ETVR_HEADER + ETVR_HEADER_FRAME)
        if beg > 0:
            self.buffer = self.buffer[beg:]
            beg = 0
        end = int.from_bytes(self.buffer[4:6], signed=False, byteorder="little")
        self.buffer += self.serial_connection.read(end - len(self.buffer))
        return beg, end

    def _get_next_jpeg(self):
        beg, end = self._get_next_packet_bounds()
        if beg is None:
            return None
        jpeg = self.buffer[beg + ETVR_HEADER_LEN: end + ETVR_HEADER_LEN]
        self.buffer = self.buffer[end + ETVR_HEADER_LEN:]
        return jpeg

    def _get_picture(self):
        if self.cancellation_event.is_set():
            return
        conn = self.serial_connection
        if conn is None or not conn.is_open:
            return
        try:
            if conn.in_waiting:
                jpeg = self._get_next_jpeg()
                if jpeg:
                    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        self.camera_status_outgoing.put(("WARNING", "Corrupted JPEG, dropping frame"))
                        return
                    if conn.in_waiting >= 32768:
                        conn.reset_input_buffer()
                        self.buffer = b""
                    if len(image.shape) > 2 and image.shape[2] > 1:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    self.total_frames += 1
                    t = time.time()
                    if t != self.prevft:
                        fps = 1 / (t - self.prevft)
                        self.prevft = t
                        self.fl = (self.fl + [fps])[-60:]
                        self.fps = sum(self.fl) / len(self.fl)

                    while not self.camera_output_outgoing.empty():
                        try:
                            self.camera_output_outgoing.get_nowait()
                        except queue.Empty:
                            break
                    try:
                        self.camera_output_outgoing.put((image, self.total_frames, self.fps), timeout=0.1)
                        time.sleep(0.01)
                    except queue.Full:
                        pass

        except serial.SerialException as e:
            self.camera_status_outgoing.put(("WARNING", str(e)))
            if conn.is_open:
                conn.close()
            self.camera_status = CameraState.DISCONNECTED
        except OSError as e:
            self.camera_status_outgoing.put(("WARNING", str(e)))
            if conn.is_open:
                conn.close()
            self.camera_status = CameraState.DISCONNECTED


class CV2Camera:
    """Standard webcam via OpenCV."""

    def __init__(self, capture_source, cancellation_event, status_queue, output_queue):
        self.capture_source = capture_source
        self.cancellation_event = cancellation_event
        self.status_queue = status_queue
        self.output_queue = output_queue
        self.frame_count = 0
        self.fps = 0
        self.cap = None
        self.total_frames = 0

    def run(self):
        try:
            if isinstance(self.capture_source, str) and self.capture_source.isdigit():
                self.capture_source = int(self.capture_source)

            self.cap = cv2.VideoCapture(self.capture_source)
            if not self.cap.isOpened():
                self.status_queue.put(("ERROR", "Failed to open camera"))
                return

            self.status_queue.put(("INFO", "Camera ready"))
            failed = 0
            start = time.time()

            while not self.cancellation_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    failed += 1
                    if failed >= 10:
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(self.capture_source)
                        failed = 0
                    time.sleep(0.2)
                    continue

                failed = 0
                if len(frame.shape) > 2 and frame.shape[2] > 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                self.frame_count += 1
                self.total_frames += 1
                elapsed = time.time() - start
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    start = time.time()

                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break
                try:
                    self.output_queue.put((frame, self.total_frames, self.fps), timeout=0.1)
                    time.sleep(0.01)
                except queue.Full:
                    pass

        except Exception as e:
            self.status_queue.put(("ERROR", str(e)))
        finally:
            if self.cap is not None:
                self.cap.release()


def get_best_codec():
    system = platform.system().lower()
    if system == "windows":
        codecs_to_try = [("XVID", "avi"), ("avc1", "mp4"), ("DIVX", "avi"), ("MJPG", "avi")]
    else:
        codecs_to_try = [("avc1", "mp4"), ("H264", "mp4"), ("XVID", "avi"), ("MJPG", "avi")]

    for codec, container in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            tmp = os.path.join(os.getcwd(), f"_tmp_test_{codec}.{container}")
            w = cv2.VideoWriter(tmp, fourcc, 30, (240, 240), False)
            ok = w.isOpened()
            w.release()
            if os.path.exists(tmp):
                os.remove(tmp)
            if ok:
                return fourcc, codec, container
        except Exception:
            pass

    return cv2.VideoWriter_fourcc(*"MJPG"), "MJPG", "avi"


def _zip_output(seed, output_dir):
    zip_name = f"{seed}_ETVR_User_Data_Output.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                fp = os.path.join(root, file)
                zf.write(fp, os.path.relpath(fp, start=output_dir))
    return zip_name


PREVIEW_SIZE = 240
BG_DARK = "#1c1c1c"
STATUS_FG_DEFAULT = "#e0e0e0"
STATUS_BG_SUCCESS = "#1a4d2e"
STATUS_FG_SUCCESS = "#c8f5d8"
WINDOW_TITLE_HEADSET = "ETVR Data Collection — Bigscreen Beyond 2e"
WINDOW_TITLE_ETVR = "ETVR Data Collection — EyeTrackVR Setup"
LEAPV2_SUBMISSION_URL = "https://ask.eyetrackvr.dev/LEAPV2_Data_Submission"


def _apply_windows_dark_titlebar(root: tk.Tk) -> None:
    """Use dark title bar / window frame on Windows 10 (2004+) and Windows 11."""
    if platform.system() != "Windows":
        return
    try:
        import ctypes

        root.update_idletasks()
        wid = root.winfo_id()
        user32 = ctypes.windll.user32
        ga_root = 2
        hwnd = user32.GetAncestor(wid, ga_root)
        if not hwnd:
            hwnd = user32.GetParent(wid)
        if not hwnd:
            return
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            ctypes.c_void_p(hwnd),
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
    except Exception:
        pass


class CollectionApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE_ETVR)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if HAS_SV_TTK:
            sv_ttk.set_theme("dark")

        self.mode_var = tk.StringVar(value="esp")
        self.src1_var = tk.StringVar(value="")
        self.src2_var = tk.StringVar(value="")

        self.cancel_events: list = []
        self.output_queues: list = []
        self.status_queues: list = []
        self.camera_threads: list = []
        self.cameras_connected = False
        self.eyes: list = []

        self.session_running = False
        self.session_cancel = threading.Event()

        self.preview_frames: dict = {}

        self.gui_queue: queue.Queue = queue.Queue()

        self._build_ui()
        self.root.after_idle(_apply_windows_dark_titlebar, self.root)
        self.root.after(200, _apply_windows_dark_titlebar, self.root)
        self._poll_gui_queue()
        self._update_previews()

    def _apply_status_style(self, success: bool):
        if success:
            self._status_lbl.configure(bg=STATUS_BG_SUCCESS, fg=STATUS_FG_SUCCESS)
        else:
            self._status_lbl.configure(bg=BG_DARK, fg=STATUS_FG_DEFAULT)

    def _open_submission_page(self):
        webbrowser.open(LEAPV2_SUBMISSION_URL, new=2)

    def _build_ui(self):
        self.root.minsize(860, 560)

        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        sidebar = ttk.Frame(outer, width=270)
        sidebar.pack(side="left", fill="y", padx=(0, 12))
        sidebar.pack_propagate(False)

        content = ttk.Frame(outer)
        content.pack(side="left", fill="both", expand=True)

        self._build_sidebar(sidebar)
        self._build_content(content)

    def _build_sidebar(self, parent):
        mode_lf = ttk.LabelFrame(parent, text="Setup Type", padding=8)
        mode_lf.pack(fill="x", pady=(0, 8))

        ttk.Radiobutton(
            mode_lf, text="EyeTrackVR Setup",
            variable=self.mode_var, value="esp",
            command=self._on_mode_change,
        ).pack(anchor="w")
        ttk.Radiobutton(
            mode_lf, text="Bigscreen Beyond 2e",
            variable=self.mode_var, value="bsb",
            command=self._on_mode_change,
        ).pack(anchor="w")

        cam_lf = ttk.LabelFrame(parent, text="Camera Settings", padding=8)
        cam_lf.pack(fill="x", pady=(0, 8))

        self._src1_lbl = ttk.Label(cam_lf, text="Left (UVC / COM port / URL):")
        self._src1_lbl.pack(anchor="w")
        self._src1_entry = ttk.Entry(cam_lf, textvariable=self.src1_var)
        self._src1_entry.pack(fill="x", pady=(2, 8))

        self._src2_lbl = ttk.Label(cam_lf, text="Right (UVC / COM port / URL):")
        self._src2_entry = ttk.Entry(cam_lf, textvariable=self.src2_var)

        self._cam_btn_row = ttk.Frame(cam_lf)
        self._cam_btn_row.pack(fill="x")
        ttk.Button(self._cam_btn_row, text="Scan", command=self._scan_cameras, width=8).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(self._cam_btn_row, text="Connect", command=self._connect_cameras).pack(
            side="left", fill="x", expand=True
        )

        ctrl_lf = ttk.LabelFrame(parent, text="Controls", padding=8)
        ctrl_lf.pack(fill="x", pady=(0, 8))

        ttk.Button(ctrl_lf, text="Test Audio", command=self._test_audio).pack(fill="x", pady=(0, 6))

        self._start_btn = ttk.Button(
            ctrl_lf, text="▶ Start Collection",
            command=self._start_collection, style="Accent.TButton",
        )
        self._start_btn.pack(fill="x", pady=(0, 4))

        self._stop_btn = ttk.Button(ctrl_lf, text="■ Stop", command=self._stop_collection, state="disabled")
        self._stop_btn.pack(fill="x")

        ttk.Button(
            ctrl_lf,
            text="Open Submission Page",
            command=self._open_submission_page,
        ).pack(fill="x", pady=(8, 0))

        status_lf = ttk.LabelFrame(parent, text="Status", padding=8)
        status_lf.pack(fill="both", expand=True)

        self._status_var = tk.StringVar(value="Not connected.")
        self._status_lbl = tk.Label(
            status_lf,
            textvariable=self._status_var,
            wraplength=235,
            justify="left",
            anchor="w",
            bg=BG_DARK,
            fg=STATUS_FG_DEFAULT,
            font=("Segoe UI", 9),
        )
        self._status_lbl.pack(anchor="w", fill="x")

        self._prompt_var = tk.StringVar(value="")
        ttk.Label(
            status_lf, textvariable=self._prompt_var,
            wraplength=235, justify="left", font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(6, 0))

        self._progress_var = tk.StringVar(value="")
        ttk.Label(status_lf, textvariable=self._progress_var, justify="left").pack(anchor="w", pady=(4, 0))

    def _build_content(self, parent):
        self._preview_lf = ttk.LabelFrame(parent, text="Camera Previews", padding=8)
        self._preview_lf.pack(fill="both", expand=True)

        self._preview_row = ttk.Frame(self._preview_lf)
        self._preview_row.pack(fill="both", expand=True)

        self._countdown_var = tk.StringVar(value="")
        self._countdown_lbl = tk.Label(
            self._preview_lf,
            textvariable=self._countdown_var,
            font=("Segoe UI", 96, "bold"),
            fg="#ff6b35",
            bg=BG_DARK,
        )

        self._preview_labels: list = []
        self._setup_preview_panes(2)

        self._on_mode_change()

    def _setup_preview_panes(self, count):
        for w in self._preview_row.winfo_children():
            w.destroy()
        self._preview_labels = []

        blank = self._blank_photo(PREVIEW_SIZE, PREVIEW_SIZE)
        if count == 2:
            preview_titles = ["Left Eye", "Right Eye"]
        else:
            e = (self.eyes[0] if self.eyes else "l").lower()
            preview_titles = ["Left Eye" if e == "l" else "Right Eye"]

        for i in range(count):
            pane = ttk.Frame(self._preview_row)
            pane.pack(side="left", padx=10, pady=4, anchor="n")

            ttk.Label(pane, text=preview_titles[i], font=("Segoe UI", 10, "bold")).pack(pady=(0, 4))

            if HAS_PIL:
                lbl = tk.Label(pane, image=blank, bg=BG_DARK, relief="flat", bd=2)
                lbl.image = blank
            else:
                lbl = tk.Label(pane, text="(Pillow not installed)", width=30, height=15,
                               bg=BG_DARK, fg="gray", relief="sunken")
            lbl.pack()
            self._preview_labels.append(lbl)

    def _blank_photo(self, w, h):
        if not HAS_PIL:
            return None
        img = Image.new("L", (w, h), color=45)
        return ImageTk.PhotoImage(img)

    def _frame_to_photo(self, frame):
        if not HAS_PIL:
            return None
        try:
            if len(frame.shape) == 2:
                pil = Image.fromarray(frame, mode="L")
            else:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil = pil.resize((PREVIEW_SIZE, PREVIEW_SIZE), Image.NEAREST)
            return ImageTk.PhotoImage(pil)
        except Exception:
            return None

    def _on_mode_change(self):
        is_esp = self.mode_var.get() == "esp"
        self.root.title(WINDOW_TITLE_ETVR if is_esp else WINDOW_TITLE_HEADSET)

        self._src2_lbl.pack_forget()
        self._src2_entry.pack_forget()

        if is_esp:
            self._src1_lbl.configure(text="Left (UVC / COM port / URL):")
            self._src2_lbl.configure(text="Right (UVC / COM port / URL):")
            self._src2_lbl.pack(anchor="w", before=self._cam_btn_row)
            self._src2_entry.pack(fill="x", pady=(2, 8), before=self._cam_btn_row)
            self._src2_entry.configure(state="normal")
        else:
            self._src1_lbl.configure(text="Source (UVC Index):")
            self._src2_entry.configure(state="disabled")

    def _parse_source(self, s: str):
        s = s.strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        if is_serial_capture_source(s):
            return s
        if not (s.lower().startswith("http://") or s.lower().startswith("https://")):
            return "http://" + s
        return s

    def _scan_cameras(self):
        self._status_var.set("Scanning cameras…")
        self._apply_status_style(False)

        def _scan():
            found = []
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        found.append(i)
                    cap.release()
            self.gui_queue.put(("status", f"Available indices: {found if found else 'none'}"))

        threading.Thread(target=_scan, daemon=True).start()

    def _connect_cameras(self):
        self._stop_cameras()
        mode = self.mode_var.get()

        if mode == "bsb":
            src = self._parse_source(self.src1_var.get())
            if src is None:
                self._status_var.set("Invalid camera source.")
                self._apply_status_style(False)
                return

            ce = threading.Event()
            sq = queue.Queue()
            lq = queue.Queue(maxsize=2)
            rq = queue.Queue(maxsize=2)

            self.cancel_events = [ce]
            self.status_queues = [sq]
            self.output_queues = [lq, rq]
            self.eyes = ["l", "r"]

            cam = SplitUVCCamera(src, ce, sq, lq, rq)
            th = threading.Thread(target=cam.run, daemon=True)
            self.camera_threads = [th]
            th.start()

            self._setup_preview_panes(2)

        else:
            src_left = self._parse_source(self.src1_var.get())
            src_right = self._parse_source(self.src2_var.get())
            if src_left is None and src_right is None:
                self._status_var.set("Enter at least one source: Left and/or Right (UVC / COM / URL).")
                self._apply_status_style(False)
                return

            sources = []
            self.eyes = []
            if src_left is not None:
                sources.append(src_left)
                self.eyes.append("l")
            if src_right is not None:
                sources.append(src_right)
                self.eyes.append("r")

            self.cancel_events = []
            self.status_queues = []
            self.output_queues = []
            self.camera_threads = []

            for src in sources:
                ce = threading.Event()
                sq = queue.Queue()
                oq = queue.Queue(maxsize=1)
                self.cancel_events.append(ce)
                self.status_queues.append(sq)
                self.output_queues.append(oq)

                if is_serial_capture_source(src):
                    cam = SerialCamera(src, ce, sq, oq)
                else:
                    cam = CV2Camera(src, ce, sq, oq)

                th = threading.Thread(target=cam.run, daemon=True)
                self.camera_threads.append(th)
                th.start()

            self._setup_preview_panes(len(sources))

        self.cameras_connected = True
        self.preview_frames = {}
        self._status_var.set("Cameras connecting… check previews.")
        self._apply_status_style(False)

    def _stop_cameras(self):
        for ce in self.cancel_events:
            ce.set()
        for th in self.camera_threads:
            th.join(timeout=2)
        self.cancel_events = []
        self.camera_threads = []
        self.output_queues = []
        self.status_queues = []
        self.cameras_connected = False
        self.preview_frames = {}

    def _test_audio(self):
        def _run():
            speak("testing audio").wait()
        threading.Thread(target=_run, daemon=True).start()

    def _start_collection(self):
        if self.session_running:
            return
        if not self.cameras_connected or not self.output_queues:
            self._status_var.set("Connect cameras first.")
            self._apply_status_style(False)
            return

        self.session_running = True
        self.session_cancel.clear()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

        threading.Thread(target=self._run_session, daemon=True).start()

    def _stop_collection(self):
        self.session_cancel.set()

    def _run_session(self):
        for i in range(5, 0, -1):
            if self.session_cancel.is_set():
                self.gui_queue.put(("countdown", ""))
                self.gui_queue.put(("status", "Cancelled."))
                self.gui_queue.put(("session_done", None))
                return
            self.gui_queue.put(("countdown", str(i)))
            self.gui_queue.put(("status", f"Starting in {i}…"))
            time.sleep(1)

        self.gui_queue.put(("countdown", ""))
        self.gui_queue.put(("status", "Session started!"))

        seed = "".join(random.choices(string.ascii_letters + string.digits, k=10))
        output_dir = "eye_captures"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        n = len(self.eyes)
        oqs = self.output_queues[:n]

        timestamp_files = []
        for eye in self.eyes:
            ts = os.path.join(output_dir, f"{seed}_{eye}_timestamps.txt")
            with open(ts, "w") as f:
                f.write("# Format: <frame_number> <prompt_text>\n")
                f.write("# Recorded on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
                f.write("# Seed: " + seed + "\n\n")
            timestamp_files.append(ts)

        first_frames = [None] * n
        deadline = time.time() + 10
        while any(f is None for f in first_frames) and time.time() < deadline:
            for i in range(n):
                if first_frames[i] is None and not oqs[i].empty():
                    try:
                        frame, _, _ = oqs[i].get_nowait()
                        first_frames[i] = frame
                        self.preview_frames[i] = frame
                    except queue.Empty:
                        pass
            time.sleep(0.05)

        if any(f is None for f in first_frames):
            self.gui_queue.put(("status", "Could not get frames from cameras. Are they connected?"))
            self.gui_queue.put(("session_done", None))
            return

        fourcc, _, container = get_best_codec()

        video_writers = []
        for i in range(n):
            h, w = first_frames[i].shape[:2]
            fn = os.path.join(output_dir, f"{seed}_full_session_{self.eyes[i]}.{container}")
            vw = cv2.VideoWriter(fn, fourcc, 60, (w, h), False)
            video_writers.append(vw)

        def drain_to_video():
            for j in range(n):
                while not oqs[j].empty():
                    try:
                        frame, _, _ = oqs[j].get_nowait()
                        video_writers[j].write(frame)
                        self.preview_frames[j] = frame
                    except queue.Empty:
                        break

        try:
            for idx, prompt in enumerate(PROMPTS):
                if self.session_cancel.is_set():
                    break

                self.gui_queue.put(("prompt", prompt))
                self.gui_queue.put(("progress", f"Prompt {idx + 1} / {len(PROMPTS)}"))
                self.gui_queue.put(("status", "Speaking…"))

                speech_done = speak(prompt)
                while not speech_done.is_set():
                    if self.session_cancel.is_set():
                        break
                    drain_to_video()
                    time.sleep(0.01)

                if not self.session_cancel.is_set():
                    speech_done = speak("now")
                    t0 = time.time()
                    while (time.time() - t0 < 2.0) or not speech_done.is_set():
                        if self.session_cancel.is_set():
                            break
                        drain_to_video()
                        time.sleep(0.01)

                if self.session_cancel.is_set():
                    break

                self.gui_queue.put(("status", "Capturing snapshot…"))

                prompt_frames = [None] * n
                frame_numbers = [None] * n
                for j in range(n):
                    try:
                        frame, frame_num, _ = oqs[j].get(timeout=5.0)
                        prompt_frames[j] = frame.copy()
                        frame_numbers[j] = frame_num
                        video_writers[j].write(frame)
                        self.preview_frames[j] = frame
                    except queue.Empty:
                        pass

                for j in range(n):
                    if frame_numbers[j] is not None:
                        with open(timestamp_files[j], "a") as f:
                            f.write(f"{frame_numbers[j]} #{prompt}#\n")

                clean = prompt.lower().replace(" ", "_")
                for j in range(n):
                    if prompt_frames[j] is not None:
                        img_fn = os.path.join(
                            output_dir,
                            f"{seed}_{self.eyes[j]}_{idx + 1:02d}_{clean}.png",
                        )
                        cv2.imwrite(img_fn, prompt_frames[j])

                speech_done = speak("captured")
                while not speech_done.is_set():
                    if self.session_cancel.is_set():
                        break
                    drain_to_video()
                    time.sleep(0.01)

        finally:
            for vw in video_writers:
                vw.release()

            if not self.session_cancel.is_set():
                zip_name = _zip_output(seed, output_dir)
                speak("you are done").wait()
                self.gui_queue.put(("status_success", f"Done! → {zip_name}"))
            else:
                self.gui_queue.put(("status", "Session cancelled."))

            self.gui_queue.put(("prompt", ""))
            self.gui_queue.put(("progress", ""))
            self.gui_queue.put(("countdown", ""))
            self.gui_queue.put(("session_done", None))

    def _poll_gui_queue(self):
        for sq in self.status_queues:
            while not sq.empty():
                try:
                    level, msg = sq.get_nowait()
                    if level == "ERROR":
                        self._status_var.set(f"Camera error: {msg}")
                        self._apply_status_style(False)
                except queue.Empty:
                    break

        while not self.gui_queue.empty():
            try:
                msg_type, data = self.gui_queue.get_nowait()
                if msg_type == "status":
                    self._status_var.set(data)
                    self._apply_status_style(False)
                elif msg_type == "status_success":
                    self._status_var.set(data)
                    self._apply_status_style(True)
                    self._open_submission_page()
                elif msg_type == "prompt":
                    self._prompt_var.set(data)
                elif msg_type == "progress":
                    self._progress_var.set(data)
                elif msg_type == "countdown":
                    self._countdown_var.set(data)
                    if data:
                        self._countdown_lbl.place(relx=0.5, rely=0.5, anchor="center")
                    else:
                        self._countdown_lbl.place_forget()
                elif msg_type == "session_done":
                    self.session_running = False
                    self._start_btn.configure(state="normal")
                    self._stop_btn.configure(state="disabled")
            except queue.Empty:
                break
        self.root.after(50, self._poll_gui_queue)

    def _update_previews(self):
        if not self.session_running and self.output_queues:
            for i in range(len(self.output_queues)):
                latest = None
                while not self.output_queues[i].empty():
                    try:
                        frame, _, _ = self.output_queues[i].get_nowait()
                        latest = frame
                    except queue.Empty:
                        break
                if latest is not None:
                    self.preview_frames[i] = latest

        if HAS_PIL:
            for i, lbl in enumerate(self._preview_labels):
                frame = self.preview_frames.get(i)
                if frame is not None:
                    photo = self._frame_to_photo(frame)
                    if photo:
                        lbl.configure(image=photo)
                        lbl.image = photo

        self.root.after(33, self._update_previews)

    def _on_close(self):
        self.session_cancel.set()
        self._stop_cameras()
        self.root.destroy()


if __name__ == "__main__":
    if not HAS_PIL:
        print("[WARNING] Pillow is not installed — camera previews will be disabled.")
        print("          Run:  pip install Pillow")
    if not HAS_SV_TTK:
        print("[WARNING] sv-ttk is not installed — using default Tk theme.")
        print("          Run:  pip install sv-ttk")

    root = tk.Tk()
    app = CollectionApp(root)
    root.mainloop()
