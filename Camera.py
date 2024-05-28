import cv2
import numpy as np
import queue
import serial
import serial.tools.list_ports
import threading
import time
import platform
from colorama import Fore
from enum import Enum
import sys

WAIT_TIME = 0.1
ETVR_HEADER = b"\xff\xa0"
ETVR_HEADER_FRAME = b"\xff\xa1"
ETVR_HEADER_LEN = 6


class CameraState(Enum):
    CONNECTING = 0
    CONNECTED = 1
    DISCONNECTED = 2


def is_serial_capture_source(addr: str) -> bool:
    return (addr.startswith("COM") or addr.startswith("/dev/cu") or addr.startswith("/dev/tty"))


class Camera:
    def __init__(self, capture_source: str, cancellation_event: threading.Event, capture_event: threading.Event, camera_status_outgoing: queue.Queue, camera_output_outgoing: queue.Queue):
        self.camera_status = CameraState.CONNECTING
        self.capture_source = capture_source
        self.camera_status_outgoing = camera_status_outgoing
        self.camera_output_outgoing = camera_output_outgoing
        self.capture_event = capture_event
        self.cancellation_event = cancellation_event
        self.current_capture_source = capture_source
        self.cv2_camera = None
        self.serial_connection = None
        self.last_frame_time = time.time()
        self.frame_number = 0
        self.fps = 0
        self.bps = 0
        self.start = True
        self.buffer = b""
        self.pf_fps = 0
        self.prevft = 0
        self.newft = 0
        self.fl = [0]
        self.error_message = f"{Fore.YELLOW}[WARN] Capture source {{}} not found, retrying...{Fore.RESET}"

    def __del__(self):
        if self.serial_connection is not None:
            self.serial_connection.close()

    def run(self):

        OPENCV_PARAMS = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]
        while True:

            if self.cancellation_event.is_set():
                print(f"{Fore.CYAN}[INFO] Exiting Capture thread{Fore.RESET}")
                return

            should_push = True
            if self.capture_source:
                addr = str(self.current_capture_source)
                if is_serial_capture_source(addr):
                    if self.serial_connection is None or self.camera_status == CameraState.DISCONNECTED or self.capture_source != self.current_capture_source:
                        port = self.capture_source
                        self.current_capture_source = port
                        self.start_serial_connection(port)
                else:
                    if self.cv2_camera is None or not self.cv2_camera.isOpened() or self.camera_status == CameraState.DISCONNECTED or self.capture_source != self.current_capture_source:
                        print(self.error_message.format(self.capture_source))
                        if self.cancellation_event.wait(WAIT_TIME):
                            return
                        self.current_capture_source = self.capture_source
                        self.cv2_camera = cv2.VideoCapture()
                        self.cv2_camera.setExceptionMode(True)
                        self.cv2_camera.open(self.current_capture_source)
                        should_push = False

            else:
                if self.cancellation_event.wait(WAIT_TIME):
                    self.camera_status = CameraState.DISCONNECTED
                    return

            if should_push and not self.capture_event.wait(timeout=0.001):
                continue

            if self.capture_source:
                addr = str(self.current_capture_source)
                if is_serial_capture_source(addr):
                    self.get_serial_camera_picture(should_push)
                else:
                    self.get_cv2_camera_picture(should_push)
                if not should_push:
                    self.camera_status = CameraState.CONNECTED

    def get_cv2_camera_picture(self, should_push):
        try:
            ret, image = self.cv2_camera.read()
            height, width = image.shape[:2]
            if int(width) > 680:
                aspect_ratio = float(width) / float(height)
                new_height = int(680 / aspect_ratio)
                image = cv2.resize(image, (680, new_height))
            if not ret:
                self.cv2_camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
                raise RuntimeError("Problem while getting frame")
            frame_number = self.cv2_camera.get(cv2.CAP_PROP_POS_FRAMES)
            current_frame_time = time.time()
            delta_time = current_frame_time - self.last_frame_time
            self.last_frame_time = current_frame_time
            if delta_time > 0:
                self.bps = len(image) / delta_time
            self.frame_number = self.frame_number + 1
            self.fps = (self.fps + self.pf_fps) / 2
            self.newft = time.time()
            self.fps = 1 / (self.newft - self.prevft)
            self.prevft = self.newft
            self.fps = int(self.fps)
            if len(self.fl) < 60:
                self.fl.append(self.fps)
            else:
                self.fl.pop(0)
                self.fl.append(self.fps)
            self.fps = sum(self.fl) / len(self.fl)
            if should_push:
                self.push_image_to_queue(image, frame_number, self.fps)
        except:
            print(f"{Fore.YELLOW}[WARN] Capture source problem, assuming camera disconnected, waiting for reconnect.{Fore.RESET}")
            self.camera_status = CameraState.DISCONNECTED
            pass

    def get_next_packet_bounds(self):
        beg = -1
        while beg == -1:
            self.buffer += self.serial_connection.read(2048)
            beg = self.buffer.find(ETVR_HEADER + ETVR_HEADER_FRAME)
        if beg > 0:
            self.buffer = self.buffer[beg:]
            beg = 0
        end = int.from_bytes(self.buffer[4:6], signed=False, byteorder="little")
        self.buffer += self.serial_connection.read(end - len(self.buffer))
        return beg, end

    def get_next_jpeg_frame(self):
        beg, end = self.get_next_packet_bounds()
        jpeg = self.buffer[beg + ETVR_HEADER_LEN : end + ETVR_HEADER_LEN]
        self.buffer = self.buffer[end + ETVR_HEADER_LEN :]
        return jpeg

    def get_serial_camera_picture(self, should_push):
        conn = self.serial_connection
        if conn is None:
            return
        try:
            if conn.in_waiting:
                jpeg = self.get_next_jpeg_frame()
                if jpeg:
                    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        print(f"{Fore.YELLOW}[WARN] Frame drop. Corrupted JPEG.{Fore.RESET}")
                        return
                    if conn.in_waiting >= 32768:
                        print(f"{Fore.CYAN}[INFO] Discarding the serial buffer ({conn.in_waiting} bytes){Fore.RESET}")
                        conn.reset_input_buffer()
                        self.buffer = b""
                    current_frame_time = time.time()
                    delta_time = current_frame_time - self.last_frame_time
                    self.last_frame_time = current_frame_time
                    if delta_time > 0:
                        self.bps = len(jpeg) / delta_time
                    self.fps = (self.fps + self.pf_fps) / 2
                    self.newft = time.time()
                    self.fps = 1 / (self.newft - self.prevft)
                    self.prevft = self.newft
                    self.fps = int(self.fps)
                    if len(self.fl) < 60:
                        self.fl.append(self.fps)
                    else:
                        self.fl.pop(0)
                        self.fl.append(self.fps)
                    self.fps = sum(self.fl) / len(self.fl)
                    self.frame_number = self.frame_number + 1
                    if should_push:
                        self.push_image_to_queue(image, self.frame_number, self.fps)
        except Exception:
            print(f"{Fore.YELLOW}[WARN] Serial capture source problem, assuming camera disconnected, waiting for reconnect.{Fore.RESET}")
            conn.close()
            self.camera_status = CameraState.DISCONNECTED
            pass

    def start_serial_connection(self, port):
        if self.serial_connection is not None and self.serial_connection.is_open:
            if self.serial_connection.port == port:
                return
            self.serial_connection.close()
        com_ports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
        if not any(p for p in com_ports if port in p):
            return
        try:
            rate = 115200 if sys.platform == "darwin" else 3000000
            conn = serial.Serial(baudrate=rate, port=port, xonxoff=False, dsrdtr=False, rtscts=False)
            if sys.platform == "win32":
                buffer_size = 32768
                conn.set_buffer_size(rx_size=buffer_size, tx_size=buffer_size)
            print(f"{Fore.CYAN}[INFO] ETVR Serial Tracker device connected on {port}{Fore.RESET}")
            self.serial_connection = conn
            self.camera_status = CameraState.CONNECTED
        except Exception:
            print(f"{Fore.CYAN}[INFO] Failed to connect on {port}{Fore.RESET}")
            self.camera_status = CameraState.DISCONNECTED

    def push_image_to_queue(self, image, frame_number, fps):
        qsize = self.camera_output_outgoing.qsize()
        if qsize > 1:
            print(f"{Fore.YELLOW}[WARN] CAPTURE QUEUE BACKPRESSURE OF {qsize}. CHECK FOR CRASH OR TIMING ISSUES IN ALGORITHM.{Fore.RESET}")
        self.camera_output_outgoing.put((image, frame_number, fps))
        self.capture_event.clear()
