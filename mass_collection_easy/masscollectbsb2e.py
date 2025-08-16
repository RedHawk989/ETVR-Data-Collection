import threading
import queue
import cv2
import time
from colorama import Fore
import os
import sys
import platform
import zipfile
import random
import string
import subprocess
import shutil
import warnings
import numpy as np
import serial
import serial.tools.list_ports
from enum import Enum

# Create a lock for synchronizing access to speech functions
speech_lock = threading.Lock()


def speak(text):
    # Create event to signal when speech is completed
    done_event = threading.Event()

    # Start in a separate thread to avoid blocking
    speech_thread = threading.Thread(
        target=_speak_platform_specific,
        args=(text, done_event),
        daemon=True
    )
    speech_thread.start()
    return done_event


def _speak_platform_specific(text, done_event):
    try:
        with speech_lock:  # Ensure only one speech process at a time
            system = platform.system().lower()

            if system == 'darwin':  # macOS
                # Use macOS built-in say command
                subprocess.run(['say', text], check=False, timeout=10)

            elif system == 'windows':
                # Use PowerShell's speech synthesizer
                ps_cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{text}");'
                subprocess.run(['powershell', '-Command', ps_cmd], check=False, timeout=20)

            elif system == 'linux':
                # Try with espeak if available
                try:
                    subprocess.run(['espeak', text], check=False, timeout=20)
                except FileNotFoundError:
                    # If espeak not found, try festival
                    try:
                        process = subprocess.Popen(['festival', '--tts'], stdin=subprocess.PIPE)
                        process.communicate(input=text.encode())
                    except FileNotFoundError:
                        print(f"{Fore.YELLOW}[TTS] {text}")
            else:
                # Unknown platform, just print
                print(f"{Fore.YELLOW}[TTS] {text}")
    except Exception as e:
        # If speech fails for any reason, fall back to printing
        print(f"{Fore.YELLOW}[TTS] {text}")
        print(f"{Fore.RED}[TTS ERROR] {str(e)}")
    finally:
        # Signal that speech is complete
        done_event.set()


# Initial welcome message
print(
    "Welcome to EyeTrackVR data collection for LEAPv2. Follow the prompts and be sure that each pose is correct to the best of your ability.")
welcome_done = speak(
    "Welcome to EyeTrack V R data collection for Leap version 2. Follow the prompts and be sure that each pose is correct to the best of your ability.")
# Wait for welcome message to complete before continuing
welcome_done.wait()

# Fix color in terminal
os.system("color")

# Constants for serial camera
WAIT_TIME = 0.1
ETVR_HEADER = b"\xff\xa0"
ETVR_HEADER_FRAME = b"\xff\xa1"
ETVR_HEADER_LEN = 6


class CameraState(Enum):
    CONNECTING = 0
    CONNECTED = 1
    DISCONNECTED = 2


def is_serial_capture_source(addr):
    if isinstance(addr, str):
        return addr.lower().startswith("com") or addr.startswith("/dev/cu") or addr.startswith("/dev/tty")
    return False


class SplitUVCCamera:

    def __init__(self, capture_source, cancellation_event, status_queue, left_output_queue, right_output_queue):
        self.capture_source = capture_source
        self.cancellation_event = cancellation_event
        self.status_queue = status_queue
        self.left_output_queue = left_output_queue
        self.right_output_queue = right_output_queue
        self.frame_count = 0
        self.fps = 0
        self.cap = None
        self.total_frames = 0  # Track total frames captured

    def run(self):
        try:
            # Convert source to integer if numeric
            if isinstance(self.capture_source, str) and self.capture_source.isdigit():
                self.capture_source = int(self.capture_source)

            # Try different backends for better compatibility
            backends_to_try = [
                cv2.CAP_DSHOW,  # DirectShow (Windows)
                cv2.CAP_V4L2,  # Video4Linux (Linux)
                cv2.CAP_AVFOUNDATION,  # AVFoundation (macOS)
                cv2.CAP_ANY  # Let OpenCV choose
            ]

            self.cap = None
            for backend in backends_to_try:
                try:
                    self.status_queue.put(("INFO", f"Trying backend: {backend}"))
                    temp_cap = cv2.VideoCapture(self.capture_source, backend)
                    if temp_cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = temp_cap.read()
                        if ret and test_frame is not None:
                            self.cap = temp_cap
                            self.status_queue.put(("INFO", f"Successfully opened camera with backend: {backend}"))
                            break
                        else:
                            temp_cap.release()
                    else:
                        temp_cap.release()
                except Exception as e:
                    self.status_queue.put(("WARNING", f"Backend {backend} failed: {str(e)}"))
                    continue

            if self.cap is None or not self.cap.isOpened():
                self.status_queue.put(("ERROR", "Failed to open camera with any backend"))
                return

            # Set camera properties for better performance - do this after successful open
            try:
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Get current resolution
                current_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                current_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.status_queue.put(("INFO", f"Camera resolution: {current_width}x{current_height}"))


            except Exception as e:
                self.status_queue.put(("WARNING", f"Could not set camera properties: {str(e)}"))

            self.status_queue.put(("INFO", "Split UVC camera initialized successfully"))
            failed_reads = 0
            max_failed = 10
            start = time.time()
            fps_counter = 0
            fps_start = time.time()

            while not self.cancellation_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    failed_reads += 1
                    self.status_queue.put(("WARNING", f"Failed to grab frame ({failed_reads}/{max_failed})"))
                    if failed_reads >= max_failed:
                        self.status_queue.put(("WARNING", "Resetting camera"))
                        self.cap.release()
                        time.sleep(1)
                        self.cap = cv2.VideoCapture(self.capture_source)
                        failed_reads = 0
                    time.sleep(0.1)
                    continue

                failed_reads = 0

                # Get frame dimensions
                height, width = frame.shape[:2]

                # Split frame down the middle
                mid_point = width // 2
                left_frame = frame[:, :mid_point]  # Left half
                right_frame = frame[:, mid_point:]  # Right half

                # Resize each half to 240x240 and convert to grayscale
                left_frame = cv2.resize(left_frame, (240, 240))
                right_frame = cv2.resize(right_frame, (240, 240))

                if len(left_frame.shape) > 2:
                    left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
                if len(right_frame.shape) > 2:
                    right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

                self.total_frames += 1
                fps_counter += 1

                # Calculate FPS
                fps_elapsed = time.time() - fps_start
                if fps_elapsed >= 1.0:
                    self.fps = fps_counter / fps_elapsed
                    fps_counter = 0
                    fps_start = time.time()

                # Clear old frames from queues to prevent backlog
                self._clear_queue(self.left_output_queue)
                self._clear_queue(self.right_output_queue)

                # Put frames in respective queues
                try:
                    self.left_output_queue.put((left_frame, self.total_frames, self.fps), timeout=0.01)
                except queue.Full:
                    pass  # Drop frame if queue is full

                try:
                    self.right_output_queue.put((right_frame, self.total_frames, self.fps), timeout=0.01)
                except queue.Full:
                    pass  # Drop frame if queue is full

                time.sleep(0.001)  # Small sleep to prevent CPU hogging

        except Exception as e:
            self.status_queue.put(("ERROR", str(e)))
        finally:
            if self.cap is not None:
                self.cap.release()

    def _clear_queue(self, q):
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break


class SerialCamera:

    def __init__(self, capture_source, cancellation_event, capture_event, status_queue, output_queue):
        self.camera_status = CameraState.CONNECTING
        self.capture_source = capture_source
        self.camera_status_outgoing = status_queue
        self.camera_output_outgoing = output_queue
        self.capture_event = capture_event
        self.cancellation_event = cancellation_event
        self.current_capture_source = capture_source
        self.serial_connection = None
        self.last_frame_time = time.time()
        self.frame_number = 0
        self.fps = 0
        self.bps = 0
        self.start = True
        self.buffer = b""
        self.pf_fps = 0
        self.prevft = time.time()
        self.newft = 0
        self.fl = [0]
        self.total_frames = 0  # Track total frames captured
        self.error_message = f"{Fore.YELLOW}[WARN] Serial capture source {{}} not found, retrying...{Fore.RESET}"

    def __del__(self):
        self.cleanup_resources()

    def cleanup_resources(self):
        if self.serial_connection is not None:
            self.serial_connection.close()

    def run(self):
        try:
            while not self.cancellation_event.is_set():
                if self.serial_connection is None or not self.serial_connection.is_open or self.camera_status == CameraState.DISCONNECTED:
                    port = self.capture_source
                    self.current_capture_source = port
                    self.start_serial_connection(port)

                if self.camera_status == CameraState.CONNECTED:
                    self.get_serial_camera_picture()
                else:
                    if self.cancellation_event.wait(WAIT_TIME):
                        break
        except Exception as e:
            self.camera_status_outgoing.put(("ERROR", str(e)))
        finally:
            self.cleanup_resources()

    def start_serial_connection(self, port):
        if self.serial_connection is not None and self.serial_connection.is_open:
            if self.serial_connection.port == port:
                return
            self.serial_connection.close()

        com_ports = [tuple(p) for p in list(serial.tools.list_ports.comports())]
        if not any(p for p in com_ports if port in p):
            self.camera_status_outgoing.put(("WARNING", f"Port {port} not found, retrying..."))
            return

        try:
            rate = 115200 if sys.platform == "darwin" else 3000000
            conn = serial.Serial(baudrate=rate, port=port, xonxoff=False, dsrdtr=False, rtscts=False)
            if sys.platform == "win32":
                buffer_size = 32768
                conn.set_buffer_size(rx_size=buffer_size, tx_size=buffer_size)

            self.camera_status_outgoing.put(("INFO", f"ETVR Serial Tracker device connected on {port}"))
            self.serial_connection = conn
            self.camera_status = CameraState.CONNECTED
        except serial.SerialException as e:
            self.camera_status_outgoing.put(("ERROR", f"Failed to connect on {port}: {str(e)}"))
            self.camera_status = CameraState.DISCONNECTED

    def get_next_packet_bounds(self):
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

    def get_next_jpeg_frame(self):
        beg, end = self.get_next_packet_bounds()
        if beg is None or end is None:
            return None
        jpeg = self.buffer[beg + ETVR_HEADER_LEN: end + ETVR_HEADER_LEN]
        self.buffer = self.buffer[end + ETVR_HEADER_LEN:]
        return jpeg

    def get_serial_camera_picture(self):
        if self.cancellation_event.is_set():
            return

        conn = self.serial_connection
        if conn is None or not conn.is_open:
            return

        try:
            if conn.in_waiting:
                jpeg = self.get_next_jpeg_frame()
                if jpeg:
                    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        self.camera_status_outgoing.put(("WARNING", "Frame drop. Corrupted JPEG."))
                        return

                    if conn.in_waiting >= 32768:
                        conn.reset_input_buffer()
                        self.buffer = b""

                    # Convert to grayscale if needed
                    if len(image.shape) > 2 and image.shape[2] > 1:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Resize to match expected size
                    image = cv2.resize(image, (240, 240))

                    self.frame_number += 1
                    self.total_frames += 1

                    current_frame_time = time.time()
                    if current_frame_time != self.prevft:
                        self.fps = 1 / (current_frame_time - self.prevft)
                        self.prevft = current_frame_time
                        if len(self.fl) < 60:
                            self.fl.append(self.fps)
                        else:
                            self.fl.pop(0)
                            self.fl.append(self.fps)
                        self.fps = sum(self.fl) / len(self.fl)

                    # Clear old frame
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
            self.camera_status_outgoing.put(("WARNING", f"Serial capture source problem: {str(e)}"))
            if conn.is_open:
                conn.close()
            self.camera_status = CameraState.DISCONNECTED
        except OSError as e:
            self.camera_status_outgoing.put(("WARNING", f"OS error: {str(e)}"))
            if conn.is_open:
                conn.close()
            self.camera_status = CameraState.DISCONNECTED


def get_best_codec():
    system = platform.system().lower()

    # For Windows, prioritize codecs known to work well
    if system == 'windows':
        # Use XVID for AVI (very reliable on Windows)
        # Use AVC1 for MP4 (more compatible than H264 tag)
        codecs_to_try = [
            ('XVID', 'avi'),  # XVID codec with AVI container
            ('avc1', 'mp4'),  # AVC1 tag for H.264 in MP4 (Windows compatible)
            ('DIVX', 'avi'),  # DIVX as fallback
            ('MJPG', 'avi')  # MJPG as last resort (larger files but reliable)
        ]
    else:  # For macOS and Linux
        codecs_to_try = [
            ('avc1', 'mp4'),  # Try AVC1 first (H.264 equivalent)
            ('H264', 'mp4'),  # Then try explicit H264
            ('XVID', 'avi'),  # XVID as fallback
            ('MJPG', 'avi')  # MJPG as last resort
        ]

    # Test each codec to find the first available one
    for codec, container in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            # Create a dummy video writer to test if codec works
            temp_file = os.path.join(os.getcwd(), f"temp_test_{codec}.{container}")
            test_writer = cv2.VideoWriter(temp_file, fourcc, 30, (240, 240), False)
            is_opened = test_writer.isOpened()
            test_writer.release()

            # Remove the temp file if created
            if os.path.exists(temp_file):
                os.remove(temp_file)

            if is_opened:
                print(f"{Fore.GREEN}[INFO] Using {codec} codec with {container} container for video compression")
                return fourcc, codec, container
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Codec {codec} with {container} not available: {str(e)}")

    # If all codecs fail, use MJPG as a last resort
    print(f"{Fore.YELLOW}[WARNING] Falling back to MJPG codec with AVI container")
    return cv2.VideoWriter_fourcc(*'MJPG'), 'MJPG', 'avi'


def list_available_cameras():

    available_cameras = []
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
                height, width = frame.shape[:2]
                print(f"{Fore.GREEN}[INFO] Camera {i}: Available ({width}x{height})")
            cap.release()
        else:
            # Try with different backends
            for backend in [cv2.CAP_DSHOW, cv2.CAP_V4L2, cv2.CAP_AVFOUNDATION]:
                try:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            available_cameras.append(i)
                            height, width = frame.shape[:2]
                            print(f"{Fore.GREEN}[INFO] Camera {i}: Available with backend {backend} ({width}x{height})")
                            cap.release()
                            break
                    cap.release()
                except:
                    continue

    if not available_cameras:
        print(f"{Fore.RED}[ERROR] No cameras found!")
    return available_cameras

    system = platform.system().lower()

    # For Windows, prioritize codecs known to work well
    if system == 'windows':
        # Use XVID for AVI (very reliable on Windows)
        # Use AVC1 for MP4 (more compatible than H264 tag)
        codecs_to_try = [
            ('XVID', 'avi'),  # XVID codec with AVI container
            ('avc1', 'mp4'),  # AVC1 tag for H.264 in MP4 (Windows compatible)
            ('DIVX', 'avi'),  # DIVX as fallback
            ('MJPG', 'avi')  # MJPG as last resort (larger files but reliable)
        ]
    else:  # For macOS and Linux
        codecs_to_try = [
            ('avc1', 'mp4'),  # Try AVC1 first (H.264 equivalent)
            ('H264', 'mp4'),  # Then try explicit H264
            ('XVID', 'avi'),  # XVID as fallback
            ('MJPG', 'avi')  # MJPG as last resort
        ]

    # Test each codec to find the first available one
    for codec, container in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            # Create a dummy video writer to test if codec works
            temp_file = os.path.join(os.getcwd(), f"temp_test_{codec}.{container}")
            test_writer = cv2.VideoWriter(temp_file, fourcc, 30, (240, 240), False)
            is_opened = test_writer.isOpened()
            test_writer.release()

            # Remove the temp file if created
            if os.path.exists(temp_file):
                os.remove(temp_file)

            if is_opened:
                print(f"{Fore.GREEN}[INFO] Using {codec} codec with {container} container for video compression")
                return fourcc, codec, container
        except Exception as e:
            print(f"{Fore.YELLOW}[WARNING] Codec {codec} with {container} not available: {str(e)}")

    # If all codecs fail, use MJPG as a last resort
    print(f"{Fore.YELLOW}[WARNING] Falling back to MJPG codec with AVI container")
    return cv2.VideoWriter_fourcc(*'MJPG'), 'MJPG', 'avi'


def main(capture_source):
    # Filter OpenCV warnings about codec fallbacks
    warnings.filterwarnings("ignore", category=UserWarning)

    seed = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    print(f"{Fore.CYAN}[INFO] Run seed: {seed}")

    output_dir = "eye_captures"

    # Clear out any old captures
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Initialize timestamp files for both eyes
    eyes = ['l', 'r']
    timestamp_files = []
    for eye in eyes:
        timestamp_file = os.path.join(output_dir, f"{seed}_{eye}_timestamps.txt")
        with open(timestamp_file, 'w') as f:
            f.write("# Format: <frame_number> <prompt_text>\n")
            f.write("# Recorded on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("# Seed: " + seed + "\n\n")
        timestamp_files.append(timestamp_file)

    # Prepare thread and queues for split camera
    cancel_event = threading.Event()
    status_queue = queue.Queue()
    left_output_queue = queue.Queue(maxsize=2)
    right_output_queue = queue.Queue(maxsize=2)

    # Create split UVC camera
    if is_serial_capture_source(capture_source):
        print(f"{Fore.RED}[ERROR] Serial cameras are not supported for split mode. Use a UVC camera.")
        return

    print(f"{Fore.CYAN}[INFO] Using split UVC camera for source: {capture_source}")
    cam = SplitUVCCamera(capture_source, cancel_event, status_queue, left_output_queue, right_output_queue)

    # Start camera thread
    camera_thread = threading.Thread(target=cam.run)
    camera_thread.start()

    # Warm-up and get first frames
    first_frames = [None, None]  # [left, right]
    output_queues = [left_output_queue, right_output_queue]
    timeout = 10

    for i, eye in enumerate(eyes):
        start = time.time()
        print(f"{Fore.YELLOW}Waiting for {eye} eye camera to warm up...")
        while first_frames[i] is None and time.time() - start < timeout:
            while not status_queue.empty():
                t, msg = status_queue.get()
                print(f"{Fore.YELLOW}[{t}] {msg}")
            try:
                if not output_queues[i].empty():
                    frame, _, _ = output_queues[i].get(timeout=0.1)
                    first_frames[i] = frame
                    break
            except queue.Empty:
                time.sleep(0.1)
        if first_frames[i] is None:
            print(f"{Fore.RED}Failed to init {eye} eye camera. Exiting.")
            cancel_event.set()
            camera_thread.join()
            return
        print(f"{Fore.GREEN}{eye.upper()} eye camera initialized.")

    # Determine the best codec and container format for the current platform
    fourcc, codec_name, container_format = get_best_codec()

    # Setup video writers for both eyes
    video_writers = []
    for i, eye in enumerate(eyes):
        h, w = first_frames[i].shape[:2]
        fn = os.path.join(output_dir, f"{seed}_full_session_{eye}.{container_format}")
        # Use original dimensions, no compression
        vw = cv2.VideoWriter(fn, fourcc, 60, (w, h), False)  # 30 FPS, original size, grayscale
        video_writers.append(vw)

    prompts = [
        "Look left", "Look left and squint", "Look right", "Look right and squint",
        "Look up", "Look up and squint", "Look down", "Look down and squint",
        "Look top-left", "Look top-right", "Look bottom-left", "Look bottom-right",
        "Look straight", "Look straight and squint", "Close your eyes",
        "Widen your eyes and look straight", "Widen your eyes and look left",
        "Widen your eyes and look right",
        "Raise eyebrows fully and look forward",
        "Raise eyebrows halfway and look forward",
        "Lower eyebrows fully and look forward",
        "Lower eyebrows halfway and look forward",
        "Raise eyebrows fully and look in random direction",
        "Lower eyebrows fully and look in random direction",
        "Close your eyes and look in random direction",
        "Alternate between left and right quickly 2 times starting now",
        "Look in a full circle starting now"
    ]

    try:
        for idx, prompt in enumerate(prompts):
            print(f"{Fore.CYAN}[PROMPT {idx + 1}/{len(prompts)}] {prompt}")

            # Start speaking the prompt and get the event
            speech_done = speak(f"{prompt}")

            # While speaking the prompt, continue capturing frames
            while not speech_done.is_set():
                # Process camera frames while waiting for speech to complete
                for j in range(2):  # 2 eyes
                    if not output_queues[j].empty():
                        try:
                            frame, frame_num, fps = output_queues[j].get_nowait()
                            # Write frame without compression
                            video_writers[j].write(frame)
                            cv2.imshow(f'Camera {eyes[j].upper()}', frame)
                        except queue.Empty:
                            pass

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
                time.sleep(0.01)  # Small sleep to prevent CPU hogging

            # Countdown
            for i in range(2, 0, -1):
                print(f"{Fore.YELLOW}Capturing in {i}...")

                # Start speaking the count and get the event
                speech_done = speak(f"{i}")

                t0 = time.time()
                # Continue capturing frames during the 1-second countdown
                while (time.time() - t0 < 1.0) or not speech_done.is_set():
                    for j in range(2):  # 2 eyes
                        if not output_queues[j].empty():
                            try:
                                frame, frame_num, fps = output_queues[j].get_nowait()
                                # Write frame without compression
                                video_writers[j].write(frame)
                                cv2.imshow(f'Camera {eyes[j].upper()}', frame)
                            except queue.Empty:
                                pass
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                    time.sleep(0.01)  # Small sleep to prevent CPU hogging

            print(f"{Fore.GREEN}Capturing now!")

            # Capture one prompt frame per eye (blocks up to 5 s)
            prompt_frames = [None, None]
            frame_numbers = [None, None]

            for j in range(2):  # 2 eyes
                try:
                    frame, frame_num, fps = output_queues[j].get(timeout=5.0)
                    prompt_frames[j] = frame.copy()
                    frame_numbers[j] = frame_num  # Store the frame number
                    video_writers[j].write(frame)
                except queue.Empty:
                    print(f"{Fore.RED}[WARNING] Could not capture frame for prompt '{prompt}' ({eyes[j]})")

            # Write timestamps for both eyes
            for j in range(2):
                if frame_numbers[j] is not None:
                    with open(timestamp_files[j], 'a') as f:
                        f.write(f"{frame_numbers[j]} #{prompt}#\n")
                    print(f"{Fore.CYAN}[TIMESTAMP] {eyes[j].upper()} - Frame {frame_numbers[j]}: {prompt}")

            # Save images without compression
            clean = prompt.lower().replace(' ', '_')
            for j in range(2):
                if prompt_frames[j] is not None:
                    img_fn = os.path.join(output_dir, f"{seed}_{eyes[j]}_{idx + 1:02d}_{clean}.png")
                    # Save images without compression as PNG
                    cv2.imwrite(img_fn, prompt_frames[j])
                    print(f"{Fore.GREEN}[INFO] Saved {eyes[j]} frame: {img_fn}")
                else:
                    print(f"{Fore.RED}[WARNING] No frame for '{prompt}' ({eyes[j]})")

            # Speak "captured" and continue recording while speaking
            speech_done = speak("captured")
            while not speech_done.is_set():
                # Process camera frames while waiting for speech to complete
                for j in range(2):  # 2 eyes
                    if not output_queues[j].empty():
                        try:
                            frame, frame_num, fps = output_queues[j].get_nowait()
                            # Write frame without compression
                            video_writers[j].write(frame)
                            cv2.imshow(f'Camera {eyes[j].upper()}', frame)
                        except queue.Empty:
                            pass
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
                time.sleep(0.01)  # Small sleep to prevent CPU hogging

    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}[INFO] Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        print(f"{Fore.CYAN}Thank you for contributing <3")
        print(f"{Fore.CYAN}Files saved in '{output_dir}'")

        # Clean up camera thread
        cancel_event.set()
        camera_thread.join()

        for vw in video_writers:
            vw.release()

        # ZIP output
        zip_name = f"{seed}_ETVR_User_Data_Output.zip"
        print(f"{Fore.CYAN}[INFO] Creating archive {zip_name}…")
        with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    filepath = os.path.join(root, file)
                    # Add file without including the folder name in the archive
                    arcname = os.path.relpath(filepath, start=output_dir)
                    zipf.write(filepath, arcname)
        print(f"{Fore.GREEN}[INFO] Created archive: {zip_name}")


if __name__ == "__main__":
    print(f"{Fore.CYAN}[INFO] Scanning for available cameras...")
    available_cameras = list_available_cameras()

    if not available_cameras:
        print(f"{Fore.RED}[ERROR] No cameras detected. Please check:")
        print(f"{Fore.YELLOW}1. Camera is properly connected")
        print(f"{Fore.YELLOW}2. Camera drivers are installed")
        print(f"{Fore.YELLOW}3. Camera is not being used by another application")
        print(f"{Fore.YELLOW}4. Camera permissions are granted")
        sys.exit(1)

    print(f"{Fore.CYAN}[INFO] Available cameras: {available_cameras}")

    src_input = input('Enter UVC camera address from the list above (e.g. 0 or 1): ')

    # Convert to integer if numeric
    if src_input.isdigit():
        source = int(src_input)
        if source not in available_cameras:
            print(f"{Fore.YELLOW}[WARNING] Camera {source} was not detected as available, but trying anyway...")
    else:
        # Handle URL or other sources
        if not (src_input.lower().startswith("http://") or src_input.lower().startswith("https://")):
            src_input = "http://" + src_input
        source = src_input

    print(f"{Fore.CYAN}[INFO] Using single camera with split view for both eyes")
    main(source)