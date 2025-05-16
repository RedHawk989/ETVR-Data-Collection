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
    """
    Platform-independent text-to-speech function that returns when speech is complete.
    Falls back to print if speech fails.
    """
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
    """Internal function to handle platform-specific TTS"""
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
    """Check if capture source is a serial port"""
    if isinstance(addr, str):
        return addr.startswith("COM") or addr.startswith("/dev/cu") or addr.startswith("/dev/tty")
    return False


class SerialCamera:
    """Camera class for handling serial camera connections (COM ports)"""

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


class CV2Camera:
    """Camera class for handling standard webcams via OpenCV"""

    def __init__(self, capture_source, cancellation_event, capture_event, status_queue, output_queue):
        self.capture_source = capture_source
        self.cancellation_event = cancellation_event
        self.capture_event = capture_event
        self.status_queue = status_queue
        self.output_queue = output_queue
        self.frame_count = 0
        self.fps = 0
        self.cap = None
        self.total_frames = 0  # Track total frames captured

    def run(self):
        try:
            # Convert source to integer if numeric
            if isinstance(self.capture_source, str) and self.capture_source.isdigit():
                self.capture_source = int(self.capture_source)

            # Open camera
            self.cap = cv2.VideoCapture(self.capture_source)

            if not self.cap.isOpened():
                self.status_queue.put(("ERROR", "Failed to open camera"))
                return

            self.status_queue.put(("INFO", "Camera initialized successfully"))
            failed_reads = 0
            max_failed = 10
            start = time.time()

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
                    time.sleep(0.2)
                    continue

                failed_reads = 0
                frame = cv2.resize(frame, (240, 240))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.frame_count += 1
                self.total_frames += 1  # Increment total frame counter
                elapsed = time.time() - start
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    start = time.time()

                # Clear old frame
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
    """
    Returns the best available codec for the current platform
    """
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


def main(capture_sources, eyes):
    # Filter OpenCV warnings about codec fallbacks
    warnings.filterwarnings("ignore", category=UserWarning)

    # One or two cameras
    n = len(capture_sources)
    seed = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    print(f"{Fore.CYAN}[INFO] Run seed: {seed}")

    output_dir = "eye_captures"

    # ——— clear out any old captures ———
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Initialize timestamps file
    timestamps_file = os.path.join(output_dir, "timestamps.txt")
    with open(timestamps_file, 'w') as f:
        f.write("# Format: <frame_number> <prompt_text>\n")
        f.write("# Recorded on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("# Seed: " + seed + "\n\n")

    # Prepare threads and queues
    cancel_events = []
    status_queues = []
    output_queues = []
    threads = []

    for i in range(n):
        ce = threading.Event()
        cancel_events.append(ce)
        sq = queue.Queue()
        oq = queue.Queue(maxsize=1)
        status_queues.append(sq)
        output_queues.append(oq)

        # Create appropriate camera type based on source
        if is_serial_capture_source(capture_sources[i]):
            print(f"{Fore.CYAN}[INFO] Using serial camera for source: {capture_sources[i]}")
            cam = SerialCamera(capture_sources[i], ce, None, sq, oq)
        else:
            print(f"{Fore.CYAN}[INFO] Using CV2 camera for source: {capture_sources[i]}")
            cam = CV2Camera(capture_sources[i], ce, None, sq, oq)

        th = threading.Thread(target=cam.run)
        threads.append(th)
        th.start()

    # Warm-up and get first frames
    first_frames = [None] * n
    timeout = 10
    for i in range(n):
        start = time.time()
        print(f"{Fore.YELLOW}Waiting for camera '{eyes[i]}' to warm up...")
        while first_frames[i] is None and time.time() - start < timeout:
            while not status_queues[i].empty():
                t, msg = status_queues[i].get()
                print(f"{Fore.YELLOW}[{t}] {msg}")
            try:
                if not output_queues[i].empty():
                    frame, _, _ = output_queues[i].get(timeout=0.1)
                    first_frames[i] = frame
                    break
            except queue.Empty:
                time.sleep(0.1)
        if first_frames[i] is None:
            print(f"{Fore.RED}Failed to init camera '{eyes[i]}'. Exiting.")
            for ev in cancel_events:
                ev.set()
            for th in threads:
                th.join()
            return
        print(f"{Fore.GREEN}Camera '{eyes[i]}' initialized.")

    # Determine the best codec and container format for the current platform
    fourcc, codec_name, container_format = get_best_codec()

    # Setup video writers
    video_writers = []
    for i in range(n):
        h, w = first_frames[i].shape[:2]
        fn = os.path.join(output_dir, f"{seed}_full_session_{eyes[i]}.{container_format}")

        # Create video writer with specific parameters for better compression
        vw = cv2.VideoWriter(fn, fourcc, 60, (w, h), False)

        # Set additional properties for better compression if available

        video_writers.append(vw)

    prompts = [
        "Look left", "Look left and squint", "Look left and half blink", "Look right", "Look right and squint",
        "Look right and half blink",
        "Look up", "Look up and squint", "Look up and half blink", "Look down", "Look down and squint",
        "Look down and half blink",
        "Look top-left", "Look top-right", "Look bottom-left", "Look bottom-right",
        "Look straight", "Look straight and squint", "Look straight and half blink",
        "Close your eyes", "Squeeze your eyes closed",
        "Widen your eyes and look in random direction",
        "Raise eyebrows fully and look straight",
        "Raise eyebrows halfway and look straight",
        "Lower eyebrows fully and look straight",
        "Lower eyebrows halfway and look straight",
        "Raise eyebrows fully and look in random direction",
        "Raise eyebrows halfway and look in random direction",
        "Lower eyebrows fully and look in random direction",
        "Lower eyebrows halfway and look in random direction",
        "Do something random and look in random direction",
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
                for j in range(n):
                    if not output_queues[j].empty():
                        frame, frame_num, _ = output_queues[j].get()

                        # Add frame compression if needed
                        if codec_name in ['MJPG', 'DIVX', 'XVID']:
                            # Apply some compression to the frame before writing
                            # Only if we're not using H.264/AVC1 which is already efficient
                            _, compressed = cv2.imencode('.jpg', frame)
                            frame = cv2.imdecode(compressed, cv2.IMREAD_GRAYSCALE)

                        video_writers[j].write(frame)
                        cv2.imshow(f'Camera {eyes[j].upper()}', frame)
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
                # This combines the wait for speech and the 1-second timer
                while (time.time() - t0 < 1.0) or not speech_done.is_set():
                    for j in range(n):
                        if not output_queues[j].empty():
                            frame, frame_num, _ = output_queues[j].get()
                            video_writers[j].write(frame)
                            cv2.imshow(f'Camera {eyes[j].upper()}', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                    time.sleep(0.01)  # Small sleep to prevent CPU hogging
            print(f"{Fore.GREEN}Capturing now!")

            # Capture one prompt frame per eye (blocks up to 5 s)
            prompt_frames = [None] * n
            frame_numbers = [None] * n
            for j in range(n):
                try:
                    frame, frame_num, _ = output_queues[j].get(timeout=5.0)
                    prompt_frames[j] = frame.copy()
                    frame_numbers[j] = frame_num  # Store the frame number
                    video_writers[j].write(frame)
                except queue.Empty:
                    print(f"{Fore.RED}[WARNING] Could not capture frame for prompt '{prompt}' ({eyes[j]})")

            # Record the timestamp (using the first camera's frame number if available)
            timestamp_frame = next((frame_numbers[i] for i in range(n) if frame_numbers[i] is not None), None)
            if timestamp_frame is not None:
                with open(timestamps_file, 'a') as f:
                    f.write(f"{timestamp_frame} #{prompt}#\n")
                print(f"{Fore.CYAN}[TIMESTAMP] Frame {timestamp_frame}: {prompt}")

            # Save images
            clean = prompt.lower().replace(' ', '_')
            for j in range(n):
                if prompt_frames[j] is not None:
                    img_fn = os.path.join(output_dir, f"{seed}_{eyes[j]}_{idx + 1:02d}_{clean}.jpeg")
                    # Save images with compression
                    cv2.imwrite(img_fn, prompt_frames[j])
                    print(f"{Fore.GREEN}[INFO] Saved {eyes[j]} frame: {img_fn}")
                else:
                    print(f"{Fore.RED}[WARNING] No frame for '{prompt}' ({eyes[j]})")

            # Speak "captured" and continue recording while speaking
            speech_done = speak("captured")
            while not speech_done.is_set():
                # Process camera frames while waiting for speech to complete
                for j in range(n):
                    if not output_queues[j].empty():
                        frame, frame_num, _ = output_queues[j].get()
                        video_writers[j].write(frame)
                        cv2.imshow(f'Camera {eyes[j].upper()}', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
                time.sleep(0.01)  # Small sleep to prevent CPU hogging


    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}[INFO] Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        print(f"{Fore.CYAN}Thank you for contributing <3")
        print(f"{Fore.CYAN}Files saved in '{output_dir}'")

        # Clean up camera threads
        for ev in cancel_events:
            ev.set()
        for th in threads:
            th.join()
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
    src_input = input('Enter camera address(es) (e.g. 0 or 0,1 or COM5): ')
    parts = [s.strip() for s in src_input.split(',') if s.strip()]
    if len(parts) > 2:
        print(f"{Fore.YELLOW}Only first two will be used.")
        parts = parts[:2]
    sources = []
    for p in parts:
        if p.isdigit() and not is_serial_capture_source(p):
            sources.append(int(p))
        else:
            sources.append(p)

    if len(sources) == 1:
        eye = input('Enter "r" for right cam or "l" for left cam: ').lower()
        if eye not in ['r', 'l']:
            print(f"{Fore.RED}Invalid selection. Defaulting to 'r'")
            eye = 'r'
        eyes = [eye]
    else:
        eyes = ['l', 'r']

    main(sources, eyes)