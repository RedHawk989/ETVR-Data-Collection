import threading
import queue
import cv2
import time
from colorama import Fore
import os
import sys
import zipfile
import random
import string

os.system("color")  # fix color in terminal not working


class Camera:
    def __init__(self, capture_source, cancellation_event, capture_event, status_queue, output_queue):
        self.capture_source = capture_source
        self.cancellation_event = cancellation_event
        self.capture_event = capture_event
        self.status_queue = status_queue
        self.output_queue = output_queue
        self.frame_count = 0
        self.fps = 0
        self.cap = None

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
                self.frame_count += 1
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
                    self.output_queue.put((frame, self.frame_count, self.fps), timeout=0.1)
                    time.sleep(0.01)
                except queue.Full:
                    pass

        except Exception as e:
            self.status_queue.put(("ERROR", str(e)))
        finally:
            if self.cap is not None:
                self.cap.release()


def main(capture_sources, eyes):
    # One or two cameras
    n = len(capture_sources)
    seed = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
    print(f"{Fore.CYAN}[INFO] Run seed: {seed}")

    output_dir = "eye_capture"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        cam = Camera(capture_sources[i], ce, None, sq, oq)
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

    # Setup video writers
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_writers = []
    for i in range(n):
        h, w = first_frames[i].shape[:2]
        fn = os.path.join(output_dir, f"{seed}_full_session_{eyes[i]}.avi")
        vw = cv2.VideoWriter(fn, fourcc, 30, (w, h))
        video_writers.append(vw)

    prompts = [
        "Look left", "Look left and squint", "Look right", "Look right and squint",
        "Look up", "Look up and squint", "Look down", "Look down and squint",
        "Look top-left", "Look top-right", "Look bottom-left", "Look bottom-right",
        "Look straight", "Look straight and squint", "Close your eyes",
        "Look straight and half blink", "Widen your eyes and look in random direction",
        "Raise eyebrows fully and look in random direction",
        "Raise eyebrows halfway and look in random direction",
        "Lower eyebrows fully and look in random direction",
        "Lower eyebrows halfway and look in random direction",
        "Do something random and look in random direction",
    ]

    try:
        for idx, prompt in enumerate(prompts):
            print(f"{Fore.CYAN}[PROMPT {idx+1}/{len(prompts)}] {prompt}")
            # Countdown
            for i in range(5, 0, -1):
                print(f"{Fore.YELLOW}Capturing in {i}...")
                t0 = time.time()
                while time.time() - t0 < 1.0:
                    for j in range(n):
                        if not output_queues[j].empty():
                            frame, _, _ = output_queues[j].get()
                            video_writers[j].write(frame)
                            cv2.imshow(f'Camera {eyes[j].upper()}', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
            print(f"{Fore.GREEN}Capturing now!")

            # Capture one prompt frame per eye (blocks up to 5 s)
            prompt_frames = [None] * n
            for j in range(n):
                try:
                    frame, _, _ = output_queues[j].get(timeout=5.0)
                    prompt_frames[j] = frame.copy()
                    video_writers[j].write(frame)
                except queue.Empty:
                    print(f"{Fore.RED}[WARNING] Could not capture frame for prompt '{prompt}' ({eyes[j]})")


            # Save images
            clean = prompt.lower().replace(' ', '_')
            for j in range(n):
                if prompt_frames[j] is not None:
                    img_fn = os.path.join(output_dir, f"{seed}_{eyes[j]}_{idx+1:02d}_{clean}.jpeg")
                    cv2.imwrite(img_fn, prompt_frames[j])
                    print(f"{Fore.GREEN}[INFO] Saved {eyes[j]} frame: {img_fn}")
                else:
                    print(f"{Fore.RED}[WARNING] No frame for '{prompt}' ({eyes[j]})")

    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}[INFO] Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        print(f"{Fore.CYAN}Thank you for contributing <3")
        print(f"{Fore.CYAN}Files saved in '{output_dir}'")
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
                for fname in files:
                    full = os.path.join(root, fname)
                    arc = os.path.relpath(full, start=os.path.dirname(output_dir))
                    zipf.write(full, arc)
        print(f"{Fore.GREEN}[INFO] Archive created: {zip_name}")


if __name__ == "__main__":
    print(f"{Fore.CYAN}[SYSTEM INFO] OpenCV version: {cv2.__version__}")
    print(f"{Fore.CYAN}[SYSTEM INFO] Python version: {sys.version}")
    # List cameras
    print(f"{Fore.CYAN}[SYSTEM INFO] Checking available cameras...")
    found = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                found.append(i)
                print(f"{Fore.GREEN}[CAMERA FOUND] {i}")
            cap.release()
        time.sleep(0.1)
    if not found:
        print(f"{Fore.YELLOW}[WARNING] No cameras detected")

    src_input = input('Enter camera address(es) (e.g. 0 or 0,1 or COM5): ')
    parts = [s.strip() for s in src_input.split(',') if s.strip()]
    if len(parts) > 2:
        print(f"{Fore.YELLOW}Only first two will be used.")
        parts = parts[:2]
    sources = []
    for p in parts:
        if p.isdigit():
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
