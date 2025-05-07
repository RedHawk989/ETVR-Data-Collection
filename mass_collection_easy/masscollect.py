import threading
import queue
import cv2
import time
from colorama import Fore
import os
import sys
import platform

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
            # Convert source to integer if it's numeric
            if isinstance(self.capture_source, str) and self.capture_source.isdigit():
                self.capture_source = int(self.capture_source)

            # Try to open camera with different backends if needed
            self.cap = cv2.VideoCapture(self.capture_source)

            # Set camera properties for better capture
            if self.cap.isOpened():
                # Try to set resolution to 640x480 for better compatibility
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Set buffering properties
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Check again if camera is open after setting properties
            if not self.cap.isOpened():
                self.status_queue.put(("ERROR", "Failed to open camera"))
                # Try alternative method if it's a numeric source
                if isinstance(self.capture_source, int):
                    alternative_source = f"video{self.capture_source}"
                    self.status_queue.put(("INFO", f"Trying alternative source: {alternative_source}"))
                    self.cap = cv2.VideoCapture(alternative_source)

            # Final check if camera is open
            if not self.cap.isOpened():
                self.status_queue.put(("ERROR", "All attempts to open camera failed"))
                return

            self.status_queue.put(("INFO", "Camera initialized successfully"))

            # Create a counter for frame reading attempts
            failed_reads = 0
            max_failed_reads = 10

            start_time = time.time()
            while not self.cancellation_event.is_set():
                ret, frame = self.cap.read()

                if not ret:
                    failed_reads += 1
                    self.status_queue.put(("WARNING", f"Failed to grab frame ({failed_reads}/{max_failed_reads})"))

                    # If we've failed too many times in a row, try to reset the camera
                    if failed_reads >= max_failed_reads:
                        self.status_queue.put(("WARNING", "Attempting to reset camera connection"))
                        self.cap.release()
                        time.sleep(1)  # Give the camera time to disconnect/reconnect
                        self.cap = cv2.VideoCapture(self.capture_source)
                        failed_reads = 0

                    time.sleep(0.2)
                    continue

                # Successfully read a frame, reset the failed counter
                failed_reads = 0

                # Resize frame to 240x240
                frame = cv2.resize(frame, (240, 240))

                self.frame_count += 1
                elapsed = time.time() - start_time

                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    start_time = time.time()

                # Always clear the queue before putting new frame
                while not self.output_queue.empty():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        break

                # Put new frame in queue
                try:
                    self.output_queue.put((frame, self.frame_count, self.fps), block=True, timeout=0.1)
                    # Small sleep to prevent CPU overuse
                    time.sleep(0.01)
                except queue.Full:
                    # If still full after clearing, just continue
                    pass

        except Exception as e:
            self.status_queue.put(("ERROR", f"Camera error: {str(e)}"))
            import traceback
            self.status_queue.put(("ERROR", traceback.format_exc()))
        finally:
            if self.cap is not None:
                self.cap.release()


def main(capture_source, eye):
    # Create output directory if it doesn't exist
    output_dir = "eye_captures"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cancellation_event = threading.Event()
    capture_event = threading.Event()
    camera_status_outgoing = queue.Queue()
    camera_output_outgoing = queue.Queue(maxsize=1)

    camera = Camera(capture_source, cancellation_event, capture_event, camera_status_outgoing, camera_output_outgoing)
    camera_thread = threading.Thread(target=camera.run)
    camera_thread.start()

    # Video of the entire session
    video_filename = os.path.join(output_dir, f"full_session_{eye}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    video_output = None  # Initialize output later once we know frame dimensions

    # For continuous recording
    all_frames = []

    prompts = [
        "Look left",
        "Look left and squint",
        "Look right",
        "Look right and squint",
        "Look up",
        "Look up and squint",
        "Look down",
        "Look down and squint",
        "Look top-left",
        "Look top-right",
        "Look bottom-left",
        "Look bottom-right",
        "Look straight",
        "Look straight and squint",
        "Close your eyes",
        "Half blink",
        "Raise eyebrows",
        "Blink naturally"
    ]

    print(f"{Fore.YELLOW}Waiting for camera to warm up...")

    # Wait for camera to initialize and get first frame
    first_frame = None
    start_time = time.time()
    timeout = 10  # 10 seconds timeout

    while first_frame is None and time.time() - start_time < timeout:
        # Check camera status
        while not camera_status_outgoing.empty():
            status_type, status_msg = camera_status_outgoing.get()
            print(f"{Fore.YELLOW}[{status_type}] {status_msg}")

        # Try to get a frame
        try:
            if not camera_output_outgoing.empty():
                image, _, _ = camera_output_outgoing.get(timeout=0.1)
                first_frame = image
                # Initialize video writer now that we have the first frame
                h, w = first_frame.shape[:2]
                video_output = cv2.VideoWriter(video_filename, fourcc, 30, (w, h))
                break
        except queue.Empty:
            time.sleep(0.1)

    if first_frame is None:
        print(f"{Fore.RED}Failed to initialize camera or get first frame. Exiting.")
        cancellation_event.set()
        camera_thread.join()
        return

    print(f"{Fore.GREEN}Camera initialized successfully!")

    try:
        # Start continuous recording
        last_capture_time = time.time()

        for idx, prompt in enumerate(prompts):
            print(f"{Fore.CYAN}[PROMPT {idx + 1}/{len(prompts)}] {prompt}")

            # Countdown for preparation
            for i in range(5, 0, -1):
                print(f"{Fore.YELLOW}Capturing in {i}...")

                # Continue recording frames during countdown
                now = time.time()
                while time.time() - now < 1.0:  # Record for about 1 second
                    if not camera_output_outgoing.empty():
                        frame = camera_output_outgoing.get(timeout=0.1)
                        image, frame_number, fps = frame
                        video_output.write(image)  # Write to continuous video
                        all_frames.append(image.copy())  # Store frame for later use

                        # Show current frame
                        cv2.imshow('Camera Frame', image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

            print(f"{Fore.GREEN}Capturing now!")

            # Get the most recent frame for this prompt
            frame_obtained = False
            prompt_frame = None

            # Try harder to get a frame from the camera
            capture_start_time = time.time()
            capture_timeout = 5.0  # Give up after 5 seconds of trying

            print(f"{Fore.YELLOW}[DEBUG] Waiting for camera frame...")

            while not frame_obtained and time.time() - capture_start_time < capture_timeout:
                # Check camera status messages
                while not camera_status_outgoing.empty():
                    status_type, status_msg = camera_status_outgoing.get()
                    print(f"{Fore.YELLOW}[{status_type}] {status_msg}")

                # Try to get the frame
                try:
                    if not camera_output_outgoing.empty():
                        frame = camera_output_outgoing.get(timeout=0.1)
                        image, frame_number, fps = frame

                        # Debug information
                        print(f"{Fore.GREEN}[DEBUG] Got frame with shape: {image.shape}")

                        if video_output is not None:
                            video_output.write(image)  # Write to continuous video
                        all_frames.append(image.copy())  # Store frame for later use

                        # Save this specific frame as the prompt image
                        prompt_frame = image.copy()
                        frame_obtained = True

                        # Show current frame
                        cv2.imshow('Camera Frame', image)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt
                    else:
                        print(f"{Fore.YELLOW}[DEBUG] Camera queue is empty, waiting...")
                        time.sleep(0.2)
                except queue.Empty:
                    print(f"{Fore.YELLOW}[DEBUG] Queue timeout, retrying...")
                    time.sleep(0.2)
                except Exception as e:
                    print(f"{Fore.RED}[ERROR] Exception while getting frame: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    time.sleep(0.2)

            # Save the prompt frame as an image file
            if frame_obtained and prompt_frame is not None:
                # Clean prompt name for filename
                clean_prompt = prompt.lower().replace(' ', '_')
                img_filename = os.path.join(output_dir, f"{eye}_{idx + 1:02d}_{clean_prompt}.png")
                cv2.imwrite(img_filename, prompt_frame)
                print(f"{Fore.GREEN}[INFO] Saved image for prompt '{prompt}' to {img_filename}")
            else:
                print(f"{Fore.RED}[WARNING] Could not capture frame for prompt '{prompt}'")

            # Continue recording for a short time after prompt
            now = time.time()
            while time.time() - now < 1.0:  # Record for about 1 second
                if not camera_output_outgoing.empty():
                    frame = camera_output_outgoing.get(timeout=0.1)
                    image, frame_number, fps = frame
                    video_output.write(image)  # Write to continuous video
                    all_frames.append(image.copy())  # Store frame for later use

                    # Show current frame
                    cv2.imshow('Camera Frame', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt

        print(f"{Fore.CYAN}[INFO] Finished capturing all prompts.")

    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}[INFO] Interrupted by user.")

    finally:
        cv2.destroyAllWindows()
        print(f"{Fore.CYAN}Thank you for contributing <3")
        print(f"{Fore.CYAN}Files saved in the '{output_dir}' directory")
        print(f"{Fore.CYAN}Make sure you DM prohurtz the files generated in the '{output_dir}' directory")
        cancellation_event.set()
        camera_thread.join()
        if video_output is not None:
            video_output.release()
        time.sleep(2)


if __name__ == "__main__":
    # Print diagnostic information
    print(f"{Fore.CYAN}[SYSTEM INFO] OpenCV version: {cv2.__version__}")
    print(f"{Fore.CYAN}[SYSTEM INFO] Python version: {sys.version}")

    # List available cameras
    print(f"{Fore.CYAN}[SYSTEM INFO] Checking available cameras...")
    available_cameras = []

    # Try the first 10 camera indices
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"{Fore.GREEN}[CAMERA FOUND] Camera index {i} is available")
            cap.release()
        time.sleep(0.1)

    if not available_cameras:
        print(f"{Fore.YELLOW}[WARNING] No cameras automatically detected")

    # Get camera input from user
    src = input('Enter camera address (e.g. COM5, 0, or IP stream): ')
    if src.isdigit():
        src = int(src)
        print(f"{Fore.CYAN}[INFO] Using camera index: {src}")
    else:
        print(f"{Fore.CYAN}[INFO] Using camera address: {src}")

    eye = input('Enter "r" for right cam or "l" for left cam: ').lower()
    if eye not in ['r', 'l']:
        print(f"{Fore.RED}Invalid eye selection. Defaulting to 'r'")
        eye = 'r'

    try:
        # Import sys for version info
        import sys

        main(src, eye)
    except Exception as e:
        print(f"{Fore.RED}[FATAL ERROR] {str(e)}")
        import traceback

        print(traceback.format_exc())
        input("Press Enter to exit...")