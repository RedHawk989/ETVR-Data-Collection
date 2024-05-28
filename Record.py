import threading
import queue
import argparse
from Camera import Camera, CameraState

def main(capture_source):
    cancellation_event = threading.Event()
    capture_event = threading.Event()
    camera_status_outgoing = queue.Queue()
    camera_output_outgoing = queue.Queue(maxsize=20)

    camera = Camera(capture_source, cancellation_event, capture_event, camera_status_outgoing, camera_output_outgoing)
    camera_thread = threading.Thread(target=camera.run)
    camera_thread.start()


    try:
        while True:

            # Simulate some processing, such as fetching frames and handling them.
            if not camera_output_outgoing.empty():
                frame = camera_output_outgoing.get()
                print(f"Received frame {frame[1]} at {frame[2]:.2f} FPS")
    except KeyboardInterrupt:
        print("Shutting down...")
        cancellation_event.set()
        camera_thread.join()

if __name__ == "__main__":

    main(0)
