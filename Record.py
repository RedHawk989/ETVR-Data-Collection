import threading
import queue
import argparse
import cv2
from Camera import Camera, CameraState
import time
from colorama import Fore


def main(capture_source, eye, start_time):
    time_pre = 0
    cancellation_event = threading.Event()
    capture_event = threading.Event()
    camera_status_outgoing = queue.Queue()
    camera_output_outgoing = queue.Queue(maxsize=20)

    camera = Camera(capture_source, cancellation_event, capture_event, camera_status_outgoing, camera_output_outgoing)
    camera_thread = threading.Thread(target=camera.run)
    camera_thread.start()

    filename = f"output_{eye}.avi"
    output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MPEG'), 60, (240, 240))


    try:
        while True:
            if not camera_output_outgoing.empty():
                current_time = time.time()
                elapsed_time = current_time - start_time

                frame = camera_output_outgoing.get()
                image, frame_number, fps = frame

                if elapsed_time >2 and elapsed_time < 62:
                    output.write(image)
                    if int(elapsed_time) > int(time_pre):
                        time_pre = int(elapsed_time)
                        print(f"{Fore.GREEN}[INFO] Frame {frame_number} sent to queue Elapsed time: {int(elapsed_time) - 2}/60 seconds")


                if elapsed_time > 63: # wait 1 frame after video stops to fix stall bug
                    raise KeyboardInterrupt

                cv2.imshow('Camera Frame', image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("[INFO] Shutting down...")

        print("DM prohurtz the .avi files")
        cancellation_event.set()
        camera_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    src = input('Enter camera address then press enter. (ex COM5, 0, http://192.168.1.1) :>  ')
    eye = input('Enter if this is r for right cam and l for left cam then press enter. :>  ').lower()
    # Record the start time

    print ('Waiting for camera to warm up...')
    start_time = time.time()
    main(src, eye, start_time)
   # main("COM15", "r", start_time)