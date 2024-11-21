import cv2
import numpy as np
import glob
import os
import sys
import collections
import re

# Constants
MAX_POINTS = 7
KEYPOINT_COLOR = (255, 255, 0)
KEYPOINT_ALPHA = 0.30  # Transparency value (0.0 - fully transparent, 1.0 - fully opaque)
SIMILARITY_LOOKBACK_COUNT = 10  # Number of previous good frames to check similarity to
SIMILARITY_THRESHOLD = 0.99
POINT_HELP_TEXT = [
    "left eyelid corner",
    "upper left eyelid",
    "upper right eyelid",
    "right eyelid corner",
    "lower right eyelid",
    "lower left eyelid",
    "pupil center",
]
last_annotated_index_path = "last_annotated_index.txt"
points_path = "points.txt"  # Replace with path of existing file to save keypoint data to

# General variables
original_points = []
image_points = []
image_index = 0
keypoints_history = []
point_id = 0

# Rotation-related variable
scale_factor = 2  # Set the desired scale factor
rotation_degrees = 0
rotation_increment = 15
rotation_matrix = np.asarray([[1, 0, 0], [0, 1, 0]])
rotation_matrix_inv = np.asarray([[1, 0, 0], [0, 1, 0]])


def mouse_callback(event, x, y, flags, param):
    global original_points, image_points, point_id, rotation_matrix_inv

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(image_points) < MAX_POINTS:
            image_points.append((point_id, x, y))

            # Points in its non-rotated coordinates
            original_point = np.asarray([x, y])
            new_point = cv2.transform(original_point.reshape((-1, 1, 2)), rotation_matrix_inv)
            new_point = np.uint16(new_point[0][0])
            original_points.append((point_id, new_point[0], new_point[1]))

            point_id += 1

    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(image_points) > 0:
            distances = [((p[1] - x) ** 2 + (p[2] - y) ** 2) for p in image_points]
            closest_point_index = distances.index(min(distances))
            image_points[closest_point_index] = (image_points[closest_point_index][0], x, y)

            # Points in its non-rotated coordinates
            original_point = np.asarray([x, y])
            new_point = cv2.transform(original_point.reshape((-1, 1, 2)), rotation_matrix_inv)
            new_point = np.uint16(new_point[0][0])
            original_points[closest_point_index] = (image_points[closest_point_index][0], new_point[0], new_point[1])


def load_last_annotated_index(file_path):
    try:
        with open(file_path, "r") as file:
            return int(file.read())
    except FileNotFoundError:
        return 0


# Function to save the last annotated frame index to a file
def save_last_annotated_index(file_path, index):
    with open(file_path, "w") as file:
        file.write(str(index))


def extract_frames(video_path, output_folder):
    # # Get the base name of the video file without the extension
    # base_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create the output folder if it doesn't exist
    # output_folder = base_name + '_frames'
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get the total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Extracting {frame_count} frames from {video_path} at {fps} FPS")

    frame_number = 0
    frame_good = 0
    frame_deque = collections.deque(maxlen=SIMILARITY_LOOKBACK_COUNT)
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"Finished extracting frames. Total frames extracted: {frame_number}, out of which {frame_good} are saved.")
                break

            is_frame_good = True
            for frame_number_template, frame_template in frame_deque:
                if cv2.matchTemplate(frame, frame_template, cv2.TM_CCOEFF_NORMED).max() >= SIMILARITY_THRESHOLD:
                    print(f"Frame {frame_number} is too similar to {frame_number_template}, dropping..")
                    is_frame_good = False
                    break
            if is_frame_good or len(frame_deque) == 0:
                # Save the frame as a JPEG file
                frame_path = os.path.join(output_folder, f"{video_path}_frame_{frame_number:04d}.jpeg")
                cv2.imwrite(frame_path, frame)
                frame_deque.append((frame_number, frame))
                frame_good += 1
                print(f"Saved frame {frame_number}")

            frame_number += 1

        except Exception as e:
            print(f"Error processing frame {frame_number}: {e}")
            break

    # Release the video capture object
    cap.release()
    sys.stderr = sys.__stderr__
    print(f"Extraction complete. Frames saved in {output_folder}")


video_path = input("Give the name of the video you would like to annotate :> ")
output_folder = "output_frames"  # input("What would you like to name the output folder, or what is the name of the folder of pre extracted frames :> ")

resume_status = input(
    "Resume annotations where you left off? Y/N? (if you are doing a new video type N, this will not work properly if new files have been added) :> "
)
resume_status = resume_status.lower()
if resume_status == "y":
    image_index = load_last_annotated_index(last_annotated_index_path)
    print("resuming")

else:
    extract_frames(video_path, output_folder)


print("\n \nKeybinds:")
print("Press 'r' to clear the keypoints.")
print("Press 'w' to save the keypoints and move to the next image.")
print("Press 's' to save the keypoints and move to the previous image.")
print("Press 'q' to place the shadow keypoints to the current image \n")
print("Press 'd' to rotate image clockwise")
print("Press 'a' to rotate image counter-clockwise\n")
print("Press 'y' to close the program \n")

print("Mouse Buttons:")
print("Left click to draw keypoints.")
print("Right click to adjust point nearest to cursor to cursor position. \n \n")


image_folder = output_folder  # path with .jpg images
image_files = glob.glob(image_folder + "/*.jpeg")

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)


while True:
    # Init
    image_points = []
    original_points = []
    point_id = 0

    # Load image
    directory_path = os.path.dirname(image_files[image_index])
    file_name = image_files[image_index].replace(directory_path + os.path.sep, "")
    image = cv2.imread(image_files[image_index])
    print(f"Annotating {file_name}")

    # Load image points
    line = None
    with open(points_path, "r") as file:
        line = re.search(f"(^|\n).*{file_name}.*", file.read())
    if line:
        parts = line[0].strip().split()
        coords = list(map(int, parts[:-1]))
        for i in range(0, len(coords), 2):
            x = coords[i] * scale_factor
            y = coords[i + 1] * scale_factor
            original_points.append((point_id, x, y))

            original_point = np.asarray([x, y])
            new_point = cv2.transform(original_point.reshape((-1, 1, 2)), rotation_matrix)
            new_point = np.uint16(new_point[0][0])
            image_points.append((point_id, new_point[0], new_point[1]))
            point_id += 1

    # Draw keypoints history on original image
    for keypoints in keypoints_history:
        for point in keypoints:
            overlay = image.copy()
            pointx = int(point[1] / 2)
            pointy = int(point[2] / 2)
            cv2.circle(overlay, (pointx, pointy), 2, KEYPOINT_COLOR, -1)
            image = cv2.addWeighted(overlay, KEYPOINT_ALPHA, image, 1 - KEYPOINT_ALPHA, 0)

    while True:
        # Scale the image
        display_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        # Rotate image
        image_center = (display_image.shape[1] // 2, display_image.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, rotation_degrees, 1)
        rotation_matrix_inv = cv2.getRotationMatrix2D(image_center, -rotation_degrees, 1)
        display_image = cv2.warpAffine(display_image, rotation_matrix, display_image.shape[0:-1])

        # Point hint when putting down points
        if point_id < MAX_POINTS:
            cv2.putText(
                display_image,
                f"put down {point_id+1}: {POINT_HELP_TEXT[point_id]}",
                (5, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Keyboard shortcuts as footer
        text_footer = np.zeros((np.uint8(0.04 * display_image.shape[1]), display_image.shape[0], 3), np.uint8)
        cv2.putText(
            text_footer,
            "r - clear | w - next | s - prev | q - copy KPs | y - quit",
            (
                np.uint8(0.01 * display_image.shape[0]),
                np.uint8(text_footer.shape[0] / 2 + 0.009 * display_image.shape[1]),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (229, 229, 229),
            1,
        )
        display_image = cv2.vconcat([display_image, text_footer])

        # Draw image annotations
        for i, point in enumerate(image_points):
            draw_point = (point[1], point[2])
            cv2.circle(display_image, draw_point, 4, (0, 255, 0), -1)
            cv2.putText(
                display_image,
                str(i + 1),
                (draw_point[0], draw_point[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        cv2.imshow("Image", display_image)
        cv2.setWindowTitle("Image", f"Current image: {file_name}")
        key = cv2.waitKey(1) & 0xFF

        if key == ord("w") or key == ord("s"):
            if len(image_points) == MAX_POINTS:
                keypoints_history = [original_points.copy()]
                sorted_points = sorted(original_points, key=lambda x: x[0])

                # Print image points
                point_string = ",".join(
                    [
                        f"{min(int(point[1] / scale_factor), image.shape[1]-1)},{min(int(point[2] / scale_factor), image.shape[0]-1)}"
                        for point in sorted_points
                    ]
                )
                point_string = point_string.replace(",", " ")

                point_string = f"{point_string} {file_name}"
                with open(points_path, "r") as file:
                    file_content = file.read()
                    line = re.search(f"(^|\n).*{file_name}.*", file_content)
                if line:
                    with open(points_path, "w") as file:
                        file.write(re.sub(f"(^|\n).*{file_name}.*", "", file_content))
                with open(points_path, "a") as file:
                    file.write(point_string + "\n")
                print(f"[INFO] SAVED DATA FOR FRAME {file_name}")

            else:
                print(f"[INFO] SKIPPED FRAME {file_name}")

            save_last_annotated_index(last_annotated_index_path, image_index)

            if key == ord("w"):
                # Move to the next image
                image_index += 1
                image_index = image_index if image_index < len(image_files) else 0
                break
            elif key == ord("s"):
                # Move to previous image
                image_index -= 1
                image_index = image_index if image_index > 0 else len(image_files) - 1
                break

        if key == ord("r"):
            print("[INFO] CLEARED KEYPOINTS")
            image_points = []
            point_id = 0

        if key == ord("y"):
            print("[INFO] CLOSING")
            exit()

        if key == ord("a") or key == ord("d"):
            if key == ord("a"):
                rotation_degrees += rotation_increment
            elif key == ord("d"):
                rotation_degrees -= rotation_increment
            # Clamp to 0..359
            rotation_degrees = rotation_degrees if rotation_degrees >= 0 else 360 + rotation_degrees
            rotation_degrees = rotation_degrees if rotation_degrees < 360 else rotation_degrees - 360
            print(f"Rotation: {rotation_degrees}")
            rotation_matrix = cv2.getRotationMatrix2D(image_center, rotation_degrees, 1)
            rotation_matrix_inv = cv2.getRotationMatrix2D(image_center, -rotation_degrees, 1)
            new_points = []
            for point in original_points:
                original_point = np.asarray([point[1], point[2]])
                new_point = cv2.transform(original_point.reshape((-1, 1, 2)), rotation_matrix)
                new_point = np.uint16(new_point[0][0])
                new_points.append((point[0], new_point[0], new_point[1]))
            image_points = new_points
            new_points = []

        if key == ord("q"):
            if len(keypoints_history) > 0:
                original_points = keypoints_history[-1].copy()  # Copy the last set of keypoints from history
                new_points = []
                for point in original_points:
                    original_point = np.asarray([point[1], point[2]])
                    new_point = cv2.transform(original_point.reshape((-1, 1, 2)), rotation_matrix)
                    new_point = np.uint16(new_point[0][0])
                    new_points.append((point[0], new_point[0], new_point[1]))
                image_points = new_points
                point_id = MAX_POINTS
                print("[INFO] Placed keypoints from history")


cv2.destroyAllWindows()
