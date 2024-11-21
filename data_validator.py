import cv2
import os


def read_data_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    data = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            print("Bad data")
            continue
        coords = list(map(int, parts[:-1]))
        try:
            image_name = parts[-1]
        except IndexError:
            print("Bad data")
            continue
        data.append((coords, image_name))
    return data


def draw_points(image, points):
    for i in range(0, len(points), 2):
        x = points[i]
        y = points[i + 1]
        point_number = i // 2  # Calculate the point number
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(image, str(point_number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return image


def main():
    # Get the input file path from the user
    file_path = "points.txt"  # input("Enter the path to the .txt file: ")
    data = read_data_file(file_path)
    if len(data) == 0:
        print("No data to review")
        quit()

    # Set the path to the folder containing images
    image_folder = "output_frames"

    index = 0
    while True:
        coords, image_name = data[index]
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not load image: {image_path}")
            index = (index + 1) % len(data)
            continue

        image_with_points = draw_points(image, coords)

        cv2.imshow("Image with Points", image_with_points)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break
        else:
            index = (index + 1) % len(data)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
